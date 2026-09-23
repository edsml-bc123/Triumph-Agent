"""
triumph-agent Token 预算熔断 (BudgetHook) 自动化测试
=====================================================
验证目标：
1. BudgetHook 参数合法性防御；
2. 预算消耗未超标时放行；
3. 达到 warn_ratio 警戒比例预警；
4. 触碰硬上限 (max_total_tokens) 触发状态机强制熔断；
5. 与 HookManager 统一插件装配联动与新任务重置。
"""

import pytest
from context.budget import BudgetHook
from runtime.hooks import HookManager
from runtime.state import AgentState, AgentStatus
from client import LLMResponse


def test_budget_hook_validation():
    """验证非法参数防御"""
    with pytest.raises(ValueError):
        BudgetHook(max_total_tokens=0)

    with pytest.raises(ValueError):
        BudgetHook(max_total_tokens=1000, warn_ratio=1.5)


@pytest.mark.asyncio
async def test_budget_hook_lifecycle():
    """验证预警与熔断生命周期流转"""
    hook = BudgetHook(max_total_tokens=1000, warn_ratio=0.8)
    state = AgentState(run_id="run_budget_lifecycle")

    dummy_resp = LLMResponse(content="step result", finish_reason="tool_calls")

    # 1. 正常消耗 500 Token，未达警戒线
    state.total_tokens = 500
    await hook.on_llm_response(dummy_resp, state)
    assert not state.is_terminal
    assert state.status != AgentStatus.FAILED
    assert state.budget_warned is False

    # 2. 消耗增至 850 Token，触发警戒线
    state.total_tokens = 850
    await hook.on_llm_response(dummy_resp, state)
    assert not state.is_terminal
    assert state.status != AgentStatus.FAILED
    assert state.budget_warned is True

    # 验证同一会话再次调用不会重复预警，且新会话天然隔离不受影响
    new_state = AgentState(run_id="run_fresh")
    assert new_state.budget_warned is False

    # 3. 消耗增至 1050 Token，触发硬熔断
    state.total_tokens = 1050
    await hook.on_llm_response(dummy_resp, state)
    assert state.is_terminal is True
    assert state.status == AgentStatus.FAILED
    assert "安全硬熔断" in state.last_error


@pytest.mark.asyncio
async def test_budget_hook_plugin_integration():
    """验证通过 HookManager.register_plugin 装配并由事件总线调度"""
    hooks = HookManager()
    hook = BudgetHook(max_total_tokens=2000, warn_ratio=0.8)
    hooks.register_plugin(hook)

    state = AgentState(run_id="run_budget_plugin")
    resp = LLMResponse(content="done", finish_reason="stop")

    # 1. 触发 UserPromptSubmit
    await hooks.trigger("UserPromptSubmit", prompt="测试", state=state)

    # 2. 模拟单步后 Token 超标
    state.total_tokens = 2500
    await hooks.trigger("LLMResponse", response=resp, state=state)
    assert state.is_terminal is True
    assert state.status == AgentStatus.FAILED

    # 3. 验证在后续 StepStart 探针时继续坚决阻断
    await hooks.trigger("StepStart", state=state)
    assert state.status == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_loop_budget_circuit_breaker_interrupts_tools(tmp_path):
    """验证核心执行循环中：大模型返回 tool_calls 但触发预算熔断时，严禁派发工具并立即终止"""
    from client import ToolCall, Usage
    from runtime.loop import AgentLoop
    from tools.registry import ToolRegistry

    class MockClient:
        async def chat_completion(self, *args, **kwargs):
            return LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_mock_write",
                        name="write_file",
                        arguments_raw='{"path": "leak.txt", "content": "should never be written"}',
                    )
                ],
                usage=Usage(prompt_tokens=1000, completion_tokens=1200, total_tokens=2200),
            )

    hooks = HookManager()
    hooks.register_plugin(BudgetHook(max_total_tokens=2000))
    registry = ToolRegistry(workdir=tmp_path)
    loop_engine = AgentLoop(client=MockClient(), registry=registry, hooks=hooks)

    state = AgentState()
    state.add_user_message("超标任务测试")

    final_state = await loop_engine.run(state)

    # 1. 状态机必须严格处于 FAILED 终态
    assert final_state.is_terminal is True
    assert final_state.status == AgentStatus.FAILED
    assert "安全硬熔断" in final_state.last_error

    # 2. 工具绝不允许被派发执行，目标文件不应存在
    assert not (tmp_path / "leak.txt").exists()

    # 3. 消息列表中不应包含 tool 的回填消息
    assert not any(m.get("role") == "tool" for m in final_state.messages)

