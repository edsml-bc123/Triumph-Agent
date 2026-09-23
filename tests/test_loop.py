"""
tests/test_loop.py - AgentLoop 核心执行循环专属自动化测试
=========================================================
全面验证：
1. 模型直接给出文本回答时，状态机无缝达标 SUCCESS 并提取 final_answer；
2. 多轮自主工具调用（write_file -> read_file）端到端物理派发闭环；
3. max_steps 步数超限硬熔断守卫（防死循环机制）；
4. 幻觉工具调用容错：未知工具名以语义自愈回填，主循环绝不崩溃；
5. 自定义 system_prompt 正确注入首条消息。
"""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from client import LLMResponse, ToolCall
from runtime.hooks import HookManager
from runtime.loop import AgentLoop
from runtime.state import AgentState, AgentStatus
from tools import BuiltinToolsPlugin, ToolRegistry


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.default_model = "qwen3.8-flash"
    client.evaluator_model = "qwen3.6-flash"
    client.chat_completion = AsyncMock()
    return client


@pytest.fixture
def env(tmp_path):
    """标准测试基础设施：纯净注册表 + 内置基础工具 + 空白钩子总线"""
    registry = ToolRegistry(workdir=tmp_path)
    registry.register_plugin(BuiltinToolsPlugin(workdir=tmp_path))
    hooks = HookManager()
    return tmp_path, registry, hooks


@pytest.mark.asyncio
async def test_loop_direct_answer_success(mock_client, env):
    """测试模型直接输出答案时状态机正常迁移到 SUCCESS"""
    workdir, registry, hooks = env
    loop = AgentLoop(registry=registry, client=mock_client, hooks=hooks)

    mock_client.chat_completion.return_value = LLMResponse(
        content="今天天气晴朗，适合外出。",
        finish_reason="stop",
        tool_calls=[],
    )

    state = AgentState()
    state.add_user_message("今天天气怎么样？")

    final_state = await loop.run(state)

    assert final_state.status == AgentStatus.SUCCESS
    assert final_state.step_count == 1
    assert final_state.final_answer == "今天天气晴朗，适合外出。"
    assert mock_client.chat_completion.call_count == 1


@pytest.mark.asyncio
async def test_loop_multi_tool_execution(mock_client, env):
    """测试多轮 ReAct 工具调用流：write_file -> read_file -> 最终汇报"""
    workdir, registry, hooks = env
    loop = AgentLoop(registry=registry, client=mock_client, hooks=hooks)

    # 模拟两轮工具调用 + 一轮终态文本回答
    turn1 = LLMResponse(
        content="我先写入文件",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="tc_write_1",
                name="write_file",
                arguments_raw=json.dumps({"path": "output.txt", "content": "Hello Agent Loop"}),
            )
        ],
    )
    turn2 = LLMResponse(
        content="我来验证读取内容",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="tc_read_1",
                name="read_file",
                arguments_raw=json.dumps({"path": "output.txt"}),
            )
        ],
    )
    turn3 = LLMResponse(
        content="文件已成功写入并验证，内容确为: Hello Agent Loop",
        finish_reason="stop",
        tool_calls=[],
    )
    mock_client.chat_completion.side_effect = [turn1, turn2, turn3]

    state = AgentState()
    state.add_user_message("创建 output.txt 并读取核对")

    final_state = await loop.run(state)

    assert final_state.status == AgentStatus.SUCCESS
    assert final_state.step_count == 3
    assert (workdir / "output.txt").exists()
    assert (workdir / "output.txt").read_text() == "Hello Agent Loop"
    assert "Hello Agent Loop" in (final_state.final_answer or "")


@pytest.mark.asyncio
async def test_loop_max_steps_hard_circuit_breaker(mock_client, env):
    """测试当模型无限发起工具调用时，达到 max_steps 主动硬熔断防死循环"""
    workdir, registry, hooks = env
    loop = AgentLoop(registry=registry, client=mock_client, hooks=hooks)

    # 模拟模型每一轮都无休止调用工具
    infinite_tool_response = LLMResponse(
        content="继续探测",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="tc_repeat",
                name="glob",
                arguments_raw=json.dumps({"pattern": "*"}),
            )
        ],
    )
    mock_client.chat_completion.return_value = infinite_tool_response

    state = AgentState(max_steps=3)
    state.add_user_message("进行无限探索")

    final_state = await loop.run(state)

    assert final_state.status == AgentStatus.FAILED
    assert final_state.step_count == 3
    assert "达到最大上限" in (final_state.last_error or "")



@pytest.mark.asyncio
async def test_loop_unknown_tool_resilience(mock_client, env):
    """测试模型调用不存在的工具时，系统安全回填错误且循环不崩溃"""
    workdir, registry, hooks = env
    loop = AgentLoop(registry=registry, client=mock_client, hooks=hooks)

    turn1 = LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="tc_phantom_1",
                name="phantom_tool",
                arguments_raw=json.dumps({"param": "val"}),
            )
        ],
    )
    turn2 = LLMResponse(
        content="已知 phantom_tool 不可用，我直接给出回答。",
        finish_reason="stop",
        tool_calls=[],
    )
    mock_client.chat_completion.side_effect = [turn1, turn2]

    state = AgentState()
    state.add_user_message("调用不存在的工具")

    final_state = await loop.run(state)

    assert final_state.status == AgentStatus.SUCCESS
    assert final_state.step_count == 2
    # 验证中间有错误回填消息
    tool_result_msg = next(
        m for m in final_state.messages
        if m.get("role") == "tool" and m.get("tool_call_id") == "tc_phantom_1"
    )
    assert "Unknown tool 'phantom_tool'" in tool_result_msg["content"]


@pytest.mark.asyncio
async def test_loop_custom_system_prompt(mock_client, env):
    """测试显式传入自定义 system_prompt 覆盖默认系统提示词"""
    workdir, registry, hooks = env
    custom_prompt = "你是一个安全审计专家，严密审查每一项指令。"
    loop = AgentLoop(
        registry=registry,
        client=mock_client,
        hooks=hooks,
        system_prompt=custom_prompt,
    )

    mock_client.chat_completion.return_value = LLMResponse(
        content="审计通过。",
        finish_reason="stop",
        tool_calls=[],
    )

    state = AgentState()
    state.add_user_message("请进行审计")

    final_state = await loop.run(state)

    assert final_state.messages[0]["role"] == "system"
    assert final_state.messages[0]["content"] == custom_prompt
