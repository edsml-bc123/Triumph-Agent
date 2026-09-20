"""
triumph-agent 生命周期钩子 (HookManager) 自动化测试
===================================================
验证目标：
1. 单一 register 接口与 @manager.on 装饰器挂载；
2. 同步 def 与异步 async def 协程透明驱动；
3. 短路拦截核心契约 (Non-None Return Blocks Pipeline)；
4. 全流程放行返回 None 契约。
"""

import pytest
from runtime.hooks import HookManager


@pytest.mark.asyncio
async def test_hooks_registration_and_execution():
    manager = HookManager()
    events_received = []

    # 1. 注册同步函数
    def sync_callback(step: int):
        events_received.append(f"sync_{step}")

    manager.register("StepStart", sync_callback)

    # 2. 用装饰器注册异步协程
    @manager.on("StepStart")
    async def async_callback(step: int):
        events_received.append(f"async_{step}")

    # 触发事件
    res = await manager.trigger("StepStart", step=1)
    assert res is None
    assert events_received == ["sync_1", "async_1"]


@pytest.mark.asyncio
async def test_hooks_short_circuit_behavior():
    manager = HookManager()
    call_log = []

    def first_hook(tool_name: str, **kwargs):
        call_log.append("first")
        return None  # 放行

    def blocking_hook(tool_name: str, **kwargs):
        call_log.append("blocking")
        if tool_name == "bash":
            return "Blocked by security rule"
        return None

    def third_hook(tool_name: str, **kwargs):
        call_log.append("third")
        return None

    # 依次挂载 3 个钩子
    manager.register("PreToolUse", first_hook)
    manager.register("PreToolUse", blocking_hook)
    manager.register("PreToolUse", third_hook)

    # 触发 bash 工具调用（应被第 2 个拦截）
    res_block = await manager.trigger("PreToolUse", tool_name="bash", command="rm -rf")
    assert res_block == "Blocked by security rule"
    # 验证短路：第 3 个钩子绝不会被执行
    assert call_log == ["first", "blocking"]

    # 触发 safe 工具调用（全部放行）
    call_log.clear()
    res_allow = await manager.trigger("PreToolUse", tool_name="read_file", path="a.txt")
    assert res_allow is None
    assert call_log == ["first", "blocking", "third"]


@pytest.mark.asyncio
async def test_hooks_register_plugin():
    manager = HookManager()

    # 定义一个符合 register_to 协议的插件
    class DummyPlugin:
        def __init__(self):
            self.triggered = False

        def register_to(self, hook_mgr):
            hook_mgr.register("Stop", self.on_stop)

        async def on_stop(self, **kwargs):
            self.triggered = True

    plugin = DummyPlugin()
    manager.register_plugin(plugin)

    # 触发并验证插件被成功调度
    await manager.trigger("Stop")
    assert plugin.triggered is True

    # 验证非法插件未实现协议时抛出 TypeError
    class InvalidPlugin:
        pass

    with pytest.raises(TypeError):
        manager.register_plugin(InvalidPlugin())


@pytest.mark.asyncio
async def test_loop_pre_tool_use_terminal_guard(tmp_path):
    """验证 PreToolUse 切面将状态置为终态时，AgentLoop 立即中止且不执行工具"""
    from client import LLMResponse, ToolCall
    from runtime.loop import AgentLoop
    from runtime.state import AgentState, AgentStatus
    from tools.registry import ToolRegistry

    class MockClient:
        async def chat_completion(self, *args, **kwargs):
            return LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments_raw='{"path": "never.txt", "content": "blocked"}',
                    )
                ],
            )

    manager = HookManager()

    # 注册一个在 PreToolUse 阶段直接熔断状态机的钩子
    @manager.on("PreToolUse")
    async def fatal_guard(tool_name, args, state, **kwargs):
        state.mark_failed("致命安全违规，紧急下线")

    registry = ToolRegistry(workdir=tmp_path)
    loop_engine = AgentLoop(client=MockClient(), registry=registry, hooks=manager)

    state = AgentState()
    state.add_user_message("触发致命违规测试")

    final_state = await loop_engine.run(state)
    assert final_state.is_terminal is True
    assert final_state.status == AgentStatus.FAILED
    assert "致命安全违规" in final_state.last_error
    assert not (tmp_path / "never.txt").exists()


@pytest.mark.asyncio
async def test_loop_crash_guarantees_stop_and_task_end(tmp_path):
    """
    验证系统级未捕获异常发生时：
    1. 异常不被静默吞掉，正常向上 raise；
    2. state 被置为 FAILED 终态；
    3. finally 块绝对保障 Stop 切面被 100% 触发；
    4. TrajectoryHook 正确闭环落盘 TaskEnd，杜绝悬空轨迹文件。
    """
    import json
    from runtime.event import TrajectoryHook
    from runtime.loop import AgentLoop
    from runtime.state import AgentState, AgentStatus
    from tools.registry import ToolRegistry

    class DummyClient:
        async def chat_completion(self, *args, **kwargs):
            raise NotImplementedError("不应执行到模型调用阶段")

    manager = HookManager()
    runs_dir = tmp_path / "runs"
    traj_hook = TrajectoryHook(runs_dir=runs_dir)
    manager.register_plugin(traj_hook)

    stop_called = False

    @manager.on("Stop")
    async def track_stop(state, **kwargs):
        nonlocal stop_called
        stop_called = True

    # 模拟在 StepStart 切面发生未被捕获的严重系统级异常
    @manager.on("StepStart")
    async def buggy_plugin(step, **kwargs):
        raise RuntimeError("第三方插件或底层运行时发生未捕获致命异常！")

    registry = ToolRegistry(workdir=tmp_path)
    loop_engine = AgentLoop(client=DummyClient(), registry=registry, hooks=manager)

    state = AgentState()
    state.add_user_message("测试循环崩溃守护")

    # 1. 验证异常向上传递，不被静默吞掉
    with pytest.raises(RuntimeError, match="第三方插件或底层运行时发生未捕获致命异常！"):
        await loop_engine.run(state)

    # 2. 验证状态机被标记为 FAILED，且包含根本原因信息
    assert state.is_terminal is True
    assert state.status == AgentStatus.FAILED
    assert "执行循环崩溃中断" in state.last_error
    assert "未捕获致命异常" in state.last_error

    # 3. 验证 Stop 钩子被 100% 触发
    assert stop_called is True

    # 4. 验证黑匣子轨迹文件是否完整闭环且含有 TaskEnd
    run_file = runs_dir.resolve() / state.run_id / "trajectory.jsonl"
    assert run_file.exists()
    lines = [json.loads(line) for line in run_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) >= 2
    assert lines[0]["event_type"] == "TaskStart"
    
    # 核心契约：即使因崩溃抛出异常，最后一条日志也必须是 TaskEnd
    last_event = lines[-1]
    assert last_event["event_type"] == "TaskEnd"
    assert last_event["data"]["status"] == "failed"
    assert "执行循环崩溃中断" in last_event["data"]["error"]

