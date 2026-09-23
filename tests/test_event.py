"""
triumph-agent 轨迹记录系统 (Trajectory Engineering) 自动化测试
==============================================================
验证目标：
1. TrajectoryRecorder 单一 record 接口规范；
2. CST 东八区纳秒时间戳持久化正确性；
3. JSONL 流式追加落盘文件完整性；
4. TrajectoryHook 插件与 HookManager 联动全生命周期录制。
"""

import json
import tempfile
from pathlib import Path
import pytest

from runtime.event import TrajectoryRecorder, TrajectoryHook
from runtime.hooks import HookManager
from runtime.state import AgentState


def test_trajectory_recorder_single_record_api():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = TrajectoryRecorder(run_id="run_test_001", runs_dir=Path(tmpdir))

        # 验证单一 record 接口调用多种类型事件
        recorder.record("TaskStart", user_prompt="分析代码", max_steps=10)
        recorder.record("StepStart", step=1)
        recorder.record("ToolExecution", tool_name="read_file", args={"path": "a.txt"}, output="ok")
        recorder.record("TaskEnd", status="success", total_steps=1)

        # 读取落盘文件验证
        jsonl_path = recorder.file_path
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4

        # 验证事件字段与东八区标记
        first_event = json.loads(lines[0])
        assert first_event["event_type"] == "TaskStart"
        assert first_event["run_id"] == "run_test_001"
        assert "CST" in first_event["cst_time"]
        assert first_event["data"]["user_prompt"] == "分析代码"


@pytest.mark.asyncio
async def test_trajectory_hook_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir)
        hooks = HookManager()

        # 挂载 TrajectoryHook
        traj_hook = TrajectoryHook(runs_dir=runs_dir)
        traj_hook.register_to(hooks)

        state = AgentState(run_id="hook_record_001")
        # 触发完整生命周期事件
        await hooks.trigger("UserPromptSubmit", prompt="测试提示词", state=state)
        await hooks.trigger("StepStart", state=state)
        await hooks.trigger(
            "PostToolUse",
            tool_name="bash",
            args={"command": "ls"},
            output="file.txt",
            duration_ms=12.5,
            tool_call_id="call_99",
            state=state,
        )
        state.mark_success("任务完成")
        await hooks.trigger("Stop", state=state)

        record_file = runs_dir.resolve() / "hook_record_001" / "trajectory.jsonl"
        assert record_file.exists()
        lines = record_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4

        events = [json.loads(line)["event_type"] for line in lines]
        assert events == ["TaskStart", "StepStart", "ToolExecution", "TaskEnd"]
