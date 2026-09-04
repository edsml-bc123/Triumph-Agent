"""
triumph-agent 运行轨迹事件记录系统 (Trajectory Engineering)
============================================================
核心设计思想：
1. 黑匣子实时追加落盘 (Streaming JSONL Append)：
   每个任务独立分配 runs/{run_id}.jsonl 文件，事件发生即刻追加并 flush，
   保证哪怕遭遇系统断电或崩溃，案发现场数据依然 100% 完整保留；
2. 结构化轨迹事件 (Structured Trajectory Events)：
   记录 TaskStart、StepStart、LLMResponse、ToolExecution、TaskEnd 五大核心生命周期事件；
3. 为断点恢复与自动化评测赋能：
   提供离线重放（Replay）、Benchmark 评测与 Prompt 调优所需的第一手物理凭据。
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

# 显式定义东八区时区 (UTC+8 / Asia/Shanghai)
CST = timezone(timedelta(hours=8))


class EventType(str, Enum):
    """轨迹事件类型枚举"""
    TASK_START = "task_start"
    STEP_START = "step_start"
    LLM_RESPONSE = "llm_response"
    TOOL_EXECUTION = "tool_execution"
    CIRCUIT_BREAK = "circuit_break"
    TASK_END = "task_end"


@dataclass
class TrajectoryEvent:
    """单条轨迹事件数据结构"""
    event_type: str
    run_id: str
    timestamp: float
    cst_time: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type,
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "cst_time": self.cst_time,
                "data": self.data,
            },
            ensure_ascii=False,
        )


class TrajectoryRecorder:
    """
    运行轨迹记录器 (JSONL Streaming Logger)
    """

    def __init__(self, run_id: str, runs_dir: Optional[Path] = None):
        self.run_id = run_id
        self.runs_dir = (runs_dir or (Path.cwd() / "runs")).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.runs_dir / f"{run_id}.jsonl"

    def record(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """
        实时追加写入一条轨迹事件，并强制刷盘 (严格记录东八区时间)
        """
        now_cst = datetime.now(CST)
        event = TrajectoryEvent(
            event_type=event_type.value,
            run_id=self.run_id,
            timestamp=now_cst.timestamp(),
            cst_time=now_cst.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " CST",
            data=data,
        )
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
            f.flush()

    def record_task_start(self, prompt: str, max_steps: int) -> None:
        self.record(
            EventType.TASK_START,
            {"user_prompt": prompt, "max_steps": max_steps},
        )

    def record_step_start(self, step_count: int) -> None:
        self.record(
            EventType.STEP_START,
            {"step": step_count},
        )

    def record_llm_response(
        self,
        content: str,
        finish_reason: str,
        tool_calls: list,
        usage: Dict[str, int],
    ) -> None:
        self.record(
            EventType.LLM_RESPONSE,
            {
                "content": content,
                "finish_reason": finish_reason,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments_raw,
                    }
                    for tc in tool_calls
                ],
                "usage": usage,
            },
        )

    def record_tool_execution(
        self,
        tool_call_id: str,
        tool_name: str,
        args: Dict[str, Any],
        output: str,
        duration_ms: float,
    ) -> None:
        self.record(
            EventType.TOOL_EXECUTION,
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": args,
                "output_preview": output[:1000],  # 记录适度长度的输出快照
                "output_length": len(output),
                "duration_ms": round(duration_ms, 2),
            },
        )

    def record_task_end(
        self,
        status: str,
        total_steps: int,
        total_tokens: int,
        final_answer: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.record(
            EventType.TASK_END,
            {
                "status": status,
                "total_steps": total_steps,
                "total_tokens": total_tokens,
                "final_answer": final_answer,
                "error": error,
            },
        )


# ----------------------------------------------------------------------
# 模块自测
# ----------------------------------------------------------------------

def _smoke_test():
    import tempfile
    print("启动 runtime/event.py 冒烟自测...")

    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = TrajectoryRecorder(run_id="test_run_001", runs_dir=Path(tmpdir))
        recorder.record_task_start("测试任务", 10)
        recorder.record_step_start(1)
        recorder.record_tool_execution("call_1", "bash", {"command": "ls"}, "file1 file2", 15.3)
        recorder.record_task_end("success", 1, 100, "完成")

        assert recorder.file_path.exists()
        lines = recorder.file_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4
        first_event = json.loads(lines[0])
        assert first_event["event_type"] == "task_start"
        assert first_event["data"]["user_prompt"] == "测试任务"
        print(f"轨迹文件生成成功: {len(lines)} 条事件已持久化落盘")

    print("runtime/event.py 冒烟自测全部通过！")


if __name__ == "__main__":
    _smoke_test()
