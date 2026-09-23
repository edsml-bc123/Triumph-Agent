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

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from client import LLMResponse
    from runtime.hooks import HookManager
    from runtime.state import AgentState

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
        base_runs_dir = (runs_dir or (Path.cwd() / "runs")).resolve()
        # 方案 A：轨迹文件与压缩产物共同落入 runs/{run_id}/ 专属子目录中
        self.run_dir = base_runs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.run_dir / "trajectory.jsonl"

    def record(self, event: str, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        单一统一记录方法 (Single Universal Record Method)：
        所有轨迹事件统一通过此方法追加落盘，保证东八区时间与实时 flush。
        :param event: 事件名称 (如 TaskStart, StepStart, LLMResponse, ToolExecution, TaskEnd, CircuitBreak 等)
        :param data: 可选的字典载荷
        :param kwargs: 任意键值对参数，自动合并入载荷
        """
        payload = dict(data or {})
        payload.update(kwargs)

        now_cst = datetime.now(CST)
        event_entry = TrajectoryEvent(
            event_type=event,
            run_id=self.run_id,
            timestamp=now_cst.timestamp(),
            cst_time=now_cst.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " CST",
            data=payload,
        )
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(event_entry.to_json() + "\n")
            f.flush()


# ----------------------------------------------------------------------
# 解耦切面插件：运行轨迹实时落盘 (TrajectoryHook)
# ----------------------------------------------------------------------

class TrajectoryHook:
    """
    运行轨迹切面插件 (基于 HookManager 统一驱动)
    职责：监听生命周期切面事件，统一调用 TrajectoryRecorder.record 实施持久化落盘。
    """

    def __init__(self, runs_dir: Optional[Path] = None):
        self.runs_dir = runs_dir
        self._recorder: Optional[TrajectoryRecorder] = None

    def register_to(self, manager: HookManager) -> None:
        """显式向 HookManager 注册生命周期回调"""
        manager.register("UserPromptSubmit", self.on_user_prompt_submit)
        manager.register("StepStart", self.on_step_start)
        manager.register("LLMResponse", self.on_llm_response)
        manager.register("PostToolUse", self.on_post_tool_use)
        manager.register("Stop", self.on_stop)

    async def on_user_prompt_submit(self, prompt: str, state: AgentState) -> None:
        self._recorder = TrajectoryRecorder(run_id=state.run_id, runs_dir=self.runs_dir)
        self._recorder.record("TaskStart", user_prompt=prompt, max_steps=state.max_steps)

    async def on_step_start(self, state: AgentState) -> None:
        if self._recorder:
            self._recorder.record("StepStart", step=state.step_count)

    async def on_llm_response(self, response: LLMResponse, state: AgentState) -> None:
        if self._recorder and getattr(response, "usage", None):
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            tool_calls = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments_raw}
                for tc in getattr(response, "tool_calls", [])
            ]
            self._recorder.record(
                "LLMResponse",
                content=response.content,
                finish_reason=response.finish_reason,
                tool_calls=tool_calls,
                usage=usage_dict,
            )

    async def on_post_tool_use(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: str,
        duration_ms: float,
        tool_call_id: str,
        state: AgentState,
    ) -> None:
        if self._recorder:
            self._recorder.record(
                "ToolExecution",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=args,
                output_preview=output[:1000],
                output_length=len(output),
                duration_ms=round(duration_ms, 2),
            )

    async def on_stop(self, state: AgentState) -> None:
        if self._recorder:
            self._recorder.record(
                "TaskEnd",
                status=state.status.value,
                total_steps=state.step_count,
                total_tokens=state.total_tokens,
                final_answer=state.final_answer,
                error=state.last_error,
            )


