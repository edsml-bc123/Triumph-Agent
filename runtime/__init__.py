"""
triumph-agent 核心执行运行时 (Runtime Engine)
"""

from .state import AgentState, AgentStatus, current_run_id_var, current_state_var
from .loop import AgentLoop
from .event import TrajectoryRecorder, EventType, TrajectoryHook
from .hooks import HookManager, HookPlugin
from .session import SessionContext, current_session_id_var
from .terminal import (
    disable_bracketed_paste,
    enable_bracketed_paste,
    read_terminal_input,
)

__all__ = [
    "AgentState",
    "AgentStatus",
    "current_run_id_var",
    "current_state_var",
    "current_session_id_var",
    "SessionContext",
    "AgentLoop",
    "TrajectoryRecorder",
    "EventType",
    "HookManager",
    "HookPlugin",
    "TrajectoryHook",
    "enable_bracketed_paste",
    "disable_bracketed_paste",
    "read_terminal_input",
]

