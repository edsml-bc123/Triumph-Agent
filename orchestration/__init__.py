"""
triumph-agent 多智能体与编排系统 (Orchestration Engine)
======================================================
包含：
1. SubAgentTool: 纯净上下文子智能体复合工具插件
2. 后续阶段逐步扩展: background (长任务), dag (任务图编排), team & worktree (战队协同)
"""

from .job import (
    BackgroundJob,
    JobHook,
    JobManager,
    JobStatus,
    JobTool,
)
from .dag import (
    DAGTask,
    DAGTaskStatus,
    DAGTaskTool,
    TaskStore,
)
from .subagent import (
    SubAgentTool,
)

__all__ = [
    "SubAgentTool",
    "BackgroundJob",
    "JobHook",
    "JobManager",
    "JobTool",
    "JobStatus",
    "DAGTask",
    "DAGTaskStatus",
    "TaskStore",
    "DAGTaskTool",
]