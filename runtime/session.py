"""
triumph-agent 会话生命周期与作用域管理器 (Session Context & Lifecycle)
===================================================================
核心设计思想：
1. 一等公民 Session：
   - 统领跨多轮 ReAct 运行（Run）的人机交互整体会话周期；
   - 彻底收敛工作区散落的临时产物，统一组织在 .sessions/{session_id}/ 下。
2. 目录规范分工：
   - tasks_dir (.sessions/{session_id}/tasks):
     会话独享的活动 DAG 任务看板，同一个会话内多次输入/多轮 Run 共享同一拓扑状态；
   - runs_dir (.sessions/{session_id}/runs):
     会话各轮 ReAct 循环的事后审计轨迹流水，每轮分配独立 run_id 子目录。
3. 物理会话轮转：
   - 当用户在终端输入 /clear 重置会话时，安全封存旧 Session 目录，即刻创建全新的会话作用域。
"""

from __future__ import annotations

import contextvars
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

# 统一东八区时区
CST = timezone(timedelta(hours=8))

# 会话上下文变量，支持异步环境跟踪
current_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_session_id", default=None
)


class SessionContext:
    """
    单次长会话运行时上下文实体
    """

    def __init__(self, workdir: Path, base_dir_name: str = ".sessions", session_id: Optional[str] = None):
        self.workdir = workdir.resolve()
        self.base_sessions_dir = self.workdir / base_dir_name
        self.base_sessions_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or self._generate_session_id()
        self.created_at = datetime.now(CST)

        # 绑定 ContextVar
        current_session_id_var.set(self.session_id)

        # 初始化会话专属目录
        self._ensure_directories()
        logger.info("初始化 Session 作用域 | Session ID: {} | 根路径: {}", self.session_id, self.session_dir)

    @staticmethod
    def _generate_session_id() -> str:
        """生成具备时间轴可读性与唯一性的会话 ID"""
        ts = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
        rand = uuid.uuid4().hex[:6]
        return f"session_{ts}_{rand}"

    @property
    def session_dir(self) -> Path:
        """当前会话的物理工作目录"""
        return self.base_sessions_dir / self.session_id

    @property
    def tasks_dir(self) -> Path:
        """当前会话的 DAG 需求拓扑任务看板目录"""
        return self.session_dir / "tasks"

    @property
    def runs_dir(self) -> Path:
        """当前会话的历次 ReAct 循环审计轨迹流水目录"""
        return self.session_dir / "runs"

    def _ensure_directories(self) -> None:
        """确保会话级目录物理存在"""
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> SessionContext:
        """
        轮转创建全新的 Session 作用域，彻底重置任务白板与运行轨迹
        """
        old_id = self.session_id
        self.session_id = self._generate_session_id()
        self.created_at = datetime.now(CST)
        current_session_id_var.set(self.session_id)
        self._ensure_directories()

        logger.info("已轮转归档旧会话 [{}]，开启全新会话作用域 [{}]", old_id, self.session_id)
        return self
