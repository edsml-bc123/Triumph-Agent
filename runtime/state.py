"""
triumph-agent 状态机与上下文模型 (State Contract)
===================================================
核心设计思想：
1. 显式状态机 (Explicit State Machine)：
   清晰定义 Agent 生命周期的全部离散状态，杜绝隐式状态判断。
2. 协议严谨的消息闭环 (Protocol Integrity)：
   严格遵循 OpenAI / 阿里云百炼 Chat Completion 消息协议规范，
   自动处理 assistant tool_calls 与 tool 返回结果的配对，防止报文组装错误。
3. 资源度量与安全硬熔断 (Safety & Circuit Breaking)：
   跟踪 step_count 与 total_tokens，在触碰上限时主动熔断，彻底根治大模型死循环。
4. 向前兼容架构 (Forward Compatible)：
   吸收 learn-claude-code s17 思想，内建 Goal 目标属性与 Stop 拦截计数器 (consecutive_blocks)。
"""

import sys
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# 异步协程级当前运行任务 ID 上下文变量 (Ambient Run Context，用于跨层透传与树状派生溯源)
current_run_id_var: ContextVar[Optional[str]] = ContextVar("current_run_id", default=None)

# 确保当前项目根目录在 sys.path 中，便于直接调用 client 模块
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from client import LLMResponse


from datetime import datetime, timedelta, timezone

# 显式定义东八区时区 (UTC+8 / Asia/Shanghai)
CST = timezone(timedelta(hours=8))


# ----------------------------------------------------------------------
# 1. 状态机枚举定义 (Lifecycle Status)
# ----------------------------------------------------------------------

class AgentStatus(str, Enum):
    """
    Agent 运行时的显式生命周期状态枚举
    """
    PENDING = "pending"                  # 初始化完成，等待用户提示词或启动信号
    RUNNING = "running"                  # 正在与大模型通信或等待推理返回
    TOOL_EXECUTING = "tool_executing"    # 正在执行外部工具（Bash / 文件读写）
    BLOCKED = "blocked"                  # 被裁判模型或安全策略拦截，需自愈重试
    SUCCESS = "success"                  # 任务正常达成（终态）
    FAILED = "failed"                    # 出现不可恢复异常或步数超限熔断（终态）


# ----------------------------------------------------------------------
# 2. 状态机核心模型 (AgentState)
# ----------------------------------------------------------------------

@dataclass
class AgentState:
    """
    Agent 运行时的中央状态模型 (Single Source of Truth)
    负责统一维护消息流、运行度量、状态转移与终态控制。
    """
    # 会话与唯一标识（严格采用东八区时间戳）
    run_id: str = field(
        default_factory=lambda: f"run_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    parent_run_id: Optional[str] = None  # 父级运行标识（若由 Subagent 派生则关联父任务，根任务为 None）
    status: AgentStatus = AgentStatus.PENDING

    # 消息上下文流 (严格遵循 OpenAI / 百炼格式)
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # 运行度量与防死循环安全熔断
    step_count: int = 0                  # 当前 ReAct 循环步数
    max_steps: int = 20                  # 单次运行最大允许执行步数（硬熔断）
    total_tokens: int = 0                # 累计消耗的 Token 数量
    budget_warned: bool = False          # 是否已触发过 Token 预算警戒预警（单任务仅提示一次）

    # 异常捕获与诊断信息
    last_error: Optional[str] = None
    final_answer: Optional[str] = None

    # 向前兼容：为后续阶段（s17 Goal Loop）预留的验收与拦截槽位
    current_goal: Optional[str] = None   # 明确的完成条件描述
    consecutive_blocks: int = 0          # 连续被裁判模型拦截的次数
    max_consecutive_blocks: int = 5      # 连续被拦截的最大上限（防止无休止空转）

    # ------------------------------------------------------------------
    # 状态转移原子方法 (State Transition Methods)
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        """追加一条用户消息，并将状态置为 RUNNING"""
        self.messages.append({"role": "user", "content": content})
        self.status = AgentStatus.RUNNING

    def add_assistant_turn(self, response: LLMResponse) -> None:
        """
        根据底层 client 返回的 LLMResponse 规范化追加 assistant 消息。
        自动组装标准 tool_calls 报文，并累加 Token 消耗。
        """
        # 1. 累加 Token 开销
        if response.usage:
            self.total_tokens += response.usage.total_tokens

        # 2. 构造 assistant 消息字典
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
        }

        # 3. 如果大模型触发了工具调用，组装标准的 tool_calls 协议字段
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments_raw,
                    },
                }
                for tc in response.tool_calls
            ]
            if not self.is_terminal:
                self.status = AgentStatus.TOOL_EXECUTING
        else:
            if not self.is_terminal:
                self.final_answer = response.content

        self.messages.append(assistant_msg)

    def add_tool_result(self, tool_call_id: str, name: str, result: str) -> None:
        """
        将工具执行的物理结果标准格式回填给大模型。
        严格绑定对应 tool_call_id，保证协议闭环。
        """
        tool_msg: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": str(result),
        }
        self.messages.append(tool_msg)
        # 终态单向流转守卫：若已经处于终态（如切面熔断），严禁将其复活为 RUNNING
        if not self.is_terminal:
            self.status = AgentStatus.RUNNING

    def increment_step(self) -> bool:
        """
        循环步数递增与硬熔断判定。
        :return: True 表示步数正常未超限；False 表示已触碰硬上限并触发熔断
        """
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.mark_failed(
                f"安全熔断：当前步数 [{self.step_count}] 达到最大上限 [{self.max_steps}]，强制终止。"
            )
            return False
        return True

    def block_turn(self, reason: str) -> bool:
        """
        [s17 Goal 拦截机制] 裁判模型认为本轮未达成目标，拦截退出。
        追加一条提示给大模型，要求其继续提供铁证。
        :return: True 表示允许继续重试；False 表示连续拦截次数超限，触发熔断
        """
        if self.is_terminal:
            return False

        self.consecutive_blocks += 1
        self.status = AgentStatus.BLOCKED

        if self.consecutive_blocks >= self.max_consecutive_blocks:
            self.mark_failed(
                f"连续阻止超限：裁判模型已连续拦截 [{self.consecutive_blocks}] 次，目标未达成，归还控制权。"
            )
            return False

        # 注入具象负反馈，促使大模型在下一轮循环中纠偏
        self.messages.append({
            "role": "user",
            "content": (
                f"【目标尚未达成 (第 {self.consecutive_blocks}/{self.max_consecutive_blocks} 次拦截)】\n"
                f"当前目标: {self.current_goal or '未显式指定'}\n"
                f"裁判理由: {reason}\n"
                f"请继续执行必要工具并提供真实的执行铁证。"
            ),
        })
        self.status = AgentStatus.RUNNING
        return True

    def mark_success(self, final_text: Optional[str] = None) -> None:
        """将状态标记为成功终态（终态单向流转，已处于失败等终态时不可覆盖）"""
        if self.is_terminal:
            return
        self.status = AgentStatus.SUCCESS
        if final_text is not None:
            self.final_answer = final_text

    def mark_failed(self, error_msg: str) -> None:
        """将状态标记为失败终态并记录错误（已处于任意终态时均不可逆）"""
        if self.is_terminal:
            return
        self.status = AgentStatus.FAILED
        self.last_error = error_msg

    @property
    def is_terminal(self) -> bool:
        """
        终态判定：Agent 是否已经结束生命周期。
        后续循环引擎只需要一行: while not state.is_terminal: 即可驱动。
        """
        return self.status in (AgentStatus.SUCCESS, AgentStatus.FAILED)

    def to_dict(self) -> Dict[str, Any]:
        """将当前内存状态序列化为字典，便于写入 runs/{id}.jsonl 或断点保存"""
        data = asdict(self)
        data["status"] = self.status.value
        return data


