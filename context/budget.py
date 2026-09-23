"""
triumph-agent 上下文与 Token 预算熔断引擎 (Token Budget Manager)
================================================================
核心设计思想（吸收路线图阶段二设计与成本工程哲学）：
1. 资费失控硬熔断 (Cost Circuit Breaker)：
   - 设定单次任务最大允许消耗的 Token 硬上限 (max_total_tokens)；
   - 一旦触碰硬上限，立即通过 state.mark_failed 终止循环，彻底杜绝大模型无限刷 Token；
2. 渐进式预警体系 (Progressive Warning)：
   - 达到警戒线（默认 80%）时触发高亮审计预警，为阶段三自动触发上下文压缩（Compactor）预留契约槽位；
3. 纯粹插件契约 (Plugin Architecture)：
   - 继承统一插件装配协议 register_to(manager)，零侵入挂载在 LLMResponse 与 StepStart 切面上。
"""

import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from loguru import logger
from client import LLMResponse
from runtime.state import AgentState, AgentStatus


class BudgetHook:
    """
    Token 预算硬上限管控与熔断切面 (Budget Hook)
    """

    def __init__(
        self,
        max_total_tokens: int = 100_000,
        warn_ratio: float = 0.8,
    ):
        """
        :param max_total_tokens: 单次任务允许消耗的 Token 硬上限（默认 10 万）
        :param warn_ratio: 预算警戒比例（默认 80% 触发预警）
        """
        if max_total_tokens <= 0:
            raise ValueError("max_total_tokens 必须为大于 0 的正整数")
        if not (0.0 < warn_ratio < 1.0):
            raise ValueError("warn_ratio 必须介于 0.0 与 1.0 之间")

        self.max_total_tokens = max_total_tokens
        self.warn_ratio = warn_ratio
        self.warn_threshold = int(self.max_total_tokens * self.warn_ratio)

    def register_to(self, manager: Any) -> None:
        """
        实现插件装配协议：将自身注册到 LLMResponse 和 StepStart 节点
        """
        manager.register("LLMResponse", self.on_llm_response)
        manager.register("StepStart", self.on_step_start)

    async def on_llm_response(self, response: LLMResponse, state: AgentState) -> None:
        """
        大模型返回后核查累计 Token 开销：
        1. 达到预警阈值时输出警报；
        2. 达到或超出硬上限时立即切断状态机（触发熔断）。
        """
        current_tokens = state.total_tokens

        # 1. 预算超标硬熔断
        if current_tokens >= self.max_total_tokens:
            reason = (
                f"安全硬熔断：当前累计消耗 Token [{current_tokens}] 达到或超过硬上限 [{self.max_total_tokens}]，"
                "强制终止执行以防止资费失控。"
            )
            logger.error(f"[Budget Circuit Breaker] {reason}")
            state.mark_failed(reason)
            return

        # 2. 达到警戒线预警 (依托 state.budget_warned 强类型属性，单会话仅警示一次且随任务自然销毁)
        if current_tokens >= self.warn_threshold and not state.budget_warned:
            state.budget_warned = True
            ratio_pct = (current_tokens / self.max_total_tokens) * 100
            logger.warning(
                f"[Budget Warning] Token 消耗已达预算警戒线 ({ratio_pct:.1f}%) | "
                f"当前消耗: {current_tokens} / 上限: {self.max_total_tokens}"
            )

    async def on_step_start(self, step: int, state: AgentState) -> None:
        """
        单步迭代前探针：若已经处于终态或 Token 已超限，确保阻断
        """
        if state.total_tokens >= self.max_total_tokens and not state.is_terminal:
            state.mark_failed(
                f"安全硬熔断：Token 消耗已超标 [{state.total_tokens} >= {self.max_total_tokens}]，终止新轮次。"
            )


