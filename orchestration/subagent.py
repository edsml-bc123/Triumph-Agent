"""
triumph-agent 子智能体调度与物理上下文隔离引擎 (Subagent Engine)
================================================================
核心哲学与设计思想（对标 learn-claude-code s06_subagent）：
1. 纯净上下文隔离 (Fresh Context Isolation)：
   - 子智能体从全新的 messages=[] 开始执行，绝不继承主 Agent 的长历史；
   - 子任务内部产生的大量密集工具调用（如阅读数十个源码、多次搜索）完全留存在子上下文内部；
   - 执行完成后，仅将提炼后的最终文本 (final_answer) 作为 ToolResult 返回给父 Agent，主上下文零污染。
2. 递归深度防御门禁 (Anti-Recursion Gate)：
   - 派生给子智能体的工具集（create_subagent_registry）必须严格剥离 subagent/task 工具本身；
   - 杜绝子智能体无限派生套娃引发死锁与 Token 熔断。
3. 统一工具插件化装配 (Tool Plugin Protocol)：
   - 实现 register_to(registry) 契约，与 HookManager.register_plugin 高度对齐；
   - 外部调用只需传入唯一必需的 client 依赖，工作区沙箱与运行目录从 registry 自动继承推导。
"""

import time
import uuid
from pathlib import Path
from typing import Any, Optional
from loguru import logger

from client import DashScopeClient
from context.compactor import CompactorHook, ContextCompactor, CompactionConfig
from runtime.hooks import HookManager
from runtime.loop import AgentLoop
from runtime.state import AgentState, AgentStatus, current_run_id_var
from runtime.event import TrajectoryHook
from security.permission import PermissionHook
from tools.registry import ToolRegistry


SUBAGENT_SYSTEM_PROMPT = (
    "你是由主智能体委派的专属子智能体 (Subagent)。\n"
    "你的职责是在当前工作区内专注于完成指定的单一子任务（如深入阅读分析文件、检索定位代码、或执行具体自测验证）。\n"
    "执行原则：\n"
    "1. 充分利用可用工具完成事实探查或任务推进；\n"
    "2. 任务达成后，提供客观、精炼、信息密度高的总结性回答，不要包含与最终结论无关的琐碎过程；\n"
    "3. 你的输出将作为工具调用结果直接反馈给主智能体。"
)


class SubAgentTool:
    """
    子智能体工具插件 (符合 register_to 契约规范)
    """

    name = "subagent"
    description = (
        "派生一个拥有全新独立上下文的子智能体 (Subagent) 去执行专注的子任务（如密集阅读代码、检索调用链、运行自测等），"
        "子智能体的繁杂中间过程被完全隔离，仅返回精炼的最终文本结果。有效防止主对话上下文被巨量中间工具结果撑爆。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "要委托给子智能体执行的具体任务提示词，需清晰描述目标与期望返回的信息。",
            },
            "max_steps": {
                "type": "integer",
                "description": "子智能体执行的最大步数限制，缺省为 10。",
            },
        },
        "required": ["prompt"],
    }

    def __init__(
        self,
        client: DashScopeClient,
        default_max_steps: int = 10,
        workdir: Optional[Path] = None,
        runs_dir: Optional[Path] = None,
    ):
        self.client = client
        self.default_max_steps = default_max_steps
        self._workdir = workdir.resolve() if workdir else None
        self._runs_dir = runs_dir.resolve() if runs_dir else None
        self.parent_registry: Optional[ToolRegistry] = None

    @property
    def workdir(self) -> Path:
        """工作区沙箱路径：优先使用显式指定值，缺省自动继承父注册表工作区"""
        if self._workdir is not None:
            return self._workdir
        if self.parent_registry is not None:
            return self.parent_registry.workdir
        raise RuntimeError("SubAgentTool 尚未通过 register_to(registry) 装配，缺少父级工具表上下文。")

    @property
    def runs_dir(self) -> Path:
        """运行轨迹落盘目录：优先使用显式指定值，缺省自动在 workdir 下构建 runs/"""
        if self._runs_dir is not None:
            return self._runs_dir
        return (self.workdir / "runs").resolve()

    def register_to(self, registry: ToolRegistry) -> None:
        """
        统一挂载到 ToolRegistry (对齐 HookPlugin 协议)
        """
        self.parent_registry = registry
        registry.register(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self._dispatch_handler,
        )

    async def _dispatch_handler(self, prompt: str, max_steps: Optional[int] = None, **kwargs: Any) -> str:
        return await self.run(prompt=prompt, max_steps=max_steps)

    async def run(
        self,
        prompt: str,
        max_steps: Optional[int] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        """
        派生并执行一个拥有全新独立上下文的子智能体任务。

        :param prompt: 委托给子智能体的具体指令
        :param max_steps: 允许子智能体执行的最大迭代步数
        :param parent_run_id: 父级任务标识（用于链路溯源）
        :return: 子智能体精炼的最终结论文本
        """
        steps_limit = max_steps or self.default_max_steps

        # 1. 自动穿透感知主 Agent 运行上下文 (优先使用显式参数，其次通过 ContextVar 异步无感继承)
        effective_parent_id = parent_run_id or current_run_id_var.get()

        # 生成子任务专属 ID (格式: sub_{uuid[:6]} 或带时间戳)
        sub_run_id = f"sub_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # 2. 推导树状分层运行目录 (Hierarchical Run Directory)
        # 若存在主任务父级上下文，自动收敛为: runs/{parent_run_id}/subagents/{sub_id}/
        # 否则回退为顶级平铺: runs/{sub_id}/
        if effective_parent_id:
            sub_base_runs_dir = (self.runs_dir / effective_parent_id / "subagents").resolve()
        else:
            sub_base_runs_dir = self.runs_dir

        logger.info(
            f"[Subagent Started] 启动子智能体 | SubRunID: {sub_run_id} | ParentRunID: {effective_parent_id} | 步数上限: {steps_limit} | 任务: {prompt[:60]}..."
        )

        # 3. 严格派生隔离的工具集（精准剥离自身 1 个工具，杜绝递归死锁）
        sub_registry = self.parent_registry.fork(exclude={self.name})

        # 4. 为子智能体装配独立的轻量切面总线
        sub_hooks = HookManager()
        # 4.1 独立黑匣子轨迹记录 (收拢至 sub_base_runs_dir / sub_run_id / trajectory.jsonl)
        sub_hooks.register_plugin(TrajectoryHook(runs_dir=sub_base_runs_dir))
        # 4.2 静默权限控制
        sub_hooks.register_plugin(PermissionHook(workdir=self.workdir, interactive=False))
        # 4.3 轻量渐进式压缩防护 (仅开启 Layer 1 单工具大输出截断，严防读大文件撑爆 Token，关闭耗时 L5 重型折叠)
        sub_compactor_cfg = CompactionConfig(
            max_single_tool_output_chars=20_000,
            preview_chars=1200,
            hard_threshold_chars=999_999_999,  # 子任务生命周期聚焦，不进行全局 LLM 摘要折叠
        )
        sub_compactor = ContextCompactor(
            workdir=self.workdir,
            base_runs_dir=sub_base_runs_dir,
            config=sub_compactor_cfg,
        )
        sub_hooks.register_plugin(CompactorHook(compactor=sub_compactor))

        # 5. 初始化子智能体专属状态机（Fresh Context，零历史消息残留）
        sub_state = AgentState(
            run_id=sub_run_id,
            parent_run_id=effective_parent_id,
            max_steps=steps_limit,
        )
        sub_state.messages.append({"role": "system", "content": SUBAGENT_SYSTEM_PROMPT})
        sub_state.add_user_message(prompt)
        sub_state.current_goal = prompt

        # 4. 驱动嵌套子执行循环
        sub_loop = AgentLoop(
            client=self.client,
            registry=sub_registry,
            hooks=sub_hooks,
        )

        start_time = time.time()
        try:
            await sub_loop.run(sub_state)
        except Exception as e:
            sub_state.mark_failed(f"子智能体运行时异常: {e}")
            logger.error(f"[Subagent Error] 执行异常中断: {e}")

        duration = time.time() - start_time

        # 5. 结果提炼与安全回填
        if sub_state.status == AgentStatus.SUCCESS:
            answer = (sub_state.final_answer or "").strip()
            if not answer:
                answer = "子智能体已成功完成工具操作，但未给出具体陈述文本。"
            logger.success(
                f"[Subagent Done] 子任务成功完成 | 耗时: {duration:.2f}s | 消耗Token: {sub_state.total_tokens} | 步数: {sub_state.step_count}"
            )
            return answer
        else:
            err = sub_state.last_error or "未知原因中断"
            logger.warning(
                f"[Subagent Aborted] 子任务未能正常完成 ({sub_state.status.value}) | 原因: {err}"
            )
            return f"Subagent execution ended without success ({sub_state.status.value}): {err}"
