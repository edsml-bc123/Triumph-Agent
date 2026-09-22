"""
triumph-agent 核心执行循环引擎 (Agent Loop)
==============================================
核心设计思想：
1. 遵循 learn-claude-code 的精简内循环设计（The pattern is small）：
   状态机驱动，保持循环控制流高内聚、职责单一；
2. 批量工具调用与结果汇聚 (Batch Execution)：
   单轮次中处理模型发起的全部 tool_calls，保持协议完整闭环；
3. 透明的可观测性终端反馈 (Observability)：
   清晰高亮展示 Agent 的思考、工具调用与物理执行回执；
4. 异常隔离与安全硬熔断：
   以 state.is_terminal 作为唯一生命周期守卫，彻底消灭死循环。
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

import time
from loguru import logger

from client import DashScopeClient
from runtime.event import TrajectoryHook
from runtime.hooks import HookManager
from runtime.state import AgentState, AgentStatus, current_run_id_var
from tools.registry import ToolRegistry, default_registry
# ----------------------------------------------------------------------
# AgentLoop 核心执行引擎
# ----------------------------------------------------------------------

class AgentLoop:
    """
    自主 ReAct 执行循环引擎
    调度 client、state 与 tools，并通过 HookManager 驱动生命周期切面 (AOP)。
    """

    def __init__(
        self,
        client: Optional[DashScopeClient] = None,
        registry: Optional[ToolRegistry] = None,
        hooks: Optional[HookManager] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = client or DashScopeClient()
        self.registry = registry or default_registry
        self.system_prompt = system_prompt or self._default_system_prompt()

        # 实例级 Hook 事件总线 (默认自动挂载 TrajectoryHook 轨迹插件)
        if hooks is not None:
            self.hooks = hooks
        else:
            self.hooks = HookManager()
            runs_dir = self.registry.workdir / "runs"
            TrajectoryHook(runs_dir=runs_dir).register_to(self.hooks)

    def _default_system_prompt(self) -> str:
        return (
            f"你是一个拥有自主代码编写和系统操作能力的 Coding Agent，当前工作目录是: {self.registry.workdir}。\n"
            "你可以使用提供的工具（bash、read_file、write_file、edit_file、glob）来分析和解决问题。\n"
            "原则：\n"
            "1. 务实求真，用实际操作和工具验证结果，不要口头编造；\n"
            "2. 当需要修改代码或执行命令时，主动调用工具；\n"
            "3. 任务彻底解决后，给出简明扼要的总结。"
        )

    async def run(self, state: AgentState) -> AgentState:
        """
        驱动 AgentState 在自主 ReAct 循环中向前流转，直至终态。
        通过统一 trigger 接口广播生命周期切面事件。
        :param state: 内存状态机实例
        :return: 达到终态（SUCCESS / FAILED）后的状态机
        """
        # 1. 确保首条消息包含系统指令 (System Prompt)
        if not any(m.get("role") == "system" for m in state.messages):
            state.messages.insert(0, {"role": "system", "content": self.system_prompt})

        # 绑定当前异步协程级任务 ID 上下文 (Ambient Context)
        token = current_run_id_var.set(state.run_id)

        try:
            # 提取用户当前最新 prompt 并触发 UserPromptSubmit 切面
            user_prompt = ""
            for m in reversed(state.messages):
                if m.get("role") == "user":
                    user_prompt = m.get("content", "")
                    break
            await self.hooks.trigger("UserPromptSubmit", prompt=user_prompt, state=state)
            if state.is_terminal:
                logger.warning(f"[CircuitBreaker] 任务提交切面触发终态: {state.last_error}")
                return state

            # 2. 核心自主循环：只要未达到终态，持续推进
            while not state.is_terminal:
                # 步数安全递增与硬熔断判定
                if not state.increment_step():
                    logger.warning(f"[CircuitBreaker] {state.last_error}")
                    break

                await self.hooks.trigger("StepStart", step=state.step_count, state=state)
                if state.is_terminal:
                    logger.warning(f"[CircuitBreaker] 轮次启动切面阻断执行: {state.last_error}")
                    break

                logger.info(f"[Loop Step {state.step_count}/{state.max_steps}] 模型推理中...")

                # 向大模型发起推理请求（注入可用工具 Schema）
                try:
                    response = await self.client.chat_completion(
                        messages=state.messages,
                        tools=self.registry.get_tools_spec(),
                    )
                except Exception as e:
                    state.mark_failed(f"通信异常：大模型 API 请求中断: {e}")
                    logger.error(f"[API Error] {state.last_error}")
                    break

                # 将大模型本轮响应结构化写入状态机（累加 Token，记录 tool_calls）
                state.add_assistant_turn(response)

                # 触发 LLM 响应切面
                await self.hooks.trigger("LLMResponse", response=response, state=state)
                if state.is_terminal:
                    logger.warning(f"[CircuitBreaker] LLM 响应后切面触发熔断: {state.last_error}")
                    break

                # 3. 终态判断分支 (Termination Decision)
                if not response.has_tool_calls:
                    # 模型未调用任何工具，给出了直接回答
                    # 仅在未处于失败等终态的前提下，正常达标退出并记录为 SUCCESS
                    if not state.is_terminal:
                        state.mark_success(response.content)
                    logger.success(f"[Assistant Answer]\n{response.content}")
                    break

                # 4. 批量工具派发与协议闭环执行 (Batch Tool Execution)
                for tc in response.tool_calls:
                    # 物理执行前置门禁：若前序工具调用已触发终态，终止后续派发
                    if state.is_terminal:
                        logger.warning(f"[CircuitBreaker] 批处理工具执行被阻断，终止后续派发: {state.last_error}")
                        break

                    # 安全反序列化参数（内置 json_repair 容错）
                    try:
                        args = tc.parse_arguments()
                    except Exception as parse_err:
                        # 参数完全不可解析时的防御回填
                        error_msg = f"Error: Failed to parse tool arguments: {parse_err}"
                        state.add_tool_result(tc.id, tc.name, error_msg)
                        logger.warning(f"[Tool Parse Error] {tc.name} -> {error_msg}")
                        continue

                    # 核心拦截点：PreToolUse 权限与安全审查 (非 None 即阻断)
                    blocked_reason = await self.hooks.trigger(
                        "PreToolUse", tool_name=tc.name, args=args, state=state
                    )
                    if state.is_terminal:
                        logger.warning(f"[CircuitBreaker] PreToolUse 切面触发终态熔断: {state.last_error}")
                        break

                    if blocked_reason is not None:
                        logger.warning(f"[Tool Blocked] {tc.name} -> {blocked_reason}")
                        # 被切面阻断时，直接回填拒绝原因，不派发物理执行
                        state.add_tool_result(tool_call_id=tc.id, name=tc.name, result=f"Error: {blocked_reason}")
                        continue

                    logger.info(f"[Tool Call] {tc.name} | args={args}")

                    # 本地安全沙箱派发执行并度量耗时
                    tool_start = time.time()
                    output = await self.registry.execute(tc.name, args)
                    tool_duration_ms = (time.time() - tool_start) * 1000

                    preview = output[:200].replace("\n", " ")
                    if len(output) > 200:
                        preview += "..."
                    logger.debug(f"[Tool Result] {tc.name} -> {preview}")

                    # 触发 PostToolUse 切面（耗时、落盘、脱敏）
                    await self.hooks.trigger(
                        "PostToolUse",
                        tool_name=tc.name,
                        args=args,
                        output=output,
                        duration_ms=tool_duration_ms,
                        tool_call_id=tc.id,
                        state=state,
                    )
                    if state.is_terminal:
                        logger.warning(f"[CircuitBreaker] PostToolUse 切面触发终态熔断: {state.last_error}")
                        break

                    # 将物理结果以标准 role: "tool" 绑定唯一 tool_call_id 回填
                    state.add_tool_result(tool_call_id=tc.id, name=tc.name, result=output)

        except Exception as e:
            # 异常兜底：防止未捕获系统级异常逃逸导致状态机处于悬空状态
            state.mark_failed(f"执行循环崩溃中断: {e}")
            logger.error(f"[Runtime Crash] {state.last_error}")
            raise
        finally:
            try:
                # 无论正常退出、切面熔断还是致命崩溃，100% 触发 Stop 终态切面，确保黑匣子 TaskEnd 必然落盘
                await self.hooks.trigger("Stop", state=state)
            finally:
                current_run_id_var.reset(token)

        return state


# ----------------------------------------------------------------------
# 模块自测：完整真实的自主多轮任务演练
# ----------------------------------------------------------------------

async def _smoke_test():
    import tempfile
    print("启动 runtime/loop.py 自主 ReAct 循环实战自测...")

    # 使用临时安全沙箱目录
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox_dir = Path(tmpdir)
        registry = ToolRegistry(workdir=sandbox_dir)
        loop_engine = AgentLoop(registry=registry)

        # 构造一个需要自主执行 2~3 轮工具调用的复杂任务
        state = AgentState(max_steps=10)
        task_prompt = (
            "请完成以下操作：\n"
            "1. 使用 write_file 工具在当前目录创建一个 'agent_test.txt'，写入 'Phase 1 Complete'；\n"
            "2. 使用 read_file 工具读取该文件，验证内容是否正确；\n"
            "3. 确认无误后，向我汇报文件中的确切内容。"
        )
        state.add_user_message(task_prompt)

        print(f"\n📋 [任务输入]:\n{task_prompt}\n")
        final_state = await loop_engine.run(state)

        # 验证自主执行的终态与各轮历史
        print("=" * 60)
        print("🔍 验证执行结果与状态机体征:")
        print(f"- Run ID:         {final_state.run_id}")
        print(f"- 最终状态:       {final_state.status.value}")
        print(f"- 迭代步数:       {final_state.step_count} 轮")
        print(f"- 累计消耗 Token: {final_state.total_tokens}")
        print(f"- 历史消息总数:   {len(final_state.messages)} 条")

        assert final_state.status == AgentStatus.SUCCESS
        assert (sandbox_dir / "agent_test.txt").exists()
        assert (sandbox_dir / "agent_test.txt").read_text().strip() == "Phase 1 Complete"
        print("\n✅ runtime/loop.py 自主 ReAct 多轮循环全流程验证成功！")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
