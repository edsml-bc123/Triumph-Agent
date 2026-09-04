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

import time
from loguru import logger

from client import DashScopeClient
from runtime.event import TrajectoryRecorder
from runtime.state import AgentState, AgentStatus
from tools.registry import ToolRegistry, default_registry

# 确保项目根目录在 sys.path 中
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
# ----------------------------------------------------------------------
# AgentLoop 核心执行引擎
# ----------------------------------------------------------------------

class AgentLoop:
    """
    自主 ReAct 执行循环引擎
    调度 client、state 与 tools，完成长期自主自治任务。
    """

    def __init__(
        self,
        client: Optional[DashScopeClient] = None,
        registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = client or DashScopeClient()
        self.registry = registry or default_registry
        self.system_prompt = system_prompt or self._default_system_prompt()

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
        :param state: 内存状态机实例
        :return: 达到终态（SUCCESS / FAILED）后的状态机
        """
        # 初始化轨迹记录器 (落盘至工作区的 runs/ 目录)
        runs_dir = self.registry.workdir / "runs"
        recorder = TrajectoryRecorder(run_id=state.run_id, runs_dir=runs_dir)

        # 1. 确保首条消息包含系统指令 (System Prompt)
        if not any(m.get("role") == "system" for m in state.messages):
            state.messages.insert(0, {"role": "system", "content": self.system_prompt})

        # 提取用户初始 prompt 并记录任务启动事件
        user_prompt = ""
        for m in state.messages:
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break
        recorder.record_task_start(prompt=user_prompt, max_steps=state.max_steps)

        # 2. 核心自主循环：只要未达到终态，持续推进
        while not state.is_terminal:
            # 步数安全递增与硬熔断判定
            if not state.increment_step():
                logger.warning(f"[CircuitBreaker] {state.last_error}")
                break

            recorder.record_step_start(state.step_count)
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

            # 记录大模型推理事件
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            recorder.record_llm_response(
                content=response.content,
                finish_reason=response.finish_reason,
                tool_calls=response.tool_calls,
                usage=usage_dict,
            )

            # 3. 终态判断分支 (Termination Decision)
            if not response.has_tool_calls:
                # 模型未调用任何工具，给出了直接回答
                # 在当前 V0 阶段：代表模型认为任务已完成，正常达标退出
                state.mark_success(response.content)
                logger.success(f"[Assistant Answer]\n{response.content}")
                break

            # 4. 批量工具派发与协议闭环执行 (Batch Tool Execution)
            for tc in response.tool_calls:
                # 安全反序列化参数（内置 json_repair 容错）
                try:
                    args = tc.parse_arguments()
                except Exception as parse_err:
                    # 参数完全不可解析时的防御回填
                    error_msg = f"Error: Failed to parse tool arguments: {parse_err}"
                    state.add_tool_result(tc.id, tc.name, error_msg)
                    logger.warning(f"[Tool Parse Error] {tc.name} -> {error_msg}")
                    continue

                logger.info(f"[Tool Call] {tc.name} | args={args}")

                # 本地安全沙箱派发执行并度量耗时
                tool_start = time.time()
                output = self.registry.execute(tc.name, args)
                tool_duration_ms = (time.time() - tool_start) * 1000

                preview = output[:200].replace("\n", " ")
                if len(output) > 200:
                    preview += "..."
                logger.debug(f"[Tool Result] {tc.name} -> {preview}")

                # 记录工具执行事件
                recorder.record_tool_execution(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    args=args,
                    output=output,
                    duration_ms=tool_duration_ms,
                )

                # 将物理结果以标准 role: "tool" 绑定唯一 tool_call_id 回填
                state.add_tool_result(tool_call_id=tc.id, name=tc.name, result=output)

        # 记录任务终态事件
        recorder.record_task_end(
            status=state.status.value,
            total_steps=state.step_count,
            total_tokens=state.total_tokens,
            final_answer=state.final_answer,
            error=state.last_error,
        )

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
