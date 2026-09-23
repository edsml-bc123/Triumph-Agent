"""
triumph-agent 交互式 CLI 终端入口 (Main Entrypoint)
=====================================================
双循环架构中的【外层会话循环 (Outer Session Loop)】：
1. 负责监听人类控制台输入，支持交互式对话与快捷退出；
2. 保持跨任务的工具环境沙箱与底层长连接复用；
3. 使用 loguru 进行结构化日志记录与可观测性管理。
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# 确保项目根目录在 sys.path 中
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# 尝试加载 readline 优化终端输入
try:
    import readline
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from client import DashScopeClient
from context import BudgetHook, CompactorHook, ContextCompactor, CompactionConfig, CompactTool
from memory import MemoryHook, MemoryManager
from orchestration import (
    BackgroundTaskHook,
    BackgroundTaskManager,
    BackgroundTaskTool,
    SubAgentTool,
)
from runtime import AgentLoop, AgentState, AgentStatus, HookManager, TrajectoryHook
from security import PermissionHook
from tools import BuiltinToolsPlugin, ToolRegistry


def print_banner():
    banner = (
        "\n" + "=" * 68 + "\n"
        "  triumph-agent v0.2 (Mini Coding Agent Runtime)\n"
        "  基于阿里云百炼原生协议 + 显式状态机 + 权限审计 + 预算熔断 + 渐进式压缩 + 三层记忆\n"
        + "=" * 68 + "\n"
        "提示: 输入任务指令（如：“查看当前目录下的文件并统计数量”），按回车执行。\n"
        "提示: 支持自动派生子智能体 (subagent) 隔离处理复杂子探索。\n"
        "提示: 支持长耗时命令后台执行 (bash run_in_background=True) 与主动唤醒通知。\n"
        "提示: 输入 /clear 或 clear 可重置当前会话历史。\n"
        "提示: 输入 /memory 或 memory 可查看当前长期记忆索引。\n"
        "提示: 输入 q 或 exit 退出程序。\n"
    )
    print(banner)


async def main():
    print_banner()

    # 初始化全局基础设施（保持连接池复用与工作区绑定）
    workdir = Path.cwd()
    async with DashScopeClient() as client:
        registry = ToolRegistry(workdir=workdir)
        bg_manager = BackgroundTaskManager(workdir=workdir, runs_dir=workdir / "runs")

        # 工业级生产梯度参数：
        # - L1: 单工具输出 >20,000 字符自动落盘截断并保留 1200 字符预览
        # - L2: 对话历史 >30 条消息自动执行中段成对归档
        # - L3/L4: 上下文 >35,000 字符（约 10k tokens）触发陈旧工具微压缩与弹性拟合，更早削减 Token 膨胀
        # - L5: 上下文 >80,000 字符触发终极全局语义摘要折叠
        # - 复杂任务预算硬上限放宽至 250,000 tokens
        compactor_cfg = CompactionConfig(
            max_single_tool_output_chars=20_000,
            preview_chars=1200,
            max_history_messages=30,
            keep_recent_tool_results=2,
            soft_threshold_chars=35_000,
            hard_threshold_chars=80_000,
        )
        compactor = ContextCompactor(workdir=workdir, config=compactor_cfg)

        # 显式装配基础工具、后台管控、子智能体与主动上下文压缩插件 (ToolPlugin 协议)
        registry.register_plugin(BuiltinToolsPlugin(workdir=workdir, bg_manager=bg_manager))
        registry.register_plugin(BackgroundTaskTool(manager=bg_manager))
        registry.register_plugin(SubAgentTool(client=client))
        registry.register_plugin(CompactTool(compactor=compactor, client=client))

        memory_mgr = MemoryManager(workdir=workdir)

        hooks = HookManager()
        hooks.register_plugin(TrajectoryHook(runs_dir=workdir / "runs"))
        hooks.register_plugin(PermissionHook(workdir=workdir, interactive=True))
        hooks.register_plugin(BudgetHook(max_total_tokens=250000))
        hooks.register_plugin(CompactorHook(compactor=compactor, client=client))
        hooks.register_plugin(MemoryHook(manager=memory_mgr, client=client))
        hooks.register_plugin(BackgroundTaskHook(manager=bg_manager))

        loop_engine = AgentLoop(client=client, registry=registry, hooks=hooks)

        # 外层会话循环 (Outer Loop) - 维护连续的多轮会话历史 (Session Context)
        session_messages: list[dict] = []

        while True:
            try:
                user_input = input("\ntriumph >> ").strip()
            except (EOFError, KeyboardInterrupt):
                logger.info("接收到中断信号，triumph-agent 正在安全退出。")
                break

            if not user_input:
                continue

            if user_input.lower() in ("q", "quit", "exit"):
                logger.info("用户主动退出，再见！")
                break

            if user_input.lower() in ("/clear", "clear"):
                session_messages.clear()
                logger.info("已重置当前会话上下文，开启全新任务对话。")
                continue

            if user_input.lower() in ("/memory", "memory"):
                index_text = memory_mgr.storage.read_index()
                if index_text:
                    print(f"\n[当前长期记忆目录]:\n{index_text}")
                else:
                    print("\n[当前长期记忆目录为空]")
                continue

            # 为当前任务初始化独立的 AgentState，并继承会话级多轮历史
            state = AgentState(max_steps=20)
            if session_messages:
                state.messages.extend(session_messages)
            state.add_user_message(user_input)
            state.current_goal = user_input

            logger.info("启动自主执行任务 | Run ID: {} | 携带历史上下文: {} 条消息", state.run_id, len(session_messages))
            start_time = asyncio.get_event_loop().time()

            try:
                # 唤醒内层 ReAct 自主执行循环
                await loop_engine.run(state)
            except Exception as e:
                state.mark_failed(f"未知异常中断: {e}")
                logger.error("运行时出现未捕获异常: {}", e)

            elapsed = asyncio.get_event_loop().time() - start_time

            # 结构化输出执行报告与会话状态延续
            if state.status == AgentStatus.SUCCESS:
                # 成功后将本轮产生的最新上下文平滑继承至会话历史中
                session_messages = list(state.messages)
                logger.success(
                    "任务执行成功 | 步数: {} | 耗时: {:.2f}s | Token开销: {} | 当前会话上下文深度: {} 条",
                    state.step_count,
                    elapsed,
                    state.total_tokens,
                    len(session_messages),
                )
            else:
                logger.error(
                    "任务执行中断/失败 | 状态: {} | 步数: {} | 错误: {}",
                    state.status.value,
                    state.step_count,
                    state.last_error,
                )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
