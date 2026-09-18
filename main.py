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
from context import BudgetHook
from runtime import AgentLoop, AgentState, AgentStatus, HookManager, TrajectoryHook
from security import PermissionHook
from tools.registry import ToolRegistry


def print_banner():
    banner = (
        "\n" + "=" * 68 + "\n"
        "  triumph-agent v0.1 (Mini Coding Agent Runtime)\n"
        "  基于阿里云百炼原生协议 + 显式状态机 + 工业级权限切面 + 预算硬熔断构建\n"
        + "=" * 68 + "\n"
        "提示: 输入任务指令（如：“查看当前目录下的文件并统计数量”），按回车执行。\n"
        "提示: 输入 q 或 exit 退出程序。\n"
    )
    print(banner)


async def main():
    print_banner()

    # 初始化全局基础设施（保持连接池复用与工作区绑定）
    workdir = Path.cwd()
    async with DashScopeClient() as client:
        registry = ToolRegistry(workdir=workdir)

        # 显式初始化生命周期钩子总线并挂载切面插件 (统一插件装配协议)
        hooks = HookManager()
        hooks.register_plugin(TrajectoryHook(runs_dir=workdir / "runs"))
        hooks.register_plugin(PermissionHook(workdir=workdir, interactive=True))
        hooks.register_plugin(BudgetHook(max_total_tokens=10000))

        loop_engine = AgentLoop(client=client, registry=registry, hooks=hooks)

        # 外层会话循环 (Outer Loop)
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

            # 为当前任务初始化独立的 AgentState
            state = AgentState(max_steps=20)
            state.add_user_message(user_input)

            logger.info("启动自主执行任务 | Run ID: {}", state.run_id)
            start_time = asyncio.get_event_loop().time()

            try:
                # 唤醒内层 ReAct 自主执行循环
                await loop_engine.run(state)
            except Exception as e:
                state.mark_failed(f"未知异常中断: {e}")
                logger.error("运行时出现未捕获异常: {}", e)

            elapsed = asyncio.get_event_loop().time() - start_time

            # 结构化输出执行报告
            if state.status == AgentStatus.SUCCESS:
                logger.success(
                    "任务执行成功 | 步数: {} | 耗时: {:.2f}s | Token开销: {}",
                    state.step_count,
                    elapsed,
                    state.total_tokens,
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
