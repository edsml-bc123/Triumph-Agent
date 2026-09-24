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
    CronHook,
    CronScheduler,
    CronTool,
    DAGTaskTool,
    JobHook,
    JobManager,
    JobTool,
    SubAgentTool,
    TaskStore,
)
from runtime import (
    AgentLoop,
    AgentState,
    AgentStatus,
    HookManager,
    SessionContext,
    TrajectoryHook,
    async_read_terminal_input,
)
from security import PermissionHook
from mcp import MCPTool
from skills import SkillLoader, SkillTool
from tools import BuiltinTool, ToolRegistry


def print_banner(session: SessionContext):
    banner = (
        "\n" + "=" * 68 + "\n"
        "  triumph-agent v0.2 (Mini Coding Agent Runtime)\n"
        "  基于阿里云百炼原生协议 + 显式状态机 + 权限审计 + 预算熔断 + 渐进式压缩 + 三层记忆 + 技能加载\n"
        + "=" * 68 + "\n"
        f"  [当前会话 ID]: {session.session_id}\n"
        f"  [会话任务看板]: {session.tasks_dir.relative_to(session.workdir)}\n"
        f"  [会话轨迹目录]: {session.runs_dir.relative_to(session.workdir)}\n"
        + "-" * 68 + "\n"
        "提示: 输入任务指令（如：“查看当前目录下的文件并统计数量”），按回车执行。\n"
        "提示: 支持自动派生子智能体 (subagent) 隔离处理复杂子探索。\n"
        "提示: 支持 DAG 拓扑多任务拆解编排 (create_task / claim_task / list_tasks)。\n"
        "提示: 支持后台异步作业托管 (run_in_background + list_jobs)。\n"
        "提示: 支持专业技能渐进式按需加载 (load_skill)。\n"
        "提示: 支持跨进程 MCP 标准协议工具扩展 (connect_mcp，内置服务: system_info)。\n"
        "提示: 输入 /clear 重置会话上下文并开启全新 Session 空间。\n"
        "提示: 输入 /memory 查看当前持久化知识库目录。\n"
        "提示: 输入 q 或 exit 退出程序。\n"
    )
    print(banner)


async def main():
    # 初始化工作区与会话作用域 (Session Context)
    workdir = Path.cwd()
    session = SessionContext(workdir=workdir)
    print_banner(session)

    async with DashScopeClient() as client:
        registry = ToolRegistry(workdir=workdir)
        job_manager = JobManager(workdir=workdir, runs_dir=session.runs_dir)
        task_store = TaskStore(workdir=workdir, tasks_dir=session.tasks_dir)
        skill_loader = SkillLoader(skills_dir=workdir / "skills")

        # 工业级生产梯度参数
        compactor_cfg = CompactionConfig(
            max_single_tool_output_chars=20_000,
            preview_chars=1200,
            max_history_messages=30,
            keep_recent_tool_results=2,
            soft_threshold_chars=35_000,
            hard_threshold_chars=80_000,
        )
        compactor = ContextCompactor(workdir=workdir, config=compactor_cfg, base_runs_dir=session.runs_dir)

        cron_scheduler = CronScheduler(storage_path=session.session_dir / "crons.json")

        registry.register_plugin(BuiltinTool(workdir=workdir, job_manager=job_manager))
        registry.register_plugin(JobTool(manager=job_manager))
        registry.register_plugin(DAGTaskTool(store=task_store))
        registry.register_plugin(SubAgentTool(client=client))
        registry.register_plugin(SkillTool(loader=skill_loader))
        registry.register_plugin(MCPTool.from_config(workdir / "mcp.json", workdir=workdir))
        registry.register_plugin(CompactTool(compactor=compactor, client=client))
        registry.register_plugin(CronTool(scheduler=cron_scheduler))

        memory_mgr = MemoryManager(workdir=workdir)

        hooks = HookManager()
        hooks.register_plugin(TrajectoryHook(runs_dir=session.runs_dir))
        hooks.register_plugin(PermissionHook(workdir=workdir, interactive=True))
        hooks.register_plugin(BudgetHook(max_total_tokens=250000))
        hooks.register_plugin(CompactorHook(compactor=compactor, client=client))
        hooks.register_plugin(MemoryHook(manager=memory_mgr, client=client))
        hooks.register_plugin(JobHook(manager=job_manager))
        hooks.register_plugin(CronHook(scheduler=cron_scheduler))

        loop_engine = AgentLoop(
            client=client,
            registry=registry,
            hooks=hooks,
            skill_loader=skill_loader,
        )

        # 外层会话循环 (Outer Loop) - 维护连续的多轮会话历史 (Session Context)
        session_messages: list[dict] = []

        try:
            while True:
                # 异步可中断读取终端输入：每 50ms 让出一次 CPU，支持检测后台定时任务到期
                user_input = await async_read_terminal_input(
                    prompt="\ntriumph >> ",
                    interrupt_check=cron_scheduler.has_pending_jobs,
                )

                is_cron = False
                if user_input is None:
                    if cron_scheduler.has_pending_jobs():
                        is_cron = True
                        print("\n\n[定时任务自动唤醒] 检测到到期定时任务，立即自主执行...")
                    else:
                        logger.info("接收到中断退出信号，triumph-agent 正在安全退出。")
                        break

                if not is_cron:
                    if not user_input.strip():
                        continue

                    if user_input.lower() in ("q", "quit", "exit"):
                        logger.info("用户主动退出，再见！")
                        break

                    if user_input.lower() in ("/clear", "clear"):
                        session_messages.clear()
                        session.reset()
                        # 重建 session 作用域下的 task_store，开启崭新看板
                        task_store.tasks_dir = session.tasks_dir
                        task_store.tasks_dir.mkdir(parents=True, exist_ok=True)
                        print(f"\n[已重置并开启全新 Session 空间]: {session.session_id}")
                        print(f"  [新任务看板]: {session.tasks_dir.relative_to(session.workdir)}")
                        print(f"  [新轨迹目录]: {session.runs_dir.relative_to(session.workdir)}")
                        continue

                    if user_input.lower() in ("/memory", "memory"):
                        index_text = memory_mgr.storage.read_index()
                        if index_text:
                            print(f"\n[当前长期记忆目录]:\n{index_text}")
                        else:
                            print("\n[当前长期记忆目录为空]")
                        continue

                # 统一执行链路：为当前任务初始化独立的 AgentState，继承多轮会话历史
                state = AgentState(max_steps=20)
                if session_messages:
                    state.messages.extend(session_messages)

                # 仅在必要时根据触发源定制目标与入参
                if is_cron:
                    state.current_goal = "执行已到期的定时调度任务"
                    task_desc = "定时任务"
                else:
                    state.add_user_message(user_input)
                    state.current_goal = user_input
                    task_desc = "用户指令任务"

                logger.info("启动{} | Run ID: {} | 携带历史上下文: {} 条消息", task_desc, state.run_id, len(session_messages))
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
                        "{}执行成功 | 步数: {} | 耗时: {:.2f}s | Token开销: {} | 当前会话上下文深度: {} 条",
                        task_desc,
                        state.step_count,
                        elapsed,
                        state.total_tokens,
                        len(session_messages),
                    )
                else:
                    logger.error(
                        "{}执行中断/失败 | 状态: {} | 步数: {} | 错误: {}",
                        task_desc,
                        state.status.value,
                        state.step_count,
                        state.last_error,
                    )
        finally:
            await registry.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
