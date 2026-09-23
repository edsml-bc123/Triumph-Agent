"""
tests/test_background.py - 异步后台长任务调度器测试套件
======================================================
全面验证：
1. 任务启动、守护监控与 COMPLETED 正常终态；
2. 异常退出捕获与 FAILED 终态；
3. kill 进程组安全关停与 KILLED 终态；
4. 日志实时落盘与 read_logs tail 读取；
5. 主动通知 (Reactive Wakeup) 抽取与单次消费保证；
6. BackgroundTaskHook 在 StepStart 切面的上下文注入机制；
7. ToolRegistry 挂载与 bash run_in_background 端到端联动；
8. 危险命令防御拦截。
"""

import sys
import time
import pytest

from orchestration.background import (
    BackgroundTaskHook,
    BackgroundTaskManager,
    BackgroundTaskTool,
    TaskStatus,
)
from runtime.hooks import HookManager
from runtime.state import AgentState
from tools import BuiltinToolsPlugin, ToolRegistry


@pytest.fixture
def temp_env(tmp_path):
    """构建专用的临时沙箱与运行目录"""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    manager = BackgroundTaskManager(workdir=workdir, runs_dir=runs_dir)
    yield workdir, runs_dir, manager
    manager.shutdown_all()


def test_background_task_success(temp_env):
    """测试常规命令后台执行、退出监听与日志落盘"""
    workdir, runs_dir, manager = temp_env

    # 启动一个简短后台任务（约 0.2s）
    py_code = "import time; time.sleep(0.2); print('OUTPUT_SUCCESS_LINE')"
    task = manager.start(f"{sys.executable} -c \"{py_code}\"")

    assert task.task_id.startswith("bg_")
    assert task.status == TaskStatus.RUNNING
    assert task.pid is not None
    assert task.log_file.exists()

    # 等待其执行完毕
    for _ in range(30):
        if task.status != TaskStatus.RUNNING:
            break
        time.sleep(0.05)

    assert task.status == TaskStatus.COMPLETED
    assert task.exit_code == 0
    assert "OUTPUT_SUCCESS_LINE" in (task.summary or "")

    # 验证日志读取
    logs = manager.read_logs(task.task_id, tail_lines=10)
    assert "OUTPUT_SUCCESS_LINE" in logs


def test_background_task_failure(temp_env):
    """测试命令异常退出时状态标记为 FAILED"""
    workdir, runs_dir, manager = temp_env

    py_code = "import sys; print('FAIL_TRACE'); sys.exit(7)"
    task = manager.start(f"{sys.executable} -c \"{py_code}\"")

    # 等待完成
    for _ in range(30):
        if task.status != TaskStatus.RUNNING:
            break
        time.sleep(0.05)

    assert task.status == TaskStatus.FAILED
    assert task.exit_code == 7
    assert "FAIL_TRACE" in (task.summary or "")


def test_background_task_kill(temp_env):
    """测试对长任务下发 kill 终止进程组"""
    workdir, runs_dir, manager = temp_env

    # 启动一个长耗时睡眠命令
    py_code = "import time; time.sleep(20)"
    task = manager.start(f"{sys.executable} -c \"{py_code}\"")
    assert task.status == TaskStatus.RUNNING

    # 执行终止
    ok = manager.kill(task.task_id)
    assert ok is True
    assert task.status == TaskStatus.KILLED

    # 再次查询确认
    fetched = manager.get(task.task_id)
    assert fetched is not None
    assert fetched.status == TaskStatus.KILLED


def test_collect_notifications_once(temp_env):
    """测试就绪通知单次消费保证，杜绝重复注入"""
    workdir, runs_dir, manager = temp_env

    py_code = "print('NOTIFICATION_TEST')"
    task = manager.start(f"{sys.executable} -c \"{py_code}\"")

    # 等待完成
    for _ in range(30):
        if task.status != TaskStatus.RUNNING:
            break
        time.sleep(0.05)

    # 第一次提取，应有 1 条
    notifications = manager.collect_notifications()
    assert len(notifications) == 1
    assert f"<task_id>{task.task_id}</task_id>" in notifications[0]
    assert "<status>completed</status>" in notifications[0]

    # 第二次提取，队列已清空，不再产生通知
    second_round = manager.collect_notifications()
    assert len(second_round) == 0


@pytest.mark.asyncio
async def test_background_hook_step_start(temp_env):
    """测试 BackgroundTaskHook 在 StepStart 时主动唤醒注入"""
    workdir, runs_dir, manager = temp_env
    hook = BackgroundTaskHook(manager=manager)
    hooks_mgr = HookManager()
    hook.register_to(hooks_mgr)

    # 模拟完成一个后台任务
    task = manager.start(f"{sys.executable} -c \"print('HOOK_DONE')\"")
    for _ in range(30):
        if task.status != TaskStatus.RUNNING:
            break
        time.sleep(0.05)

    # 构造状态机并触发 StepStart
    state = AgentState()
    state.add_user_message("继续分析任务")

    await hooks_mgr.trigger("StepStart", step=1, state=state)

    # 验证末尾用户消息成功合成了后台长任务更新
    last_msg = state.messages[-1]
    assert "[Background Task Updates]" in last_msg["content"]
    assert f"<task_id>{task.task_id}</task_id>" in last_msg["content"]


@pytest.mark.asyncio
async def test_tool_registry_and_tools_integration(temp_env):
    """测试通过 ToolRegistry 派发 run_in_background 及 check/kill/list 工具"""
    workdir, runs_dir, manager = temp_env
    registry = ToolRegistry(workdir=workdir)
    registry.register_plugin(BuiltinToolsPlugin(workdir=workdir, bg_manager=manager))

    # 装配后台长任务工具插件
    tool_plugin = BackgroundTaskTool(manager=manager)
    tool_plugin.register_to(registry)

    # 1. 验证 bash 工具的 run_in_background 分支
    start_res = await registry.execute(
        "bash",
        {
            "command": f"{sys.executable} -c \"import time; time.sleep(0.2); print('BASH_BG_OK')\"",
            "run_in_background": True,
        },
    )
    assert "[Background task bg_" in start_res
    task_id = start_res.split("[Background task ")[1].split(" ")[0]

    # 2. 验证 list_tasks
    list_res = await registry.execute("list_tasks", {})
    assert task_id in list_res

    # 3. 验证 check_task
    check_res = await registry.execute("check_task", {"task_id": task_id, "tail_lines": 5})
    assert f"=== 任务状态: {task_id} ===" in check_res

    # 等待其完成
    for _ in range(30):
        t = manager.get(task_id)
        if t and t.status != TaskStatus.RUNNING:
            break
        time.sleep(0.05)

    # 再次 check 日志
    check_after = await registry.execute("check_task", {"task_id": task_id, "tail_lines": 10})
    assert "BASH_BG_OK" in check_after

    # 4. 验证 kill_task
    kill_res = await registry.execute("kill_task", {"task_id": task_id})
    assert "成功" in kill_res or "已结束" in kill_res


def test_dangerous_command_blocked(temp_env):
    """测试危险命令即使在后台也坚决拒绝执行"""
    _, _, manager = temp_env
    with pytest.raises(PermissionError):
        manager.start("rm -rf /")
