"""
tests/test_job.py - 异步后台作业托管测试套件
===========================================
全面验证（零历史包袱纯粹架构）：
1. 作业启动、守护监控与 COMPLETED 正常终态；
2. 异常退出捕获与 FAILED 终态；
3. kill 作业进程组安全关停与 KILLED 终态；
4. 日志实时落盘与 read_logs tail 读取；
5. 主动通知 (Reactive Wakeup) 抽取与单次消费保证；
6. JobHook 在 StepStart 切面的上下文注入机制；
7. ToolRegistry 挂载与 bash run_in_background 端到端联动 (check_job, kill_job, list_jobs)；
8. 验证 list_jobs 与 DAG list_tasks 完全解耦互不覆盖。
"""

import sys
import time
import pytest

from orchestration.dag import DAGTaskTool, TaskStore
from orchestration.job import (
    JobHook,
    JobManager,
    JobStatus,
    JobTool,
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
    manager = JobManager(workdir=workdir, runs_dir=runs_dir)
    yield workdir, runs_dir, manager
    manager.shutdown_all()


def test_job_success(temp_env):
    """测试常规命令后台作业执行、退出监听与日志落盘"""
    workdir, runs_dir, manager = temp_env

    # 启动一个简短后台作业（约 0.2s）
    py_code = "import time; time.sleep(0.2); print('OUTPUT_SUCCESS_LINE')"
    job = manager.start(f"{sys.executable} -c \"{py_code}\"")

    assert job.job_id.startswith("job_")
    assert job.status == JobStatus.RUNNING
    assert job.pid is not None
    assert job.log_file.exists()

    # 等待其执行完毕
    for _ in range(30):
        if job.status != JobStatus.RUNNING:
            break
        time.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    assert job.exit_code == 0
    assert "OUTPUT_SUCCESS_LINE" in (job.summary or "")

    # 验证日志读取
    logs = manager.read_logs(job.job_id, tail_lines=10)
    assert "OUTPUT_SUCCESS_LINE" in logs


def test_job_failure(temp_env):
    """测试作业命令异常退出时状态标记为 FAILED"""
    workdir, runs_dir, manager = temp_env

    py_code = "import sys; print('FAIL_TRACE'); sys.exit(7)"
    job = manager.start(f"{sys.executable} -c \"{py_code}\"")

    # 等待完成
    for _ in range(30):
        if job.status != JobStatus.RUNNING:
            break
        time.sleep(0.05)

    assert job.status == JobStatus.FAILED
    assert job.exit_code == 7
    assert "FAIL_TRACE" in (job.summary or "")


def test_job_kill(temp_env):
    """测试对长作业下发 kill 终止整个作业进程组"""
    workdir, runs_dir, manager = temp_env

    # 启动一个长耗时睡眠作业
    py_code = "import time; time.sleep(20)"
    job = manager.start(f"{sys.executable} -c \"{py_code}\"")
    assert job.status == JobStatus.RUNNING

    # 执行终止
    ok = manager.kill(job.job_id)
    assert ok is True
    assert job.status == JobStatus.KILLED

    # 再次查询确认
    fetched = manager.get(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.KILLED


def test_job_drain_notifications_once(temp_env):
    """测试就绪通知单次消费保证，杜绝重复注入"""
    workdir, runs_dir, manager = temp_env

    py_code = "print('NOTIFICATION_TEST')"
    job = manager.start(f"{sys.executable} -c \"{py_code}\"")

    # 等待完成
    for _ in range(30):
        if job.status != JobStatus.RUNNING:
            break
        time.sleep(0.05)

    # 第一次提取，应有 1 条
    notifications = manager.drain_unnotified_completions()
    assert len(notifications) == 1
    assert notifications[0].job_id == job.job_id
    assert notifications[0].status == JobStatus.COMPLETED

    # 第二次提取，队列已清空，不再产生通知
    second_round = manager.drain_unnotified_completions()
    assert len(second_round) == 0


@pytest.mark.asyncio
async def test_job_hook_step_start(temp_env):
    """测试 JobHook 在 StepStart 时主动唤醒注入"""
    workdir, runs_dir, manager = temp_env
    hook = JobHook(manager=manager)
    hooks_mgr = HookManager()
    hook.register_to(hooks_mgr)

    # 模拟完成一个后台作业
    job = manager.start(f"{sys.executable} -c \"print('HOOK_DONE')\"")
    for _ in range(30):
        if job.status != JobStatus.RUNNING:
            break
        time.sleep(0.05)

    # 构造状态机并触发 StepStart
    state = AgentState()
    state.add_user_message("继续分析任务")

    await hooks_mgr.trigger("StepStart", state=state)

    # 验证末尾用户消息成功合成了后台作业更新
    last_msg = state.messages[-1]
    assert "[Background Job Updates]" in last_msg["content"]
    assert job.job_id in last_msg["content"]


@pytest.mark.asyncio
async def test_tool_registry_job_and_dag_no_collision(temp_env):
    """
    核心解耦断言：
    验证 ToolRegistry 同时装配 JobTool 和 DAGTaskTool 时：
    - list_jobs (后台作业) 与 list_tasks (DAG 需求任务) 共存；
    - 两者各自拥有独立的参数、handler 与返回结构，彻底杜绝覆盖！
    """
    workdir, runs_dir, manager = temp_env
    registry = ToolRegistry(workdir=workdir)
    registry.register_plugin(BuiltinToolsPlugin(workdir=workdir, job_manager=manager))

    # 装配后台作业工具插件
    job_plugin = JobTool(manager=manager)
    job_plugin.register_to(registry)

    # 装配 DAG 任务工具插件
    task_store = TaskStore(workdir=workdir)
    dag_plugin = DAGTaskTool(store=task_store)
    dag_plugin.register_to(registry)

    # 验证两个 list 工具均在注册表中并立存活
    assert "list_jobs" in registry._handlers
    assert "list_tasks" in registry._handlers

    # 1. 验证 bash 工具的 run_in_background 分支
    start_res = await registry.execute(
        "bash",
        {
            "command": f"{sys.executable} -c \"import time; time.sleep(0.2); print('BASH_BG_OK')\"",
            "run_in_background": True,
        },
    )
    assert "[Background job job_" in start_res
    job_id = start_res.split("[Background job ")[1].split(" ")[0]

    # 2. 验证 list_jobs
    list_res = await registry.execute("list_jobs", {})
    assert job_id in list_res

    # 3. 验证 check_job
    check_res = await registry.execute("check_job", {"job_id": job_id, "tail_lines": 5})
    assert f"=== 后台作业状态: {job_id} ===" in check_res

    # 等待其完成
    for _ in range(30):
        j = manager.get(job_id)
        if j and j.status != JobStatus.RUNNING:
            break
        time.sleep(0.05)

    # 再次 check 日志
    check_after = await registry.execute("check_job", {"job_id": job_id, "tail_lines": 10})
    assert "BASH_BG_OK" in check_after

    # 4. 验证 kill_job
    kill_res = await registry.execute("kill_job", {"job_id": job_id})
    assert "成功" in kill_res or "已" in kill_res

    # 5. 验证 DAG 的 list_tasks 是纯正的业务需求看板，不受作业数据干扰
    dag_list_res = await registry.execute("list_tasks", {})
    assert "No tasks" in dag_list_res
