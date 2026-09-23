"""
tests/test_session.py - 会话生命周期与作用域隔离测试
=====================================================
验证目标：
1. SessionContext 初始化与 session_id 唯一性；
2. 目录拓扑自动收敛至 .sessions/{session_id}/tasks 与 runs；
3. 多轮 Run 在同一 Session 下共享 DAG 任务拓扑看板；
4. session.reset() 物理轮转与旧会话隔离封存；
5. current_session_id_var 上下文变量跟踪。
"""

from orchestration.dag import TaskStore
from runtime.session import SessionContext, current_session_id_var


def test_session_context_initialization(tmp_path):
    """验证会话初始化、目录生成与上下文变量注入"""
    workdir = tmp_path
    session = SessionContext(workdir=workdir)

    assert session.session_id.startswith("session_")
    assert current_session_id_var.get() == session.session_id

    # 验证专属子目录存在
    assert session.session_dir.exists()
    assert session.tasks_dir.exists()
    assert session.runs_dir.exists()

    assert session.tasks_dir.parent == session.session_dir
    assert session.runs_dir.parent == session.session_dir


def test_session_multi_run_shared_tasks(tmp_path):
    """验证同一个会话内，多个 Run 共享同一个 DAG Task 看板"""
    workdir = tmp_path
    session = SessionContext(workdir=workdir)

    # 模拟 Run 1：在当前 session 的看板上创建任务
    store_run_1 = TaskStore(workdir=workdir, tasks_dir=session.tasks_dir)
    task1 = store_run_1.create(subject="任务1：设计架构")

    # 模拟 Run 2（如触发 max_steps 后下一轮唤醒）：使用同一个 session 的看板
    store_run_2 = TaskStore(workdir=workdir, tasks_dir=session.tasks_dir)
    tasks_in_run_2 = store_run_2.list()

    # 验证 Run 2 完全可见 Run 1 创建的任务
    assert len(tasks_in_run_2) == 1
    assert tasks_in_run_2[0].id == task1.id
    assert tasks_in_run_2[0].subject == "任务1：设计架构"


def test_session_reset_isolation(tmp_path):
    """验证会话 reset 彻底开启新空间，且旧会话目录与任务保持隔离封存"""
    workdir = tmp_path
    session = SessionContext(workdir=workdir)

    # 在 Session 1 下写入任务
    store_s1 = TaskStore(workdir=workdir, tasks_dir=session.tasks_dir)
    t1 = store_s1.create(subject="旧会话的需求任务")
    old_session_dir = session.session_dir
    old_session_id = session.session_id

    # 用户执行 /clear 触发 session.reset()
    session.reset()
    new_session_id = session.session_id
    new_session_dir = session.session_dir

    assert new_session_id != old_session_id
    assert new_session_dir != old_session_dir
    assert current_session_id_var.get() == new_session_id

    # 验证旧会话目录完整封存
    assert old_session_dir.exists()
    assert (old_session_dir / "tasks" / f"{t1.id}.json").exists()

    # 验证新会话的任务看板是完全空白的
    store_s2 = TaskStore(workdir=workdir, tasks_dir=session.tasks_dir)
    assert len(store_s2.list()) == 0
