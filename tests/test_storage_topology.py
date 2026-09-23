"""
tests/test_storage_topology.py - 端到端存储拓扑与一等公民生命周期集成测试
========================================================================
全面验证架构物理落盘标准：
1. 会话级一等公民：.sessions/{session_id}/ 成为工作区唯一合法临时产物存储区；
2. 任务看板会话共享：.sessions/{session_id}/tasks/{task_id}.json 跨多轮 Run 状态持久共享；
3. 运行审计严格隔离：.sessions/{session_id}/runs/{run_id}/trajectory.jsonl 按轮独立记录；
4. 后台作业日志隔离：.sessions/{session_id}/runs/{run_id}/jobs/{job_id}.log 随 Run 归属隔离；
5. 上下文压缩剪裁落盘：.sessions/{session_id}/runs/{run_id}/tool_outputs/ 专属落盘；
6. 轮转隔离性断言：session.reset() 物理开启新空间，旧 Session 目录完整封存归档，根目录零残留。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pytest

from context.compactor import CompactionConfig, ContextCompactor
from orchestration.dag import DAGTaskStatus, TaskStore
from orchestration.job import JobManager, JobStatus
from runtime.event import TrajectoryHook
from runtime.hooks import HookManager
from runtime.session import SessionContext, current_session_id_var
from runtime.state import AgentState, current_run_id_var


@pytest.mark.asyncio
async def test_end_to_end_session_file_topology(tmp_path: Path):
    """
    端到端测试：校验 Session -> Tasks / Runs -> Jobs / Compactor 物理存储结构
    """
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # ------------------------------------------------------------------
    # 1. 初始化 Session (会话层)
    # ------------------------------------------------------------------
    session = SessionContext(workdir=workdir)
    session_id = session.session_id

    assert session.session_dir.exists(), "会话目录应已创建"
    assert session.tasks_dir.exists(), "会话 tasks 目录应已创建"
    assert session.runs_dir.exists(), "会话 runs 目录应已创建"
    assert current_session_id_var.get() == session_id

    # 校验相对路径规范
    assert session.session_dir == workdir / ".sessions" / session_id
    assert session.tasks_dir == session.session_dir / "tasks"
    assert session.runs_dir == session.session_dir / "runs"

    # ------------------------------------------------------------------
    # 2. 模拟跨 Run 共享的 DAG 任务规划 (Task 看板层)
    # ------------------------------------------------------------------
    task_store = TaskStore(workdir=workdir, tasks_dir=session.tasks_dir)
    task1 = task_store.create(subject="设计微内核架构", description="第一阶段核心任务")
    task2 = task_store.create(subject="实现后台作业托管", description="第二阶段依赖任务")
    task_store.update_dependencies(task2.id, add_blocked_by=[task1.id])

    task1_path = session.tasks_dir / f"{task1.id}.json"
    task2_path = session.tasks_dir / f"{task2.id}.json"

    assert task1_path.exists(), f"任务 1 物理文件未落盘: {task1_path}"
    assert task2_path.exists(), f"任务 2 物理文件未落盘: {task2_path}"
    task2 = task_store.load(task2.id)
    assert task1.id in task2.blockedBy

    # ------------------------------------------------------------------
    # 3. 模拟第一轮 ReAct 运行 (Run 1: 触发后台作业与轨迹审计)
    # ------------------------------------------------------------------
    state_run_1 = AgentState()
    run_id_1 = state_run_1.run_id
    current_run_id_var.set(run_id_1)

    # 3.1 挂载审计切面并产生轨迹流水 (trajectory.jsonl)
    hooks = HookManager()
    traj_hook = TrajectoryHook(runs_dir=session.runs_dir)
    traj_hook.register_to(hooks)

    await hooks.trigger("UserPromptSubmit", prompt="请启动后台测试作业并归档超长日志", state=state_run_1)

    # 3.2 在该 Run 下启动后台长作业 (Job)
    job_mgr = JobManager(workdir=workdir, runs_dir=session.runs_dir)
    py_code = "import time; time.sleep(0.1); print('JOB_OUTPUT_TOPOLOGY_OK')"
    job = job_mgr.start(f"{sys.executable} -c \"{py_code}\"")

    expected_job_dir = session.runs_dir / run_id_1 / "jobs"
    expected_job_log = expected_job_dir / f"{job.job_id}.log"

    assert expected_job_dir.exists(), "Job 专属日志目录未创建"
    assert expected_job_log.exists(), f"Job 物理日志未落盘: {expected_job_log}"

    # 等待作业完成
    for _ in range(30):
        if job.status != JobStatus.RUNNING:
            break
        time.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    assert "JOB_OUTPUT_TOPOLOGY_OK" in expected_job_log.read_text(encoding="utf-8")

    # 3.3 触发上下文压缩引擎落盘归档 (tool_outputs)
    compactor = ContextCompactor(
        workdir=workdir,
        config=CompactionConfig(max_single_tool_output_chars=100),
        base_runs_dir=session.runs_dir,
    )
    oversized_output = "Line 001: Important Info\n" + "A" * 500 + "\nLine End: Final Info"
    output_file = compactor.save_tool_output(
        tool_call_id="call_mock_123",
        output=oversized_output,
        run_id=run_id_1,
    )

    expected_tool_outputs_dir = session.runs_dir / run_id_1 / "tool_outputs"
    assert expected_tool_outputs_dir.exists(), "压缩归档目录应正确生成"
    assert output_file.exists(), "超大工具输出物理落盘文件应存在"
    assert "Line 001" in output_file.read_text(encoding="utf-8")

    # 3.4 结束 Run 1，验证 trajectory.jsonl
    await hooks.trigger("Stop", state=state_run_1)
    expected_event_log = session.runs_dir / run_id_1 / "trajectory.jsonl"
    assert expected_event_log.exists(), f"审计流水未落盘: {expected_event_log}"
    assert "TaskStart" in expected_event_log.read_text(encoding="utf-8")
    assert "TaskEnd" in expected_event_log.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # 4. 模拟第二轮 ReAct 运行 (Run 2: 验证同一 Session 内的任务状态延续与独立 Run 目录)
    # ------------------------------------------------------------------
    state_run_2 = AgentState()
    run_id_2 = state_run_2.run_id
    current_run_id_var.set(run_id_2)

    # 验证 Run 2 继承并更新 Run 1 创建的 Task 看板
    tasks_in_run_2 = task_store.list()
    # 认领并完成任务 1，自动解锁任务 2
    ok_claim, _ = task_store.claim(task1.id, owner="agent")
    assert ok_claim is True
    ok_complete, _, unblocked = task_store.complete(task1.id, owner="agent")
    assert ok_complete is True
    assert task2.subject in unblocked

    updated_task2 = task_store.load(task2.id)
    assert updated_task2 is not None
    assert updated_task2.status == DAGTaskStatus.PENDING.value
    assert task_store.can_start(task2.id) is True

    # Run 2 启动专属审计与作业
    await hooks.trigger("UserPromptSubmit", prompt="继续推进任务2", state=state_run_2)
    job2 = job_mgr.start(f"{sys.executable} -c \"print('JOB_2_OK')\"")
    expected_job2_log = session.runs_dir / run_id_2 / "jobs" / f"{job2.job_id}.log"
    assert expected_job2_log.exists(), "Run 2 的作业日志应独立落盘在 run_id_2 下"

    await hooks.trigger("Stop", state=state_run_2)
    expected_event_log_2 = session.runs_dir / run_id_2 / "trajectory.jsonl"
    assert expected_event_log_2.exists(), "Run 2 的审计流水应落盘在 run_id_2 下"

    # 验证两个 Run 目录彼此隔离
    assert (session.runs_dir / run_id_1).is_dir()
    assert (session.runs_dir / run_id_2).is_dir()
    assert run_id_1 != run_id_2

    # ------------------------------------------------------------------
    # 5. 模拟用户执行 /clear 轮转会话 (session.reset())
    # ------------------------------------------------------------------
    old_session_id = session.session_id
    old_session_dir = session.session_dir

    session.reset()
    new_session_id = session.session_id
    new_session_dir = session.session_dir

    # 验证全新会话空间生成
    assert new_session_id != old_session_id
    assert new_session_dir != old_session_dir
    assert new_session_dir.exists()
    assert (new_session_dir / "tasks").exists()
    assert (new_session_dir / "runs").exists()

    # 验证旧会话全部资产完整封存，无任何损坏
    assert (old_session_dir / "tasks" / f"{task1.id}.json").exists()
    assert (old_session_dir / "tasks" / f"{task2.id}.json").exists()
    assert (old_session_dir / "runs" / run_id_1 / "jobs" / f"{job.job_id}.log").exists()
    assert (old_session_dir / "runs" / run_id_1 / "trajectory.jsonl").exists()
    assert (old_session_dir / "runs" / run_id_1 / "tool_outputs").exists()
    assert (old_session_dir / "runs" / run_id_2 / "jobs" / f"{job2.job_id}.log").exists()
    assert (old_session_dir / "runs" / run_id_2 / "trajectory.jsonl").exists()

    # ------------------------------------------------------------------
    # 6. 工作区根目录零污染断言
    # ------------------------------------------------------------------
    # 工作区下除了 .sessions 目录，绝不允许产生旧版的散落目录
    forbidden_roots = ["runs", ".tasks", "tasks", "jobs", "processes"]
    for forbidden in forbidden_roots:
        assert not (workdir / forbidden).exists(), f"工作区根目录下散落了禁止的临时目录: {forbidden}"

    # 统计 .sessions 下的子目录数量应恰好为 2 (旧 session 和新 session)
    sessions_found = list((workdir / ".sessions").iterdir())
    assert len(sessions_found) == 2
    session_names = {s.name for s in sessions_found}
    assert old_session_id in session_names
    assert new_session_id in session_names
