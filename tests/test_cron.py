"""
tests/test_cron.py - Cron 定时调度与心跳保活引擎测试套件
=========================================================
全面验证（对标 learn-claude-code/s12_cron_scheduler）：
1. 5 字段 Cron 表达式解析与边界校验 (validate_cron)；
2. 精确时间匹配与时间旅行验证 (cron_matches)；
3. 任务调度、ID 分配与取消 (schedule, cancel)；
4. 到期触发与同分钟防重复入队 (poll_due_jobs)；
5. 两阶段 ACK 机制：周期任务重置等待 vs 单次任务自动销毁 (acknowledge)；
6. 两阶段 Rollback 机制：异常恢复重新入队保证任务不丢失 (rollback)；
7. 磁盘原子落盘与重启故障恢复 (durable persistence & recovery)；
8. CronHook 拦截切面与上下文注入 (StepStart / LLMResponse / Stop)；
9. ToolRegistry 纯白板插件集成与命令执行 (schedule_cron, list_crons, cancel_cron)。
"""

import json
from datetime import datetime
from pathlib import Path
import pytest

from orchestration.cron import (
    CronHook,
    CronScheduler,
    CronTool,
    cron_matches,
    validate_cron,
)
from runtime.hooks import HookManager
from runtime.state import AgentState
from tools import ToolRegistry


# ----------------------------------------------------------------------
# 1. 表达式语法验证与时间匹配单测
# ----------------------------------------------------------------------

def test_validate_cron_valid():
    """测试合法的 5 字段 Cron 表达式校验通过"""
    assert validate_cron("* * * * *") is None
    assert validate_cron("0 9 * * *") is None
    assert validate_cron("*/10 * * * *") is None
    assert validate_cron("0 9 * * 1-5") is None
    assert validate_cron("15,45 8-18 1,15 * 0") is None


def test_validate_cron_invalid():
    """测试非法表达式或边界越界抛出错误"""
    # 字段数量错误
    assert "恰好 5 个字段" in validate_cron("0 9 * *")
    assert "恰好 5 个字段" in validate_cron("0 9 * * * *")

    # 分钟越界
    assert "minute" in validate_cron("60 * * * *")
    # 小时越界
    assert "hour" in validate_cron("0 24 * * *")
    # 日越界
    assert "day-of-month" in validate_cron("0 0 32 * *")
    # 月越界
    assert "month" in validate_cron("0 0 1 13 *")
    # 周越界
    assert "day-of-week" in validate_cron("0 0 * * 7")

    # 步长非法
    assert "步长非法" in validate_cron("*/0 * * * *")
    # 区间倒置
    assert "区间起始值大于结束值" in validate_cron("10-5 * * * *")


def test_cron_matches_time_travel():
    """时间旅行测试：验证不同时刻与 cron 表达式的精准匹配"""
    # 2026-08-10 是周一 (weekday=0, cron_weekday=1)
    monday_0900 = datetime(2026, 8, 10, 9, 0)
    monday_0905 = datetime(2026, 8, 10, 9, 5)
    monday_0903 = datetime(2026, 8, 10, 9, 3)
    # 2026-08-16 是周日 (weekday=6, cron_weekday=0)
    sunday_0900 = datetime(2026, 8, 16, 9, 0)

    # 1. 每天上午 9 点
    assert cron_matches("0 9 * * *", monday_0900)
    assert not cron_matches("0 9 * * *", monday_0905)

    # 2. 每 5 分钟
    assert cron_matches("*/5 * * * *", monday_0900)
    assert cron_matches("*/5 * * * *", monday_0905)
    assert not cron_matches("*/5 * * * *", monday_0903)

    # 3. 工作日上午 9 点 (1-5)
    assert cron_matches("0 9 * * 1-5", monday_0900)
    assert not cron_matches("0 9 * * 1-5", sunday_0900)

    # 4. 周日上午 9 点 (0)
    assert cron_matches("0 9 * * 0", sunday_0900)
    assert not cron_matches("0 9 * * 0", monday_0900)


# ----------------------------------------------------------------------
# 2. 调度、入队与防重测试
# ----------------------------------------------------------------------

def test_cron_schedule_and_cancel():
    """测试任务创建、唯一 ID 生成与取消"""
    scheduler = CronScheduler()

    job1 = scheduler.schedule("0 9 * * *", "每日测试", recurring=True)
    assert job1.id.startswith("cron_")
    assert job1.cron == "0 9 * * *"
    assert job1.recurring is True
    assert job1.durable is True

    # 重复创建应分配不同 ID
    job2 = scheduler.schedule("0 10 * * *", "临时任务", recurring=False, durable=False)
    assert job2.id != job1.id
    assert len(scheduler.list_jobs()) == 2

    # 取消任务
    assert scheduler.cancel(job1.id) is True
    assert scheduler.cancel(job1.id) is False  # 重复取消
    assert len(scheduler.list_jobs()) == 1


def test_cron_schedule_validation_errors():
    """测试非法表达式或空 Prompt 抛出明确异常"""
    scheduler = CronScheduler()
    with pytest.raises(ValueError, match="Cron 表达式校验失败"):
        scheduler.schedule("invalid_cron", "测试")

    with pytest.raises(ValueError, match="Prompt 不能为空"):
        scheduler.schedule("0 9 * * *", "   ")


def test_cron_poll_due_jobs_and_deduplication():
    """测试到期检测与同一分钟防重复入队"""
    scheduler = CronScheduler()
    scheduler.schedule("0 9 * * *", "上午9点任务")
    scheduler.schedule("*/10 * * * *", "每10分钟任务")

    moment_0900 = datetime(2026, 8, 10, 9, 0)
    due_0900 = scheduler.poll_due_jobs(moment_0900)
    # 两项均在 9:00 到期
    assert len(due_0900) == 2
    assert scheduler.has_pending_jobs() is True

    # 同一分钟内再次检测，不应重复入队
    dup_due = scheduler.poll_due_jobs(moment_0900)
    assert len(dup_due) == 0

    # 消费队列
    consumed = scheduler.consume_queue()
    assert len(consumed) == 2
    assert scheduler.has_pending_jobs() is False


# ----------------------------------------------------------------------
# 3. 两阶段 ACK 与 Rollback 事务性测试
# ----------------------------------------------------------------------

def test_cron_two_phase_ack_recurring_vs_oneshot():
    """测试 ACK 机制：周期任务重置 pending_delivery，单次任务自动注销"""
    scheduler = CronScheduler()
    rec_job = scheduler.schedule("0 9 * * *", "周期任务", recurring=True)
    one_job = scheduler.schedule("0 9 * * *", "单次任务", recurring=False)

    moment = datetime(2026, 8, 10, 9, 0)
    due = scheduler.poll_due_jobs(moment)
    assert len(due) == 2

    consumed = scheduler.consume_queue()
    # 模拟模型处理成功，提交 ACK
    scheduler.acknowledge(consumed)

    # 验证周期任务依然存在且 pending_delivery 恢复为 False
    current_rec = scheduler.get_job(rec_job.id)
    assert current_rec is not None
    assert current_rec.pending_delivery is False

    # 验证一次性任务已被自动销毁
    assert scheduler.get_job(one_job.id) is None


def test_cron_two_phase_rollback():
    """测试 Rollback 机制：模型处理失败时任务回退重新排队"""
    scheduler = CronScheduler()
    job = scheduler.schedule("0 9 * * *", "需保护的任务")

    moment = datetime(2026, 8, 10, 9, 0)
    scheduler.poll_due_jobs(moment)

    consumed = scheduler.consume_queue()
    assert len(consumed) == 1
    assert scheduler.has_pending_jobs() is False

    # 模拟模型故障或崩溃，执行 Rollback
    scheduler.rollback(consumed)

    # 验证任务重新回到队列中
    assert scheduler.has_pending_jobs() is True
    re_consumed = scheduler.consume_queue()
    assert len(re_consumed) == 1
    assert re_consumed[0].id == job.id


# ----------------------------------------------------------------------
# 4. 原子持久化与重启崩溃恢复测试
# ----------------------------------------------------------------------

def test_cron_atomic_persistence_and_recovery(tmp_path: Path):
    """测试磁盘原子落盘与进程重启恢复"""
    storage_file = tmp_path / ".cron_tasks.json"
    scheduler = CronScheduler(storage_path=storage_file)

    durable_job = scheduler.schedule("0 9 * * *", "持久化任务", durable=True)
    session_job = scheduler.schedule("0 10 * * *", "临时任务", durable=False)

    assert storage_file.exists()
    payload = json.loads(storage_file.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["id"] == durable_job.id

    # 模拟进程重启
    new_scheduler = CronScheduler(storage_path=storage_file)
    restored = new_scheduler.list_jobs()
    assert len(restored) == 1
    assert restored[0].id == durable_job.id
    assert new_scheduler.get_job(session_job.id) is None


def test_cron_recovery_at_least_once_delivery(tmp_path: Path):
    """测试关机前处于 pending_delivery 的任务重启后重新入队 (At-least-once)"""
    storage_file = tmp_path / ".cron_tasks.json"
    scheduler = CronScheduler(storage_path=storage_file)
    job = scheduler.schedule("0 9 * * *", "关键报表生成", durable=True)

    moment = datetime(2026, 8, 10, 9, 0)
    scheduler.poll_due_jobs(moment)
    # 此时 job 已进入 pending_delivery 并落盘，但未 ACK

    # 模拟突发重启
    recovered_scheduler = CronScheduler(storage_path=storage_file)
    assert recovered_scheduler.has_pending_jobs() is True
    recovered_queue = recovered_scheduler.consume_queue()
    assert len(recovered_queue) == 1
    assert recovered_queue[0].id == job.id


# ----------------------------------------------------------------------
# 5. CronHook 与 Agent 切面注入测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cron_hook_injection_and_ack():
    """测试 CronHook 在 StepStart 注入定时任务，并在 LLM 成功后自动 ACK"""
    scheduler = CronScheduler()
    hook = CronHook(scheduler=scheduler)
    hooks_mgr = HookManager()
    hook.register_to(hooks_mgr)

    job = scheduler.schedule("0 9 * * *", "汇总每日指标")
    moment = datetime(2026, 8, 10, 9, 0)
    scheduler.poll_due_jobs(moment)

    state = AgentState(max_steps=5)
    state.add_user_message("你好，帮我写代码")

    # 1. 触发 StepStart
    await hooks_mgr.trigger("StepStart", state)
    assert len(state.messages) == 1
    last_msg = state.messages[0]["content"]
    assert "[定时调度任务触发通知 - 请优先处理以下到期任务]" in last_msg
    assert job.id in last_msg
    assert "汇总每日指标" in last_msg

    # 2. 模拟 LLM 成功响应，触发 LLMResponse
    await hooks_mgr.trigger("LLMResponse", state, response={"text": "收到定时任务"})
    # 校验已被自动 ACK
    assert scheduler.get_job(job.id).pending_delivery is False


@pytest.mark.asyncio
async def test_cron_hook_rollback_on_stop():
    """测试在处理过程中若未完成并调用 Stop，CronHook 会自动执行 Rollback"""
    scheduler = CronScheduler()
    hook = CronHook(scheduler=scheduler)
    hooks_mgr = HookManager()
    hook.register_to(hooks_mgr)

    job = scheduler.schedule("0 9 * * *", "重要任务")
    scheduler.poll_due_jobs(datetime(2026, 8, 10, 9, 0))

    state = AgentState(max_steps=5)
    state.add_user_message("开始执行")

    await hooks_mgr.trigger("StepStart", state)
    assert scheduler.has_pending_jobs() is False  # 已被取出处于 in-flight

    # 模拟异常中止，触发 Stop
    await hooks_mgr.trigger("Stop", state)
    # 验证任务已被 Rollback 重新推回待交付队列
    assert scheduler.has_pending_jobs() is True


# ----------------------------------------------------------------------
# 6. ToolRegistry 与 CronTool 工具插件装配测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cron_tool_plugin_and_execution(tmp_path: Path):
    """测试 CronTool 作为插件向 ToolRegistry 注册并执行调用"""
    registry = ToolRegistry(workdir=tmp_path)
    cron_tool = CronTool(storage_path=tmp_path / "crons.json")
    registry.register_plugin(cron_tool)

    specs = {t["function"]["name"]: t for t in registry.get_tools_spec()}
    assert "schedule_cron" in specs
    assert "list_crons" in specs
    assert "cancel_cron" in specs

    # 1. 执行 schedule_cron
    res_sched = await registry.execute(
        "schedule_cron",
        {
            "cron": "*/15 * * * *",
            "prompt": "每15分钟健康检查",
            "recurring": True,
            "durable": True,
        },
    )
    assert "成功创建定时任务: cron_" in res_sched
    assert "每15分钟健康检查" in res_sched

    # 2. 执行 list_crons
    res_list = await registry.execute("list_crons", {})
    assert "*/15 * * * *" in res_list
    assert "每15分钟健康检查" in res_list

    # 提取生成的 job_id
    jobs = cron_tool.scheduler.list_jobs()
    assert len(jobs) == 1
    job_id = jobs[0].id

    # 3. 执行 cancel_cron
    res_cancel = await registry.execute("cancel_cron", {"job_id": job_id})
    assert f"成功：定时任务 '{job_id}' 已取消" in res_cancel

    # 再次查询列表应为空
    empty_list = await registry.execute("list_crons", {})
    assert "当前没有任何已调度的定时任务" in empty_list

    # 4. 关闭注册表，级联释放 CronTool（自动停止后台心跳线程）
    await registry.close()
    assert cron_tool.scheduler._thread is None or not cron_tool.scheduler._thread.is_alive()


def test_cron_daemon_thread_lifecycle():
    """测试独立后台守护线程的启动、心跳轮询与优雅关停"""
    import time
    scheduler = CronScheduler()

    # 启动后台守护线程，轮询间隔设为 0.05 秒
    scheduler.start(interval_seconds=0.05)
    assert scheduler._thread is not None
    assert scheduler._thread.is_alive() is True
    assert scheduler._thread.name == "cron-scheduler-heartbeat"
    assert scheduler._thread.daemon is True

    # 调度一个每分钟都会匹配的任务
    scheduler.schedule("* * * * *", "常驻守护任务")

    # 等待后台守护线程自主完成轮询
    for _ in range(20):
        if scheduler.has_pending_jobs():
            break
        time.sleep(0.02)

    assert scheduler.has_pending_jobs() is True

    # 优雅停机
    scheduler.stop()
    assert scheduler._thread is None

