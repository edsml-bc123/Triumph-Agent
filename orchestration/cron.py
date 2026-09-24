"""
triumph-agent 定时调度与心跳保活引擎 (Cron Scheduler Runtime)
============================================================
对标 learn-claude-code/s12_cron_scheduler 生产级规范，核心机制：
1. 标准五段式 Cron 表达式解析与精确校验 (Minute, Hour, Day, Month, Weekday)；
2. CronJob 任务模型与两阶段 ACK / Rollback 事务性交付保障（At-least-once）；
3. 磁盘持久化原子落盘机制（临时文件 + os.replace），支持进程重启自动恢复；
4. 周期心跳轮询引擎与时间旅行（Time Travel）单测支持；
5. CronHook 切面无缝注入与 CronTool 纯白板工具插件封装。
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from runtime.hooks import HookManager
from runtime.state import AgentState
from tools import ToolRegistry

# 统一东八区时区
CST = timezone(timedelta(hours=8))


# ----------------------------------------------------------------------
# 1. 标准五段式 Cron 表达式解析与时间匹配
# ----------------------------------------------------------------------

def _validate_cron_field(field_val: str, minimum: int, maximum: int) -> Optional[str]:
    """验证单个 cron 字段的语法和数值区间"""
    if field_val == "*":
        return None
    if field_val.startswith("*/"):
        step_str = field_val[2:]
        if not step_str.isdigit() or int(step_str) <= 0:
            return f"步长非法: '{field_val}'"
        return None
    if "," in field_val:
        for part in field_val.split(","):
            err = _validate_cron_field(part.strip(), minimum, maximum)
            if err:
                return err
        return None
    if "-" in field_val:
        parts = field_val.split("-", 1)
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return f"区间格式非法: '{field_val}'"
        start_val, end_val = int(parts[0]), int(parts[1])
        if start_val > end_val:
            return f"区间起始值大于结束值: '{field_val}'"
        if start_val < minimum or end_val > maximum:
            return f"区间 '{field_val}' 超出允许范围 [{minimum}-{maximum}]"
        return None
    if not field_val.isdigit():
        return f"无法识别的字段格式: '{field_val}'"
    val = int(field_val)
    if val < minimum or val > maximum:
        return f"数值 {val} 超出允许范围 [{minimum}-{maximum}]"
    return None


def validate_cron(cron_expr: str) -> Optional[str]:
    """
    验证 5 字段 Cron 表达式合法性：
    字段顺序：[分 0-59] [时 0-23] [日 1-31] [月 1-12] [周 0-6 (0=周日)]
    返回 None 表示校验通过，返回错误字符串表示校验失败。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Cron 表达式必须包含恰好 5 个字段 (分 时 日 月 周)，当前解析到 {len(fields)} 个"

    field_specs = [
        ("minute", 0, 59),
        ("hour", 0, 23),
        ("day-of-month", 1, 31),
        ("month", 1, 12),
        ("day-of-week", 0, 6),
    ]

    for (name, min_v, max_v), val in zip(field_specs, fields):
        err = _validate_cron_field(val, min_v, max_v)
        if err:
            return f"{name} 字段错误: {err}"
    return None


def _cron_field_matches(field_val: str, current_value: int) -> bool:
    """检查单个字段是否匹配当前时间值"""
    if field_val == "*":
        return True
    if field_val.startswith("*/"):
        step = int(field_val[2:])
        return current_value % step == 0
    if "," in field_val:
        return any(_cron_field_matches(part.strip(), current_value) for part in field_val.split(","))
    if "-" in field_val:
        start_str, end_str = field_val.split("-", 1)
        return int(start_str) <= current_value <= int(end_str)
    return current_value == int(field_val)


def cron_matches(cron_expr: str, moment: datetime) -> bool:
    """
    判断指定的 5 字段 Cron 表达式在 moment 这一分钟是否触发。
    注意：周字段中 0 代表周日，1-6 代表周一至周六。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute_str, hour_str, day_str, month_str, weekday_str = fields
    # moment.weekday(): 0=周一, 6=周日 -> 转换为 0=周日, 1=周一...6=周六
    cron_weekday = (moment.weekday() + 1) % 7

    if not (
        _cron_field_matches(minute_str, moment.minute)
        and _cron_field_matches(hour_str, moment.hour)
        and _cron_field_matches(month_str, moment.month)
    ):
        return False

    day_matches = _cron_field_matches(day_str, moment.day)
    weekday_matches = _cron_field_matches(weekday_str, cron_weekday)

    # 标准 Unix Cron 规则：
    # 如果日和星期都是 *，全匹配
    # 如果其中一个是 *，以另一个匹配为准
    # 如果日和星期都指定了具体值，满足其一即触发（OR 逻辑）
    if day_str == "*" and weekday_str == "*":
        return True
    if day_str == "*":
        return weekday_matches
    if weekday_str == "*":
        return day_matches
    return day_matches or weekday_matches


# ----------------------------------------------------------------------
# 2. CronJob 任务实体定义
# ----------------------------------------------------------------------

@dataclass
class CronJob:
    """
    Cron 定时任务实体
    """
    id: str                          # 唯一标识符，形如 'cron_a1b2c3d4'
    cron: str                        # 5 字段表达式
    prompt: str                      # 到期派发给 Agent 的目标提示词
    recurring: bool = True           # 是否周期性重复执行（False 为一次性触发）
    durable: bool = True             # 是否磁盘持久化（False 为纯会话内存存续）
    pending_delivery: bool = False   # 是否已到期入队，但尚未被模型成功确认（ACK）
    last_fired: Optional[str] = None # 最近一次触发的分钟标记，如 '2026-09-24 16:45'
    created_at: str = field(default_factory=lambda: datetime.now(CST).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "cron": self.cron,
            "prompt": self.prompt,
            "recurring": self.recurring,
            "durable": self.durable,
            "pending_delivery": self.pending_delivery,
            "last_fired": self.last_fired,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CronJob:
        return cls(
            id=data["id"],
            cron=data["cron"],
            prompt=data["prompt"],
            recurring=data.get("recurring", True),
            durable=data.get("durable", True),
            pending_delivery=data.get("pending_delivery", False),
            last_fired=data.get("last_fired"),
            created_at=data.get("created_at", datetime.now(CST).isoformat()),
        )


# ----------------------------------------------------------------------
# 3. Cron 调度与心跳保活引擎 (CronScheduler)
# ----------------------------------------------------------------------

class CronScheduler:
    """
    Cron 定时任务调度器与心跳保活引擎
    - 负责管理所有已注册的 CronJob；
    - 负责周期性检查到期任务并推入待交付队列；
    - 严格遵循两阶段 ACK / Rollback 机制；
    - 使用原子临时文件 + os.replace 进行持久化。
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._scheduled_jobs: Dict[str, CronJob] = {}
        self._cron_queue: List[CronJob] = []
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.storage_path:
            self._load_durable_jobs()

    # --- 基础查询与管理 ---

    def list_jobs(self) -> List[CronJob]:
        """获取所有已调度的任务列表拷贝"""
        with self._lock:
            return list(self._scheduled_jobs.values())

    def get_job(self, job_id: str) -> Optional[CronJob]:
        """根据 ID 获取任务"""
        with self._lock:
            return self._scheduled_jobs.get(job_id)

    def has_pending_jobs(self) -> bool:
        """检查是否有待处理交付的任务队列"""
        with self._lock:
            return bool(self._cron_queue)

    def _allocate_id(self) -> str:
        """生成全局唯一的任务 ID"""
        for _ in range(100):
            cand = f"cron_{secrets.token_hex(4)}"
            if cand not in self._scheduled_jobs:
                return cand
        raise RuntimeError("无法为定时任务分配唯一 ID")

    def schedule(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
    ) -> CronJob:
        """
        注册并调度一个新的 Cron 任务。
        若校验失败则抛出 ValueError。
        """
        err = validate_cron(cron)
        if err:
            raise ValueError(f"Cron 表达式校验失败: {err}")
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("任务 Prompt 不能为空")

        with self._lock:
            job_id = self._allocate_id()
            job = CronJob(
                id=job_id,
                cron=cron.strip(),
                prompt=clean_prompt,
                recurring=recurring,
                durable=durable,
            )
            self._scheduled_jobs[job_id] = job

            if durable and self.storage_path:
                try:
                    self._save_durable_jobs()
                except Exception:
                    self._scheduled_jobs.pop(job_id, None)
                    raise

            logger.info(f"[CronScheduler] 成功调度任务 {job.id}: {job.cron} -> {job.prompt[:40]}")
            return job

    def cancel(self, job_id: str) -> bool:
        """取消指定 ID 的任务，若成功返回 True，任务不存在返回 False"""
        with self._lock:
            job = self._scheduled_jobs.get(job_id)
            if job is None:
                return False

            old_queue = list(self._cron_queue)
            self._scheduled_jobs.pop(job_id, None)
            self._cron_queue = [q for q in self._cron_queue if q.id != job_id]

            if job.durable and self.storage_path:
                try:
                    self._save_durable_jobs()
                except Exception:
                    self._scheduled_jobs[job_id] = job
                    self._cron_queue = old_queue
                    raise

            logger.info(f"[CronScheduler] 已取消任务 {job_id}")
            return True

    # --- 调度轮询与到期入队 ---

    def poll_due_jobs(self, moment: Optional[datetime] = None) -> List[CronJob]:
        """
        检查指定时刻（默认为当前本地时间）是否有任务到期触发。
        若触发，将其标记为 pending_delivery 并推入待消费队列。
        返回本次新入队的 Job 列表。
        """
        now = moment or datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        newly_due: List[CronJob] = []

        with self._lock:
            for job in list(self._scheduled_jobs.values()):
                # 已在交付队列中，或本分钟已经触发过，跳过
                if job.pending_delivery or job.last_fired == minute_marker:
                    continue

                if cron_matches(job.cron, now):
                    old_pending = job.pending_delivery
                    old_last_fired = job.last_fired

                    job.pending_delivery = True
                    job.last_fired = minute_marker

                    if job.durable and self.storage_path:
                        try:
                            self._save_durable_jobs()
                        except Exception as e:
                            job.pending_delivery = old_pending
                            job.last_fired = old_last_fired
                            logger.error(f"[CronScheduler] 任务 {job.id} 入队持久化失败: {e}")
                            continue

                    self._cron_queue.append(job)
                    newly_due.append(job)
                    logger.info(f"[CronScheduler] 任务到期入队: {job.id} ({job.prompt[:40]})")

        return newly_due

    # --- 两阶段 ACK / Rollback 事务 ---

    def consume_queue(self) -> List[CronJob]:
        """
        原子消费并清空当前待交付的任务队列。
        调用者取走任务后，必须在处理完成后调用 acknowledge 或 rollback。
        """
        with self._lock:
            jobs = list(self._cron_queue)
            self._cron_queue.clear()
            return jobs

    def acknowledge(self, jobs: List[CronJob]) -> None:
        """
        第二阶段：成功确认 (ACK)。
        - 周期任务 (recurring=True): 清除 pending_delivery=False，等待下一次周期；
        - 一次性任务 (recurring=False): 直接从调度表中销毁。
        """
        if not jobs:
            return

        with self._lock:
            changed: List[tuple[CronJob, bool]] = []
            removed: List[CronJob] = []

            for delivered in jobs:
                current = self._scheduled_jobs.get(delivered.id)
                if current is None:
                    continue
                changed.append((current, current.pending_delivery))
                if current.recurring:
                    current.pending_delivery = False
                else:
                    removed.append(current)
                    self._scheduled_jobs.pop(current.id, None)

            if self.storage_path and any(j.durable for j, _ in changed):
                try:
                    self._save_durable_jobs()
                except Exception:
                    # 发生持久化故障，回滚内存状态并重新排队
                    for j in removed:
                        self._scheduled_jobs[j.id] = j
                    for j, pending in changed:
                        j.pending_delivery = pending
                    queued_ids = {q.id for q in self._cron_queue}
                    for j, _ in changed:
                        if j.id not in queued_ids:
                            self._cron_queue.append(j)
                    raise

            logger.info(f"[CronScheduler] 成功 ACK 确认 {len(jobs)} 个定时任务")

    def rollback(self, jobs: List[CronJob]) -> None:
        """
        第二阶段：异常回滚 (Rollback)。
        若大模型执行异常或会话崩溃，任务重新压回待交付队列，确保任务不丢失。
        """
        if not jobs:
            return

        with self._lock:
            queued_ids = {q.id for q in self._cron_queue}
            for delivered in jobs:
                current = self._scheduled_jobs.get(delivered.id)
                if current is None:
                    continue
                current.pending_delivery = True
                if current.id not in queued_ids:
                    self._cron_queue.insert(0, current)
                    queued_ids.add(current.id)

            logger.warning(f"[CronScheduler] 已回滚 {len(jobs)} 个定时任务并重新推入等待队列")

    # --- 持久化存储与崩溃恢复 ---

    def _save_durable_jobs(self) -> None:
        """原子落盘保存所有 durable=True 的任务至磁盘文件"""
        if not self.storage_path:
            return

        with self._lock:
            payload = [
                job.to_dict()
                for job in self._scheduled_jobs.values()
                if job.durable
            ]
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.storage_path.with_name(
                f"{self.storage_path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
            )
            try:
                temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(temp_path, self.storage_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    def _load_durable_jobs(self) -> None:
        """从磁盘反序列化恢复持久化任务"""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            raw_text = self.storage_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                return
            payload = json.loads(raw_text)
            if not isinstance(payload, list):
                raise ValueError("持久化文件必须包含 JSON 列表")
        except Exception as e:
            logger.error(f"[CronScheduler] 读取持久化文件 {self.storage_path} 失败: {e}")
            return

        loaded_count = 0
        with self._lock:
            for item in payload:
                try:
                    job = CronJob.from_dict(item)
                    if validate_cron(job.cron) is not None:
                        continue
                    if not job.id.startswith("cron_"):
                        continue
                    self._scheduled_jobs[job.id] = job
                    # 若上次异常关机前正处于交付中，重新推入队列以保证 At-least-once
                    if job.pending_delivery:
                        self._cron_queue.append(job)
                    loaded_count += 1
                except Exception as ex:
                    logger.warning(f"[CronScheduler] 跳过损坏的持久化任务记录: {ex}")
                    continue

        if loaded_count > 0:
            logger.info(f"[CronScheduler] 成功恢复 {loaded_count} 个持久化 Cron 任务")

    # --- 独立后台守护线程心跳保活 ---

    def start(self, interval_seconds: float = 1.0) -> None:
        """启动后台独立守护线程进行时间轮询，彻底脱离主线程 I/O 阻塞影响"""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()

            def _worker():
                logger.debug("[CronScheduler] 后台心跳守护线程已启动")
                while not self._stop_event.wait(interval_seconds):
                    try:
                        self.poll_due_jobs()
                    except Exception as e:
                        logger.error(f"[CronScheduler] 轮询异常: {e}")
                logger.debug("[CronScheduler] 后台心跳守护线程已退出")

            self._thread = threading.Thread(
                target=_worker,
                name="cron-scheduler-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """优雅关停后台心跳守护线程"""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
            logger.debug("[CronScheduler] 后台心跳守护线程已停止")


# ----------------------------------------------------------------------
# 4. CronHook 切面拦截器
# ----------------------------------------------------------------------

class CronHook:
    """
    Cron 切面生命周期监听器
    在每轮 LLM 推理前 (StepStart)，检查是否有到期未交付的 Cron 任务；
    若有，自动将其提取并以规范的上下文提醒形式注入 messages。
    配合两阶段 ACK：在模型成功回应后提交确认，若模型报错则触发回滚。
    """

    def __init__(self, scheduler: CronScheduler):
        self.scheduler = scheduler
        self._in_flight_jobs: List[CronJob] = []

    def register_to(self, hooks: HookManager) -> None:
        hooks.register("StepStart", self.on_step_start)
        hooks.register("LLMResponse", self.on_llm_response)
        hooks.register("Stop", self.on_stop)

    async def on_step_start(self, state: AgentState) -> None:
        """在每轮推理开始时检查到期定时任务并注入"""
        # 兜底先轮询一次当前时刻
        self.scheduler.poll_due_jobs()

        due_jobs = self.scheduler.consume_queue()
        if not due_jobs:
            return

        self._in_flight_jobs.extend(due_jobs)

        notices = []
        for job in due_jobs:
            freq = "周期性" if job.recurring else "单次"
            notices.append(
                f"<scheduled_cron_task>\n"
                f"  Job ID: {job.id}\n"
                f"  Cron: {job.cron} ({freq})\n"
                f"  Prompt: {job.prompt}\n"
                f"</scheduled_cron_task>"
            )

        inject_content = (
            f"[定时调度任务触发通知 - 请优先处理以下到期任务]:\n"
            + "\n".join(notices)
        )

        logger.info(f"[CronHook] 捕获到 {len(due_jobs)} 项到期任务，向上下文注入调度提示")

        if state.messages and state.messages[-1].get("role") == "user":
            last_content = state.messages[-1].get("content", "")
            if isinstance(last_content, str):
                state.messages[-1]["content"] = f"{last_content}\n\n{inject_content}"
            elif isinstance(last_content, list):
                last_content.append({"type": "text", "text": f"\n\n{inject_content}"})
        else:
            state.messages.append({
                "role": "user",
                "content": inject_content,
            })

    async def on_llm_response(self, state: AgentState, response: Any) -> None:
        """当模型成功产生响应时，提交两阶段 ACK"""
        if self._in_flight_jobs:
            jobs_to_ack = list(self._in_flight_jobs)
            self._in_flight_jobs.clear()
            self.scheduler.acknowledge(jobs_to_ack)

    async def on_stop(self, state: AgentState) -> None:
        """若因异常或提前终止导致 in-flight 任务未被 ACK，执行事务回滚"""
        if self._in_flight_jobs:
            jobs_to_restore = list(self._in_flight_jobs)
            self._in_flight_jobs.clear()
            self.scheduler.rollback(jobs_to_restore)


# ----------------------------------------------------------------------
# 5. CronTool 工具插件 (ToolPlugin & ClosablePlugin)
# ----------------------------------------------------------------------

class CronTool:
    """
    Cron 任务管理工具插件
    向大模型暴露 schedule_cron, list_crons, cancel_cron 三大工具
    """

    def __init__(self, scheduler: Optional[CronScheduler] = None, storage_path: Optional[Path] = None):
        self.scheduler = scheduler or CronScheduler(storage_path=storage_path)

    def register_to(self, registry: ToolRegistry) -> None:
        """向注册表注册工具，并自动拉起后台心跳守护线程"""
        # 挂载即启动：自动拉起独立后台守护线程
        self.scheduler.start()

        # 1. 注册 schedule_cron
        registry.register(
            name="schedule_cron",
            description=(
                "调度一个定时任务。使用标准的 5 字段 Cron 表达式（分 时 日 月 周）。\n"
                "例如：'*/5 * * * *' 代表每 5 分钟执行一次，'0 9 * * 1-5' 代表工作日上午 9 点执行。\n"
                "到期后系统会将 prompt 自动注入对话流中供模型处理。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cron": {
                        "type": "string",
                        "description": "标准 5 字段 Cron 表达式，如 '*/10 * * * *' 或 '0 9 * * *'",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "定时触发时需要执行的任务指令/提示词",
                    },
                    "recurring": {
                        "type": "boolean",
                        "description": "是否为周期性任务。True 为反复触发，False 为触发一次后自动注销。默认为 True。",
                    },
                    "durable": {
                        "type": "boolean",
                        "description": "是否持久化保存到磁盘。True 在系统重启后依然有效，False 仅在当前会话内存中有效。默认为 True。",
                    },
                },
                "required": ["cron", "prompt"],
            },
            handler=self._handle_schedule_cron,
        )

        # 2. 注册 list_crons
        registry.register(
            name="list_crons",
            description="列出当前系统已调度的所有定时任务列表及其状态。",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_list_crons,
        )

        # 3. 注册 cancel_cron
        registry.register(
            name="cancel_cron",
            description="根据定时任务 ID 取消指定的定时任务。",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "要取消的定时任务 ID，如 'cron_1a2b3c4d'",
                    },
                },
                "required": ["job_id"],
            },
            handler=self._handle_cancel_cron,
        )

    async def close(self) -> None:
        """优雅关闭调度器后台任务"""
        self.scheduler.stop()

    def _handle_schedule_cron(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
    ) -> str:
        try:
            job = self.scheduler.schedule(
                cron=cron,
                prompt=prompt,
                recurring=recurring,
                durable=durable,
            )
            freq_str = "周期任务" if job.recurring else "单次任务"
            durable_str = "磁盘持久化" if job.durable else "会话内存"
            return (
                f"成功创建定时任务: {job.id}\n"
                f"- 表达式: {job.cron}\n"
                f"- 类型: {freq_str} | {durable_str}\n"
                f"- 提示词: {job.prompt}"
            )
        except Exception as e:
            return f"Error: 创建定时任务失败: {e}"

    def _handle_list_crons(self) -> str:
        jobs = self.scheduler.list_jobs()
        if not jobs:
            return "当前没有任何已调度的定时任务。"

        lines = [
            "| Job ID | Cron | Type | Storage | Pending | Last Fired | Prompt |",
            "|---|---|---|---|---|---|---|",
        ]
        for j in jobs:
            freq = "周期" if j.recurring else "单次"
            storage = "持久化" if j.durable else "内存"
            pending = "是" if j.pending_delivery else "否"
            last_fired = j.last_fired or "-"
            prompt_preview = j.prompt[:30] + "..." if len(j.prompt) > 30 else j.prompt
            lines.append(
                f"| `{j.id}` | `{j.cron}` | {freq} | {storage} | {pending} | {last_fired} | {prompt_preview} |"
            )
        return "\n".join(lines)

    def _handle_cancel_cron(self, job_id: str) -> str:
        ok = self.scheduler.cancel(job_id)
        if ok:
            return f"成功：定时任务 '{job_id}' 已取消。"
        return f"Error: 未找到任务 ID '{job_id}' 或取消失败。"
