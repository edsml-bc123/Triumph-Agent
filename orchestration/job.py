"""
triumph-agent 异步后台作业托管运行时 (Background Job Runtime)
===========================================================
核心设计哲学（零历史包袱纯粹架构）：
1. 后台作业托管与非阻塞执行 (Non-blocking Job Supervision)：
   - 长命令（如本地开发服务器、测试跑批、构建任务等）转入后台独立作业组执行；
   - 立即返回 job_id（如 job_0001），主循环绝不阻塞挂起。
2. 进程树安全与防孤儿泄漏 (Process Group Isolation)：
   - 通过 start_new_session=True 为后台作业建立独立 Process Group；
   - 终止时通过 os.killpg 连根关停全组派生进程，结合 atexit 注册全局兜底清理。
3. 专属独立日志落盘 (Isolated Log Streams)：
   - 作业 stdout/stderr 实时流式重定向至指定 runs/{run_id}/jobs/{job_id}.log；
   - 内存仅保留最近 500 字符摘要，杜绝巨型日志撑爆 Agent 思考上下文。
4. 主循环主动唤醒与通知注入 (Reactive Wakeup AOP)：
   - 实现 HookPlugin 协议，接入 StepStart 切面；
   - 作业退出后自动在下一轮将 <job_notification> 注入消息流，驱动模型自主收敛。
5. 统一协议插件装配 (Tool & Hook Plugin Protocol)：
   - 向模型暴露 check_job, kill_job, list_jobs 三大作业管控指令；
   - 彻底解除与 DAG 需求任务 (Task) 的术语和工具名同名覆盖冲突。
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
from loguru import logger

from runtime.hooks import HookManager
from runtime.state import AgentState, current_run_id_var
from tools import ToolRegistry

# 统一东八区时区
CST = timezone(timedelta(hours=8))


class JobStatus(str, Enum):
    """后台作业运行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class BackgroundJob:
    """
    后台作业元数据实体
    """
    job_id: str
    command: str
    status: JobStatus = JobStatus.RUNNING
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    log_file: Path = field(default_factory=lambda: Path("job.log"))
    started_at: datetime = field(default_factory=lambda: datetime.now(CST))
    completed_at: Optional[datetime] = None
    summary: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": self.command,
            "status": self.status.value,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "log_file": str(self.log_file),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": self.summary,
            "error": self.error,
        }


class JobManager:
    """
    后台作业管理中心 (Thread-Safe Job Supervisor)
    负责非阻塞派生独立进程组、监控作业退出、清理孤儿进程以及归档日志。
    """

    def __init__(self, workdir: Path, runs_dir: Optional[Path] = None):
        self.workdir = workdir.resolve()
        self.runs_dir = (runs_dir or (self.workdir / "runs")).resolve()
        self.jobs: Dict[str, BackgroundJob] = {}
        self._counter: int = 0
        self._lock = threading.Lock()
        self._subprocesses: Dict[str, subprocess.Popen] = {}
        self._notified_job_ids: set[str] = set()

        # 注册退出钩子，防止 Agent 崩溃时遗留孤儿进程
        atexit.register(self.shutdown_all)

    def _next_job_id(self) -> str:
        """生成具有自增序号特征的作业标识符，如 job_0001"""
        with self._lock:
            self._counter += 1
            return f"job_{self._counter:04d}"

    def start(self, command: str) -> BackgroundJob:
        """
        以非阻塞方式在独立进程组中启动一条长作业命令
        """
        job_id = self._next_job_id()
        run_id = current_run_id_var.get() or "global"

        # 作业日志统一落盘至 runs/{run_id}/jobs/{job_id}.log
        job_log_dir = self.runs_dir / run_id / "jobs"
        job_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = job_log_dir / f"{job_id}.log"

        job_meta = BackgroundJob(
            job_id=job_id,
            command=command,
            log_file=log_file,
            status=JobStatus.RUNNING,
        )

        log_fp = open(log_file, "w", encoding="utf-8")

        try:
            # 建立独立 Process Group (start_new_session=True)
            subproc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(self.workdir),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            job_meta.pid = subproc.pid

            with self._lock:
                self.jobs[job_id] = job_meta
                self._subprocesses[job_id] = subproc

            logger.info(
                f"[JobManager] 成功托管后台作业 | ID: {job_id} | PID: {subproc.pid} | 日志: {log_file.name}"
            )

            # 启动守护线程监听退出
            t = threading.Thread(
                target=self._supervise_job,
                args=(job_id, subproc, log_fp),
                daemon=True,
            )
            t.start()
            return job_meta

        except Exception as e:
            log_fp.close()
            job_meta.status = JobStatus.FAILED
            job_meta.error = str(e)
            with self._lock:
                self.jobs[job_id] = job_meta
            logger.error(f"[JobManager] 启动后台作业失败: {e}")
            raise

    def _supervise_job(self, job_id: str, subproc: subprocess.Popen, log_fp: TextIO) -> None:
        """常驻守护线程，跟踪作业生命周期直至自然退出或被中断"""
        try:
            exit_code = subproc.wait()
            log_fp.flush()
            log_fp.close()

            with self._lock:
                job = self.jobs.get(job_id)
                if not job:
                    return

                job.exit_code = exit_code
                job.completed_at = datetime.now(CST)

                if job.status == JobStatus.KILLED:
                    pass
                elif exit_code == 0:
                    job.status = JobStatus.COMPLETED
                else:
                    job.status = JobStatus.FAILED

                # 提取末尾 500 字符作为内存摘要
                job.summary = self._extract_tail_summary(job.log_file, max_chars=500)

            logger.debug(
                f"[JobManager] 后台作业已退出 | ID: {job_id} | Code: {exit_code} | Status: {job.status.value}"
            )
        except Exception as e:
            logger.warning(f"[JobManager] 监控作业 {job_id} 出现异常: {e}")

    def _extract_tail_summary(self, log_path: Path, max_chars: int = 500) -> str:
        """安全读取日志末尾固定大小字符"""
        if not log_path.exists():
            return ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                return content[-max_chars:] if len(content) > max_chars else content
        except Exception:
            return ""

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        """获取指定后台作业当前状态"""
        with self._lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> List[BackgroundJob]:
        """返回所有受管作业列表（按启动时间正序）"""
        with self._lock:
            return list(self.jobs.values())

    def kill(self, job_id: str) -> bool:
        """
        优雅 + 强制杀死整个作业进程组 (Process Group)
        """
        with self._lock:
            job = self.jobs.get(job_id)
            subproc = self._subprocesses.get(job_id)

        if not job or not subproc:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.KILLED):
            return True

        if subproc.poll() is not None:
            job.status = JobStatus.COMPLETED if subproc.returncode == 0 else JobStatus.FAILED
            job.exit_code = subproc.returncode
            return True

        pgid = job.pid
        if not pgid:
            return False

        try:
            # 1. 尝试优雅 SIGTERM
            os.killpg(pgid, signal.SIGTERM)
            for _ in range(15):
                if subproc.poll() is not None:
                    break
                time.sleep(0.1)

            # 2. 若超时未退，强制 SIGKILL
            if subproc.poll() is None:
                logger.warning(f"[JobManager] 作业进程组 {pgid} 未能在优雅期内退出，发送 SIGKILL")
                os.killpg(pgid, signal.SIGKILL)
                subproc.wait(timeout=2.0)

            with self._lock:
                job.status = JobStatus.KILLED
                job.exit_code = -signal.SIGKILL
                job.completed_at = datetime.now(CST)

            logger.info(f"[JobManager] 已连根关停作业进程组 | ID: {job_id} | PGID: {pgid}")
            return True
        except ProcessLookupError:
            with self._lock:
                job.status = JobStatus.KILLED
            return True
        except Exception as e:
            logger.error(f"[JobManager] 终止作业进程组 {pgid} 异常: {e}")
            return False

    def shutdown_all(self) -> None:
        """关闭所有仍在运行中的作业"""
        with self._lock:
            running_ids = [
                jid for jid, j in self.jobs.items()
                if j.status == JobStatus.RUNNING
            ]
        for jid in running_ids:
            self.kill(jid)

    def read_logs(self, job_id: str, tail_lines: int = 50) -> str:
        """读取指定作业的最新日志尾部"""
        job = self.get(job_id)
        if not job:
            return f"Error: 未找到作业 ID '{job_id}'。"
        if not job.log_file.exists():
            return f"[提示] 日志文件尚未生成或已被清理 ({job.log_file})"

        try:
            with open(job.log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
                return "".join(tail)
        except Exception as e:
            return f"Error: 读取日志失败: {e}"

    def drain_unnotified_completions(self) -> List[BackgroundJob]:
        """
        原子消费已完成/失败但尚未向模型通知的作业列表
        """
        with self._lock:
            ready = []
            for jid, j in self.jobs.items():
                if jid not in self._notified_job_ids and j.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.KILLED,
                ):
                    ready.append(j)
                    self._notified_job_ids.add(jid)
            return ready


class JobHook:
    """
    后台作业生命周期切面钩子 (HookPlugin 规范)
    """

    def __init__(self, manager: JobManager):
        self.manager = manager

    def register_to(self, hooks: HookManager) -> None:
        hooks.register("StepStart", self.on_step_start)
        hooks.register("Stop", self.on_stop)

    async def on_step_start(self, state: AgentState) -> None:
        """在每一轮 LLM 推理前，检查是否有新结束的后台作业，若有则注入通知"""
        finished = self.manager.drain_unnotified_completions()
        if not finished:
            return

        notifications = []
        for j in finished:
            summary_info = f"\nOutput Tail:\n{j.summary}" if j.summary else ""
            notifications.append(
                f"<job_notification>\n"
                f"  Job ID: {j.job_id}\n"
                f"  Command: {j.command}\n"
                f"  Status: {j.status.value}\n"
                f"  Exit Code: {j.exit_code}{summary_info}\n"
                f"</job_notification>"
            )

        inject_text = "\n\n".join(notifications)
        logger.info(f"[JobHook] 检测到 {len(finished)} 项后台作业退出，向上下文注入状态更新")

        if state.messages and state.messages[-1].get("role") == "user":
            last_content = state.messages[-1].get("content", "")
            if isinstance(last_content, str):
                state.messages[-1]["content"] = f"{last_content}\n\n[Background Job Updates]:\n{inject_text}"
            elif isinstance(last_content, list):
                last_content.append({"type": "text", "text": f"\n[Background Job Updates]:\n{inject_text}"})
        else:
            state.messages.append({
                "role": "user",
                "content": f"[Background Job Notification]\n{inject_text}",
            })

    async def on_stop(self, state: AgentState) -> None:
        """运行结束切面日志"""
        jobs = self.manager.list_jobs()
        running_count = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
        if running_count > 0:
            logger.info(f"[JobHook] 会话结束，当前仍有 {running_count} 项后台作业正在持续执行中")


class JobTool:
    """
    后台作业管控工具插件 (ToolPlugin 规范)
    向模型暴露 check_job, kill_job, list_jobs 作业管控指令，
    彻底解除原先与 DAG 需求任务 (list_tasks 等) 的重名与语义混淆。
    """

    def __init__(self, manager: JobManager):
        self.manager = manager

    def register_to(self, registry: ToolRegistry) -> None:
        """向 ToolRegistry 注册后台作业管理工具集"""
        # 1. 注册 check_job
        registry.register(
            name="check_job",
            description="查询指定后台长作业的当前状态、PID、退出码及最新日志尾部输出。",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "后台作业标识符，如 'job_0001'",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "读取日志文件末尾的行数，缺省为 50 行。",
                    },
                },
                "required": ["job_id"],
            },
            handler=self._handle_check_job,
        )

        # 2. 注册 kill_job
        registry.register(
            name="kill_job",
            description="强制终止正在后台执行的长作业及其进程树，释放系统资源。",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "要终止的后台作业标识符，如 'job_0001'",
                    },
                },
                "required": ["job_id"],
            },
            handler=self._handle_kill_job,
        )

        # 3. 注册 list_jobs (彻底释放 list_tasks 给 DAG 需求任务独占)
        registry.register(
            name="list_jobs",
            description="列出当前系统托管的所有后台作业及其最新运行状态列表。",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_list_jobs,
        )

    def _handle_check_job(self, job_id: str, tail_lines: int = 50) -> str:
        job = self.manager.get(job_id)
        if not job:
            return f"Error: 未找到后台作业 ID '{job_id}'。"

        recent_logs = self.manager.read_logs(job_id, tail_lines=tail_lines)
        return (
            f"=== 后台作业状态: {job.job_id} ===\n"
            f"- 命令内容: {job.command}\n"
            f"- 当前状态: {job.status.value}\n"
            f"- 进程 PID: {job.pid}\n"
            f"- 退出代码: {job.exit_code if job.exit_code is not None else '运行中'}\n"
            f"- 日志文件: {job.log_file}\n"
            f"- 启动时间: {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- 完成时间: {job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else '-'}\n"
            f"\n--- 日志最新 {tail_lines} 行输出 ---\n"
            f"{recent_logs}"
        )

    def _handle_kill_job(self, job_id: str) -> str:
        ok = self.manager.kill(job_id)
        if ok:
            return f"成功：后台作业 '{job_id}' 已被终止。"
        return f"Error: 终止后台作业 '{job_id}' 失败或作业不存在。"

    def _handle_list_jobs(self) -> str:
        jobs = self.manager.list_jobs()
        if not jobs:
            return "当前没有任何受管的后台作业。"

        lines = [
            "| Job ID | Status | PID | Exit | Started At | Command |",
            "|---|---|---|---|---|---|",
        ]
        for j in jobs:
            cmd_preview = j.command[:35] + "..." if len(j.command) > 35 else j.command
            exit_str = str(j.exit_code) if j.exit_code is not None else "-"
            lines.append(
                f"| `{j.job_id}` | {j.status.value} | {j.pid or '-'} | {exit_str} | "
                f"{j.started_at.strftime('%H:%M:%S')} | `{cmd_preview}` |"
            )
        return "\n".join(lines)
