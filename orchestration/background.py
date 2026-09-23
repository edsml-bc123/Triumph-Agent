"""
triumph-agent 异步后台长任务调度器 (Background Tasks Runtime)
============================================================
核心哲学与设计思想（对标 learn-claude-code s11_background_tasks）：
1. 守护进程托管与非阻塞执行 (Non-blocking Process Supervision)：
   - 长命令（如测试跑批、构建、后台服务等）转入后台独立进程组执行；
   - 立即返回 task_id，防止主循环在单个工具调用上发生阻塞挂起。
2. 进程树安全与防孤儿泄漏 (Process Group Isolation)：
   - 通过 start_new_session=True 为后台子进程建立独立 Process Group；
   - 终止时通过 os.killpg 连根关停全组派生进程，结合 atexit 注册全局兜底清理。
3. 专属独立日志落盘 (Isolated Log Streams)：
   - 进程 stdout/stderr 实时流式重定向至 runs/{run_id}/tasks/{task_id}.log；
   - 内存仅保留最近 500 字符摘要，杜绝巨型输出打爆 Agent 上下文与内存。
4. 主循环主动唤醒与通知注入 (Reactive Wakeup AOP)：
   - 实现 HookPlugin 协议，接入 StepStart 切面；
   - 任务退出后自动在下一轮将 <task_notification> 注入消息流，驱动模型自主收敛。
5. 统一协议插件装配 (Tool & Hook Plugin Protocol)：
   - 遵循 ToolPlugin 与 HookPlugin 规范，无缝挂载至 ToolRegistry 与 HookManager。
"""

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from runtime.hooks import HookManager
from runtime.state import AgentState, current_run_id_var
from tools import ToolRegistry

# 统一东八区时区
CST = timezone(timedelta(hours=8))


class TaskStatus(str, Enum):
    """后台任务运行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class BackgroundTask:
    """
    后台任务元数据实体
    """
    task_id: str
    command: str
    status: TaskStatus = TaskStatus.RUNNING
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    log_file: Path = field(default_factory=lambda: Path("task.log"))
    started_at: datetime = field(default_factory=lambda: datetime.now(CST))
    completed_at: Optional[datetime] = None
    summary: Optional[str] = None
    notified: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "command": self.command,
            "status": self.status.value,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "log_file": str(self.log_file),
            "started_at": self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": self.completed_at.strftime("%Y-%m-%d %H:%M:%S") if self.completed_at else None,
            "summary": self.summary,
            "notified": self.notified,
            "error": self.error,
        }


class BackgroundTaskManager:
    """
    后台任务管理中心 (Thread-Safe Supervisor)
    负责子进程生命周期、进程树终结、日志分流与就绪事件暂存。
    """

    def __init__(
        self,
        workdir: Optional[Path] = None,
        runs_dir: Optional[Path] = None,
    ):
        self.workdir = (workdir or Path.cwd()).resolve()
        self.runs_dir = (runs_dir or (self.workdir / "runs")).resolve()
        self._tasks: Dict[str, BackgroundTask] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._ready_queue: List[str] = []
        self._counter: int = 0
        self._lock = threading.RLock()

        # 注册解释器退出时的孤儿进程清理兜底
        atexit.register(self.shutdown_all)

    def start(self, command: str, run_id: Optional[str] = None) -> BackgroundTask:
        """
        以非阻塞方式在独立进程组中启动长耗时命令
        """
        clean_cmd = command.strip()
        if not clean_cmd:
            raise ValueError("后台执行命令不能为空")

        # 危险命令防御拦截
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in clean_cmd for d in dangerous):
            raise PermissionError(f"命令包含被策略禁止的危险行为: {clean_cmd}")

        with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"

            # 确定日志路径：优先使用传入 run_id，其次尝试协程环境变量，缺省 default
            effective_run_id = run_id or current_run_id_var.get() or "default"
            log_dir = self.runs_dir / effective_run_id / "tasks"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{task_id}.log"

            # 环境变量注入：优先使用当前 Python 解释器所在的 bin 目录
            child_env = os.environ.copy()
            current_bin_dir = str(Path(sys.executable).parent)
            child_env["PATH"] = f"{current_bin_dir}:{child_env.get('PATH', '')}"

            # 打开日志文件流
            log_fp = open(log_file, "w", encoding="utf-8", errors="replace")

            try:
                # 开启全新进程组 (start_new_session=True)
                process = subprocess.Popen(
                    clean_cmd,
                    shell=True,
                    cwd=self.workdir,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                    env=child_env,
                )
            except Exception as e:
                log_fp.close()
                raise RuntimeError(f"启动后台子进程失败: {e}") from e

            task = BackgroundTask(
                task_id=task_id,
                command=clean_cmd,
                status=TaskStatus.RUNNING,
                pid=process.pid,
                log_file=log_file,
                started_at=datetime.now(CST),
            )
            self._tasks[task_id] = task
            self._processes[task_id] = process

            # 启动监控守护线程
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(task_id, process, log_fp, log_file),
                daemon=True,
                name=f"bg-monitor-{task_id}",
            )
            monitor_thread.start()

            logger.info(f"[BackgroundTask] 成功启动任务 {task_id} (PID: {process.pid}): {clean_cmd[:80]}")
            return task

    def _monitor_process(
        self,
        task_id: str,
        process: subprocess.Popen,
        log_fp: Any,
        log_file: Path,
    ) -> None:
        """
        后台监控守护线程：等待子进程退出并收集结算信息
        """
        try:
            return_code = process.wait()
        except Exception as e:
            logger.error(f"[BackgroundTask] 监控任务 {task_id} 异常: {e}")
            return_code = -1
        finally:
            try:
                log_fp.flush()
                log_fp.close()
            except Exception:
                pass

        # 读取最后 500 字符作为快速摘要
        summary = ""
        try:
            if log_file.exists():
                content = log_file.read_text(encoding="utf-8", errors="replace").strip()
                summary = content[-500:] if len(content) > 500 else content
        except Exception as read_err:
            summary = f"(读取日志摘要失败: {read_err})"

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            # 如果已经被外部显式杀死，则保留 KILLED 状态
            if task.status != TaskStatus.KILLED:
                if return_code == 0:
                    task.status = TaskStatus.COMPLETED
                else:
                    task.status = TaskStatus.FAILED

            task.exit_code = return_code
            task.completed_at = datetime.now(CST)
            task.summary = summary if summary else "(no output)"

            # 将完成任务推入待通知队列
            if task_id not in self._ready_queue:
                self._ready_queue.append(task_id)

            logger.info(f"[BackgroundTask] 任务 {task_id} 执行结束: status={task.status.value}, exit_code={return_code}")

    def get(self, task_id: str) -> Optional[BackgroundTask]:
        """查询任务信息"""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[BackgroundTask]:
        """列出全部受管后台任务"""
        with self._lock:
            return list(self._tasks.values())

    def read_logs(self, task_id: str, tail_lines: int = 50) -> str:
        """
        安全读取任务日志尾部切片
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return f"Error: 未找到后台任务 '{task_id}'。"
            log_path = task.log_file

        if not log_path.exists():
            return "(任务尚未产生日志输出)"

        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not lines:
                return "(日志为空)"
            if len(lines) <= tail_lines:
                return "\n".join(lines)
            return (
                f"... (前序 {len(lines) - tail_lines} 行已省略) ...\n"
                + "\n".join(lines[-tail_lines:])
            )
        except Exception as e:
            return f"Error: 读取日志失败: {e}"

    def kill(self, task_id: str) -> bool:
        """
        安全杀死后台任务的整个进程组 (SIGTERM -> SIGKILL)
        """
        with self._lock:
            task = self._tasks.get(task_id)
            process = self._processes.get(task_id)

            if not task:
                return False

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED):
                return True

            task.status = TaskStatus.KILLED

        if process and process.pid:
            # 针对进程组全量关停
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

            # 给予 0.2 秒优雅关停窗口
            time.sleep(0.2)

            try:
                # 仍存活则强制 SIGKILL
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        with self._lock:
            task.completed_at = datetime.now(CST)
            task.exit_code = -9
            if task_id not in self._ready_queue:
                self._ready_queue.append(task_id)

        logger.warning(f"[BackgroundTask] 已强制终止任务 {task_id} (PID: {task.pid}) 及其子进程树")
        return True

    def collect_notifications(self) -> List[str]:
        """
        提取新完成且尚未推送通知的任务，并格式化为标准 XML 通知块
        """
        ready_tasks: List[BackgroundTask] = []
        with self._lock:
            for tid in self._ready_queue:
                t = self._tasks.get(tid)
                if t and not t.notified:
                    ready_tasks.append(t)
                    t.notified = True
            self._ready_queue.clear()

        notifications: List[str] = []
        for t in ready_tasks:
            xml_block = (
                f"<task_notification>\n"
                f"  <task_id>{t.task_id}</task_id>\n"
                f"  <status>{t.status.value}</status>\n"
                f"  <exit_code>{t.exit_code}</exit_code>\n"
                f"  <command>{t.command}</command>\n"
                f"  <log_file>{t.log_file}</log_file>\n"
                f"  <summary>{t.summary or '(no output)'}</summary>\n"
                f"</task_notification>"
            )
            notifications.append(xml_block)

        return notifications

    def shutdown_all(self) -> None:
        """
        清理所有存活后台子进程（用于环境退出或重置）
        """
        with self._lock:
            active_ids = [
                tid for tid, t in self._tasks.items()
                if t.status == TaskStatus.RUNNING
            ]
        for tid in active_ids:
            try:
                self.kill(tid)
            except Exception:
                pass


class BackgroundTaskHook:
    """
    后台任务生命周期主动切面插件 (HookPlugin 规范)
    监听 StepStart 切面，主动将就绪的后台任务通知注入消息流。
    """

    def __init__(self, manager: BackgroundTaskManager):
        self.manager = manager

    def register_to(self, hooks: HookManager) -> None:
        """挂载至 HookManager"""
        hooks.register("StepStart", self.on_step_start)
        hooks.register("Stop", self.on_stop)

    async def on_step_start(self, step: int, state: AgentState) -> None:
        """
        每轮 ReAct 步循环开始前，探查是否有已就绪的后台任务结果，自动注入上下文
        """
        notifications = self.manager.collect_notifications()
        if not notifications:
            return

        inject_text = "\n\n".join(notifications)
        logger.info(f"[BackgroundTaskHook] 检测到 {len(notifications)} 项就绪后台任务，注入第 {step} 步上下文")

        # 将通知作为当前环境上下文注入到消息流
        # 如果末尾消息已经是用户角色，则追加在其末尾；否则以 user 角色注入独立通知
        if state.messages and state.messages[-1].get("role") == "user":
            last_content = state.messages[-1].get("content", "")
            if isinstance(last_content, str):
                state.messages[-1]["content"] = f"{last_content}\n\n[Background Task Updates]:\n{inject_text}"
            elif isinstance(last_content, list):
                last_content.append({"type": "text", "text": f"\n[Background Task Updates]:\n{inject_text}"})
        else:
            state.messages.append({
                "role": "user",
                "content": f"[Background Task Notification]\n{inject_text}",
            })

    async def on_stop(self, state: AgentState) -> None:
        """运行结束切面日志"""
        tasks = self.manager.list_tasks()
        running_count = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
        if running_count > 0:
            logger.info(f"[BackgroundTaskHook] 会话结束，当前仍有 {running_count} 项后台任务正在持续执行中")


class BackgroundTaskTool:
    """
    后台任务控制管理工具插件 (ToolPlugin 规范)
    向模型暴露 check_task, kill_task, list_tasks 三大管控指令。
    """

    def __init__(self, manager: BackgroundTaskManager):
        self.manager = manager

    def register_to(self, registry: ToolRegistry) -> None:
        """向 ToolRegistry 注册后台长任务管理工具集 (check_task, kill_task, list_tasks)"""
        # 1. 注册 check_task
        registry.register(
            name="check_task",
            description="查询指定后台长任务的当前状态、进程信息、退出码及最新日志尾部输出。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "后台任务标识符，如 'bg_0001'",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "读取日志文件末尾的行数，缺省为 50 行。",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_check_task,
        )

        # 2. 注册 kill_task
        registry.register(
            name="kill_task",
            description="强制终止正在后台执行的长任务进程树，释放系统资源。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "要终止的后台任务标识符，如 'bg_0001'",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_kill_task,
        )

        # 3. 注册 list_tasks
        registry.register(
            name="list_tasks",
            description="列出当前系统管理的所有后台任务及其最新运行状态列表。",
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_list_tasks,
        )

    def _handle_check_task(self, task_id: str, tail_lines: int = 50) -> str:
        task = self.manager.get(task_id)
        if not task:
            return f"Error: 未找到任务 ID '{task_id}'。"

        recent_logs = self.manager.read_logs(task_id, tail_lines=tail_lines)
        return (
            f"=== 任务状态: {task.task_id} ===\n"
            f"- 命令内容: {task.command}\n"
            f"- 当前状态: {task.status.value}\n"
            f"- 进程 PID: {task.pid}\n"
            f"- 退出码:   {task.exit_code if task.exit_code is not None else '运行中'}\n"
            f"- 启动时间: {task.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- 结束时间: {task.completed_at.strftime('%Y-%m-%d %H:%M:%S') if task.completed_at else '尚未结束'}\n"
            f"- 日志文件: {task.log_file}\n"
            f"\n--- 日志最新 {tail_lines} 行切片 ---\n"
            f"{recent_logs}"
        )

    def _handle_kill_task(self, task_id: str) -> str:
        ok = self.manager.kill(task_id)
        if ok:
            return f"成功：后台任务 '{task_id}' 已被终止。"
        return f"Error: 终止任务 '{task_id}' 失败或任务不存在。"

    def _handle_list_tasks(self) -> str:
        tasks = self.manager.list_tasks()
        if not tasks:
            return "当前没有任何后台任务。"

        lines = ["| Task ID | Status | PID | Exit | Started At | Command |",
                 "|---|---|---|---|---|---|"]
        for t in tasks:
            cmd_preview = t.command[:35] + "..." if len(t.command) > 35 else t.command
            exit_str = str(t.exit_code) if t.exit_code is not None else "-"
            lines.append(
                f"| {t.task_id} | {t.status.value} | {t.pid} | {exit_str} | "
                f"{t.started_at.strftime('%H:%M:%S')} | `{cmd_preview}` |"
            )
        return "\n".join(lines)
