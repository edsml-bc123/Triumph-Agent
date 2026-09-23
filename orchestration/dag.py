"""
triumph-agent 有向无环图任务编排系统 (DAG Task System)
======================================================
核心设计思想（对标 learn-claude-code s10_task_system）：
1. 跨会话持久化任务状态图 (Persistent Task Graph)：
   - 任务以独立 JSON 文件持久化于工作区 `.tasks/task_[0-9a-f]{8}.json`；
   - 具备完整的工作区安全沙箱与符号链接防逃逸校验；
2. 两阶段 DAG 构建契约 (Two-Phase DAG Construction)：
   - 第一阶段：动态创建节点 (create_task) 分配全局唯一 ID；
   - 第二阶段：根据分配的运行时 ID 声明 blockedBy 依赖有向边 (update_task)；
3. 严格拓扑合法性检验与环路检测 (Cycle Detection)：
   - 深度优先搜索 (DFS) 传递依赖回溯，硬拦截任何自依赖与死锁环路；
   - 依赖修改权限守卫（仅允许 pending 且无人认领的任务更新拓扑边）；
4. 依赖门禁与主动拓扑解锁计算 (Gating & Unblocking)：
   - 前置依赖全部 completed 前禁止认领 (claim_task)；
   - 节点完成时 (complete_task) 实时计算下游最新被解锁就绪的任务列表并高亮通知；
5. 遵循 ToolPlugin 规范，提供 6 大标准百炼/OpenAI Function Schema 工具。
"""

import json
import re
import secrets
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools import ToolRegistry

TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")


class DAGTaskStatus(str, Enum):
    """DAG 任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class DAGTask:
    """DAG 拓扑任务节点实体"""
    id: str
    subject: str
    description: str = ""
    status: str = DAGTaskStatus.PENDING.value
    owner: Optional[str] = None
    blockedBy: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskStore:
    """
    持久化任务存储中心 (Thread-Safe & Workspace-Sandboxed)
    负责任务文件的安全落盘、拓扑图校验与环路检测。
    """

    def __init__(self, workdir: Path, tasks_dir: Optional[Path] = None):
        self.workdir = workdir.resolve()
        target_dir = tasks_dir or (self.workdir / ".tasks")
        self.directory = target_dir.resolve()
        self._lock = threading.RLock()

    @property
    def tasks_dir(self) -> Path:
        return self.directory

    @tasks_dir.setter
    def tasks_dir(self, val: Path) -> None:
        self.directory = val.resolve()

    def _root(self, create: bool = False) -> Path:
        """获取并校验任务存储根目录，防止逃逸工作区"""
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        root = self.directory.resolve()
        try:
            root.relative_to(self.workdir)
        except ValueError as e:
            raise ValueError("Task store escapes the workspace") from e
        return root

    def _path(self, task_id: str, create_root: bool = False) -> Path:
        """根据 task_id 计算并严格校验文件路径"""
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        root = self._root(create=create_root)
        path = (root / f"{task_id}.json").resolve()
        try:
            path.relative_to(root)
        except ValueError as e:
            raise ValueError(f"Invalid task ID: {task_id!r}") from e
        return path

    def exists(self, task_id: str) -> bool:
        """判断任务文件是否存在"""
        try:
            return self._path(task_id).is_file()
        except ValueError:
            return False

    def create(self, subject: str, description: str = "") -> DAGTask:
        """
        阶段一：创建任务节点并返回运行时分配的唯一 ID
        """
        clean_subject = subject.strip()
        if not clean_subject:
            raise ValueError("Task subject cannot be empty")

        with self._lock:
            self._root(create=True)
            for _ in range(100):
                task_id = f"task_{secrets.token_hex(4)}"
                task = DAGTask(
                    id=task_id,
                    subject=clean_subject,
                    description=description.strip(),
                    status=DAGTaskStatus.PENDING.value,
                    owner=None,
                    blockedBy=[],
                )
                try:
                    file_path = self._path(task.id, create_root=True)
                    with file_path.open("x", encoding="utf-8") as handle:
                        json.dump(task.to_dict(), handle, indent=2, ensure_ascii=False)
                    return task
                except FileExistsError:
                    continue
            raise RuntimeError("Could not allocate a unique task ID after 100 attempts")

    def _depends_on(self, task_id: str, target_id: str) -> bool:
        """
        拓扑可达性检测：检查 task_id 是否通过传递依赖链路依赖 target_id。
        用于在添加新边 target_id -> task_id 前检测是否会构成回路。
        """
        pending = [task_id]
        visited = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            try:
                current_task = self.load(current)
                pending.extend(current_task.blockedBy)
            except Exception:
                continue
        return False

    def update_dependencies(self, task_id: str, add_blocked_by: List[str]) -> DAGTask:
        """
        阶段二：使用分配的运行时 ID 为节点添加前置依赖有向边
        """
        if not isinstance(add_blocked_by, list):
            raise ValueError("addBlockedBy must be a list of task IDs")

        with self._lock:
            task = self.load(task_id)
            if task.status != DAGTaskStatus.PENDING.value or task.owner is not None:
                raise ValueError(
                    f"Task {task_id} dependencies can only be updated while pending and unowned"
                )

            # 依赖项保持顺序且去重
            dependencies = list(dict.fromkeys(add_blocked_by))
            for dependency in dependencies:
                if dependency == task_id:
                    raise ValueError("Task cannot depend on itself")
                if not self.exists(dependency):
                    raise ValueError(f"Dependency not found: {dependency}")
                if dependency not in task.blockedBy and self._depends_on(dependency, task_id):
                    raise ValueError(f"Dependency cycle detected: {task_id} -> {dependency}")

            for dep in dependencies:
                if dep not in task.blockedBy:
                    task.blockedBy.append(dep)

            self.save(task)
            return task

    def save(self, task: DAGTask) -> None:
        """原子持久化任务状态至磁盘"""
        with self._lock:
            file_path = self._path(task.id, create_root=True)
            file_path.write_text(
                json.dumps(task.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def load(self, task_id: str) -> DAGTask:
        """从磁盘加载任务实体并强类型校验"""
        with self._lock:
            file_path = self._path(task_id)
            if not file_path.is_file():
                raise FileNotFoundError(f"Task not found: {task_id}")
            data = json.loads(file_path.read_text(encoding="utf-8"))
            task = DAGTask(**data)
            if task.id != task_id:
                raise ValueError(f"Task file ID does not match {task_id}")
            if task.status not in (
                DAGTaskStatus.PENDING.value,
                DAGTaskStatus.IN_PROGRESS.value,
                DAGTaskStatus.COMPLETED.value,
            ):
                raise ValueError(f"Invalid task status: {task.status}")
            return task

    def list(self) -> List[DAGTask]:
        """按 ID 排序扫描并读取所有任务"""
        with self._lock:
            if not self.directory.exists():
                return []
            root = self._root()
            tasks: List[DAGTask] = []
            for path in sorted(root.glob("task_*.json")):
                try:
                    tasks.append(self.load(path.stem))
                except Exception:
                    continue
            return tasks

    def incomplete_dependencies(self, task: DAGTask) -> List[str]:
        """返回当前任务前置依赖中所有尚未完成的 task_id 列表"""
        incomplete: List[str] = []
        for dependency in task.blockedBy:
            try:
                dep_task = self.load(dependency)
                if dep_task.status != DAGTaskStatus.COMPLETED.value:
                    incomplete.append(dependency)
            except Exception:
                incomplete.append(dependency)
        return incomplete

    def can_start(self, task_id: str) -> bool:
        """检查任务是否可以开始（前置依赖全部已处于 completed 状态）"""
        task = self.load(task_id)
        return len(self.incomplete_dependencies(task)) == 0

    def claim(self, task_id: str, owner: str = "agent") -> Tuple[bool, str]:
        """认领任务：pending -> in_progress"""
        with self._lock:
            task = self.load(task_id)
            if task.status != DAGTaskStatus.PENDING.value:
                return False, f"Task {task_id} is {task.status}, cannot claim"
            unmet = self.incomplete_dependencies(task)
            if unmet:
                return False, f"Blocked by: {unmet}"
            task.owner = owner
            task.status = DAGTaskStatus.IN_PROGRESS.value
            self.save(task)
            return True, f"Claimed {task.id} ({task.subject})"

    def complete(self, task_id: str, owner: str = "agent") -> Tuple[bool, str, List[str]]:
        """
        完成任务：in_progress -> completed
        并自动扫描整张图，计算出此前被阻塞、由于当前任务完成而最新被解锁就绪的任务标题列表。
        """
        with self._lock:
            task = self.load(task_id)
            if task.status != DAGTaskStatus.IN_PROGRESS.value:
                return False, f"Task {task_id} is {task.status}, cannot complete", []
            if task.owner != owner:
                return False, f"Task {task_id} is owned by {task.owner}, not {owner}", []

            all_tasks_before = self.list()
            ready_before = {
                t.id
                for t in all_tasks_before
                if t.status == DAGTaskStatus.PENDING.value
                and t.blockedBy
                and self.can_start(t.id)
            }

            task.status = DAGTaskStatus.COMPLETED.value
            self.save(task)

            all_tasks_after = self.list()
            unblocked_subjects = [
                t.subject
                for t in all_tasks_after
                if t.status == DAGTaskStatus.PENDING.value
                and t.blockedBy
                and t.id not in ready_before
                and self.can_start(t.id)
            ]

            return True, f"Completed {task.id} ({task.subject})", unblocked_subjects


class DAGTaskTool:
    """
    DAG 任务管理工具插件 (符合 ToolPlugin 协议规范)
    向大模型提供 create_task, update_task, list_tasks, get_task, claim_task, complete_task 6 大标准工具。
    """

    def __init__(self, store: TaskStore):
        self.store = store

    def register_to(self, registry: ToolRegistry) -> None:
        """向 ToolRegistry 注册 6 大标准任务工具"""

        # 1. create_task
        def run_create_task(subject: str, description: str = "") -> str:
            try:
                task = self.store.create(subject=subject, description=description)
                return f"Created {task.id}: {task.subject}"
            except Exception as e:
                return f"Error: {e}"

        registry.register(
            name="create_task",
            description="创建新任务节点并返回运行时动态分配的全局唯一任务 ID。阶段一应先创建所有节点，随后使用返回的 ID 调用 update_task 绑定依赖。",
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "任务简短主题或标题"},
                    "description": {"type": "string", "description": "任务的详细实现规范或验收条件"},
                },
                "required": ["subject"],
                "additionalProperties": False,
            },
            handler=run_create_task,
        )

        # 2. update_task
        def run_update_task(task_id: str, addBlockedBy: List[str]) -> str:
            try:
                task = self.store.update_dependencies(task_id=task_id, add_blocked_by=addBlockedBy)
                deps_str = ", ".join(task.blockedBy) or "(none)"
                return f"Updated {task.id} blockedBy: {deps_str}"
            except Exception as e:
                return f"Error: {e}"

        registry.register(
            name="update_task",
            description="阶段二：使用 create_task 返回的真实 ID 为目标任务绑定前置依赖 (blockedBy)。系统会自动进行循环死锁 (Cycle Detection) 检查。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "需要添加依赖的目标任务 ID", "pattern": "^task_[0-9a-f]{8}$"},
                    "addBlockedBy": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^task_[0-9a-f]{8}$"},
                        "description": "该任务所依赖的前置任务 ID 列表（必须等待这些任务全部完成方可开始）",
                        "minItems": 1,
                    },
                },
                "required": ["task_id", "addBlockedBy"],
                "additionalProperties": False,
            },
            handler=run_update_task,
        )

        # 3. list_tasks
        def run_list_tasks() -> str:
            try:
                tasks = self.store.list()
                if not tasks:
                    return "No tasks. Use create_task to add some."
                lines = []
                for task in tasks:
                    marker = {
                        DAGTaskStatus.PENDING.value: "[ ]",
                        DAGTaskStatus.IN_PROGRESS.value: "[>]",
                        DAGTaskStatus.COMPLETED.value: "[x]",
                    }.get(task.status, "[?]")
                    dependencies = f" (blockedBy: {', '.join(task.blockedBy)})" if task.blockedBy else ""
                    owner = f" [{task.owner}]" if task.owner else ""
                    lines.append(f"{marker} {task.id}: {task.subject} [{task.status}]{owner}{dependencies}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error: {e}"

        registry.register(
            name="list_tasks",
            description="列出所有任务的状态看板，包含执行状态标记 ([ ]待办, [>]进行中, [x]已完成)、认领人以及前置依赖信息。",
            parameters={"type": "object", "properties": {}},
            handler=run_list_tasks,
        )

        # 4. get_task
        def run_get_task(task_id: str) -> str:
            try:
                task = self.store.load(task_id)
                return json.dumps(task.to_dict(), indent=2, ensure_ascii=False)
            except Exception as e:
                return f"Error: {e}"

        registry.register(
            name="get_task",
            description="根据任务 ID 获取单个任务的完整细节规范，包含详细描述与依赖列表。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "要查询的任务 ID"}
                },
                "required": ["task_id"],
            },
            handler=run_get_task,
        )

        # 5. claim_task
        def run_claim_task(task_id: str, owner: str = "agent") -> str:
            try:
                ok, msg = self.store.claim(task_id=task_id, owner=owner)
                return msg
            except Exception as e:
                return f"Error: {e}"

        registry.register(
            name="claim_task",
            description="认领一个前置依赖已全部完成的 pending 任务，将其状态置为 in_progress。若前置依赖未全部完成，系统将拒绝认领。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "要认领执行的任务 ID"}
                },
                "required": ["task_id"],
            },
            handler=run_claim_task,
        )

        # 6. complete_task
        def run_complete_task(task_id: str, owner: str = "agent") -> str:
            try:
                ok, msg, unblocked = self.store.complete(task_id=task_id, owner=owner)
                if unblocked:
                    msg += f"\nUnblocked: {', '.join(unblocked)}"
                return msg
            except Exception as e:
                return f"Error: {e}"

        registry.register(
            name="complete_task",
            description="将当前 Agent 认领的正在执行中的任务标记为 completed。系统会自动扫描下游并高亮提示最新被解锁的任务。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "已完成的任务 ID"}
                },
                "required": ["task_id"],
            },
            handler=run_complete_task,
        )
