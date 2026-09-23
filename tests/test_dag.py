"""
triumph-agent 有向无环图任务系统 (DAG Task System) 自动化测试
=============================================================
验证目标（对标 learn-claude-code s10_task_system）：
1. TaskStore 任务持久化落盘与工作区沙箱安全边界；
2. 两阶段 DAG 构建协议 (节点创建 -> 依赖边绑定)；
3. 拓扑图环路检测 (Cycle Detection) 与自依赖严密拦截；
4. 依赖门禁与认领守卫 (前置未全部 completed 禁止 claim)；
5. 任务完成与拓扑解锁实时计算 (Unblocked notification)；
6. 认领人归属一致性保护 (Owner Verification)；
7. DAGTaskTool 插件遵循 ToolPlugin 契约规范与 6 大工具分发闭环。
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from orchestration.dag import DAGTask, DAGTaskStatus, DAGTaskTool, TaskStore
from tools.registry import ToolRegistry


# ----------------------------------------------------------------------
# 1. 存储与沙箱安全测试
# ----------------------------------------------------------------------

def test_task_store_create_and_persistence(tmp_path):
    """验证任务创建、唯一 ID 分配与 JSON 文件落盘"""
    workdir = tmp_path
    store = TaskStore(workdir=workdir)

    task = store.create(subject="创建数据库 Schema", description="建立 users 和 orders 表")

    assert task.id.startswith("task_")
    assert len(task.id) == 13  # "task_" (5) + 8位hex = 13
    assert task.subject == "创建数据库 Schema"
    assert task.description == "建立 users 和 orders 表"
    assert task.status == DAGTaskStatus.PENDING.value
    assert task.owner is None
    assert task.blockedBy == []

    # 验证物理文件持久化
    task_file = workdir / ".tasks" / f"{task.id}.json"
    assert task_file.exists()

    loaded = store.load(task.id)
    assert loaded.id == task.id
    assert loaded.subject == task.subject
    assert loaded.description == task.description


def test_task_store_rejects_empty_subject(tmp_path):
    """验证空标题被安全拦截"""
    store = TaskStore(workdir=tmp_path)
    with pytest.raises(ValueError) as exc_info:
        store.create(subject="   ")
    assert "Task subject cannot be empty" in str(exc_info.value)


def test_task_store_sandbox_escape_protection(tmp_path):
    """验证通过符号链接指向工作区外被沙箱一票否决"""
    outside_dir = tmp_path / "outside_sandbox"
    outside_dir.mkdir()
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    # 在工作区下创建一个指向外部目录的 .tasks 软链接
    tasks_symlink = workdir / ".tasks"
    tasks_symlink.symlink_to(outside_dir, target_is_directory=True)

    store = TaskStore(workdir=workdir)

    with pytest.raises(ValueError) as exc_info:
        store.create(subject="unsafe task")
    assert "Task store escapes the workspace" in str(exc_info.value)
    assert list(outside_dir.iterdir()) == []


def test_task_store_invalid_task_id_paths(tmp_path):
    """验证路径穿越型非法 task_id 被拒绝"""
    store = TaskStore(workdir=tmp_path)
    with pytest.raises(ValueError) as exc_info:
        store.load("../outside")
    assert "Invalid task ID" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        store.load("invalid_id")
    assert "Invalid task ID" in str(exc_info.value)


# ----------------------------------------------------------------------
# 2. 两阶段 DAG 构建与依赖管理测试
# ----------------------------------------------------------------------

def test_task_store_two_phase_dag_construction(tmp_path):
    """验证两阶段建图：创建节点 -> 绑定依赖边"""
    store = TaskStore(workdir=tmp_path)

    # 阶段一：动态创建 4 个节点
    t_schema = store.create("设计 Schema")
    t_api = store.create("编写 API")
    t_tests = store.create("添加测试")
    t_docs = store.create("编写文档")

    # 阶段二：绑定依赖 (API 依赖 Schema，Tests 依赖 API，Docs 依赖 Schema)
    store.update_dependencies(t_api.id, [t_schema.id])
    store.update_dependencies(t_tests.id, [t_api.id])
    store.update_dependencies(t_docs.id, [t_schema.id])

    # 校验依赖持久化
    assert store.load(t_schema.id).blockedBy == []
    assert store.load(t_api.id).blockedBy == [t_schema.id]
    assert store.load(t_tests.id).blockedBy == [t_api.id]
    assert store.load(t_docs.id).blockedBy == [t_schema.id]


def test_task_store_dependencies_idempotent_dedup(tmp_path):
    """验证依赖添加是幂等的，且会自动去重"""
    store = TaskStore(workdir=tmp_path)
    t1 = store.create("Task 1")
    t2 = store.create("Task 2")

    store.update_dependencies(t2.id, [t1.id, t1.id])
    assert store.load(t2.id).blockedBy == [t1.id]

    # 再次调用同一依赖，保持单条
    store.update_dependencies(t2.id, [t1.id])
    assert store.load(t2.id).blockedBy == [t1.id]


def test_task_store_cycle_detection_and_self_dependency(tmp_path):
    """验证自依赖与传递闭包成环（Cycle Detection）被坚决拦截，且不留脏数据"""
    store = TaskStore(workdir=tmp_path)
    t1 = store.create("Task 1")
    t2 = store.create("Task 2")
    t3 = store.create("Task 3")

    # 1. 尝试自依赖
    with pytest.raises(ValueError) as exc_self:
        store.update_dependencies(t1.id, [t1.id])
    assert "Task cannot depend on itself" in str(exc_self.value)

    # 2. 尝试添加不存在的依赖
    with pytest.raises(ValueError) as exc_missing:
        store.update_dependencies(t1.id, ["task_00000000"])
    assert "Dependency not found" in str(exc_missing.value)

    # 3. 正常构建单向链: 1 -> 2 -> 3
    store.update_dependencies(t2.id, [t1.id])
    store.update_dependencies(t3.id, [t2.id])

    # 4. 尝试成环: 1 依赖 3 (构成 1 -> 2 -> 3 -> 1)
    with pytest.raises(ValueError) as exc_cycle:
        store.update_dependencies(t1.id, [t3.id])
    assert "Dependency cycle detected" in str(exc_cycle.value)
    # 验证 t1 没有被篡改产生脏数据
    assert store.load(t1.id).blockedBy == []


def test_task_store_cannot_update_dependencies_once_claimed(tmp_path):
    """验证任务一旦被认领或已执行，禁止随意篡改依赖"""
    store = TaskStore(workdir=tmp_path)
    t1 = store.create("Task 1")
    t2 = store.create("Task 2")

    # 认领 t1
    store.claim(t1.id, owner="agent_01")

    # 尝试修改 t1 的依赖
    with pytest.raises(ValueError) as exc_info:
        store.update_dependencies(t1.id, [t2.id])
    assert "can only be updated while pending and unowned" in str(exc_info.value)


# ----------------------------------------------------------------------
# 3. 依赖门禁与主动拓扑解锁计算测试
# ----------------------------------------------------------------------

def test_task_store_claim_blocked_and_unblock_on_completion(tmp_path):
    """验证门禁拦截 (未完成前禁止认领) 与完成时即时拓扑解锁通知"""
    store = TaskStore(workdir=tmp_path)

    schema = store.create("设计 Schema")
    api = store.create("编写 API")
    store.update_dependencies(api.id, [schema.id])

    # 1. 尝试认领尚未解锁的 api 任务 -> 拦截
    can_api, msg_api = store.claim(api.id, owner="agent")
    assert can_api is False
    assert f"Blocked by: ['{schema.id}']" in msg_api
    assert store.load(api.id).status == DAGTaskStatus.PENDING.value

    # 2. 认领前置任务 schema -> 成功
    can_schema, msg_schema = store.claim(schema.id, owner="agent")
    assert can_schema is True
    assert store.load(schema.id).status == DAGTaskStatus.IN_PROGRESS.value

    # 3. 完成 schema -> 触发拓扑解锁计算，返回 Unblocked
    ok, comp_msg, unblocked = store.complete(schema.id, owner="agent")
    assert ok is True
    assert store.load(schema.id).status == DAGTaskStatus.COMPLETED.value
    assert unblocked == ["编写 API"]

    # 4. 此时 api 任务的前置已全部 completed -> 认领成功
    can_api_now, msg_api_now = store.claim(api.id, owner="agent")
    assert can_api_now is True
    assert store.load(api.id).status == DAGTaskStatus.IN_PROGRESS.value


def test_task_store_owner_verification(tmp_path):
    """验证认领人归属一致性保护：非认领人禁止标记完成"""
    store = TaskStore(workdir=tmp_path)
    t = store.create("独占任务")

    store.claim(t.id, owner="agent_alice")

    # agent_bob 尝试偷跑标记完成
    ok, err_msg, _ = store.complete(t.id, owner="agent_bob")
    assert ok is False
    assert "owned by agent_alice, not agent_bob" in err_msg
    assert store.load(t.id).status == DAGTaskStatus.IN_PROGRESS.value

    # agent_alice 正常完成
    ok_real, _, _ = store.complete(t.id, owner="agent_alice")
    assert ok_real is True
    assert store.load(t.id).status == DAGTaskStatus.COMPLETED.value


def test_task_store_list_and_sorting(tmp_path):
    """验证任务看板列表按 ID 排序与状态输出"""
    store = TaskStore(workdir=tmp_path)
    t1 = store.create("任务 A")
    t2 = store.create("任务 B")

    tasks = store.list()
    assert len(tasks) == 2
    task_ids = [t.id for t in tasks]
    assert task_ids == sorted(task_ids)


# ----------------------------------------------------------------------
# 4. DAGTaskTool 插件规范与工具分发闭环测试
# ----------------------------------------------------------------------

def test_dag_task_tool_registration_and_specs(tmp_path):
    """验证 DAGTaskTool 遵循 ToolPlugin 契约协议并向 ToolRegistry 注册 6 大标准工具"""
    store = TaskStore(workdir=tmp_path)
    tool_plugin = DAGTaskTool(store=store)

    registry = ToolRegistry(workdir=tmp_path)
    registry.register_plugin(tool_plugin)

    specs = registry.get_tools_spec()
    tool_names = [s["function"]["name"] for s in specs]

    expected_tools = [
        "create_task",
        "update_task",
        "list_tasks",
        "get_task",
        "claim_task",
        "complete_task",
    ]
    for expected in expected_tools:
        assert expected in tool_names

    # 校验 update_task 的 Schema 规范
    update_spec = next(s for s in specs if s["function"]["name"] == "update_task")
    assert "task_id" in update_spec["function"]["parameters"]["properties"]
    assert "addBlockedBy" in update_spec["function"]["parameters"]["properties"]
    assert update_spec["function"]["parameters"]["required"] == ["task_id", "addBlockedBy"]


@pytest.mark.asyncio
async def test_dag_task_tool_end_to_end_execution(tmp_path):
    """模拟大模型通过 ToolRegistry 派发执行完整的 DAG 编排生命周期"""
    store = TaskStore(workdir=tmp_path)
    tool_plugin = DAGTaskTool(store=store)

    registry = ToolRegistry(workdir=tmp_path)
    registry.register_plugin(tool_plugin)

    # 1. 创建节点 1 与节点 2
    res_c1 = await registry.execute("create_task", {"subject": "准备环境", "description": "配置 Python 虚拟环境"})
    assert res_c1.startswith("Created task_")
    tid_1 = res_c1.split()[1].rstrip(":")

    res_c2 = await registry.execute("create_task", {"subject": "执行测试", "description": "运行 pytest"})
    assert res_c2.startswith("Created task_")
    tid_2 = res_c2.split()[1].rstrip(":")

    # 2. 添加依赖: tid_2 依赖 tid_1
    res_update = await registry.execute("update_task", {"task_id": tid_2, "addBlockedBy": [tid_1]})
    assert f"Updated {tid_2} blockedBy: {tid_1}" in res_update

    # 3. 查看看板
    res_list = await registry.execute("list_tasks", {})
    assert "[ ]" in res_list
    assert f"(blockedBy: {tid_1})" in res_list

    # 4. 尝试提前认领 tid_2 (受阻)
    res_claim_bad = await registry.execute("claim_task", {"task_id": tid_2})
    assert f"Blocked by: ['{tid_1}']" in res_claim_bad

    # 5. 认领并完成 tid_1
    res_claim_1 = await registry.execute("claim_task", {"task_id": tid_1})
    assert f"Claimed {tid_1}" in res_claim_1

    res_comp_1 = await registry.execute("complete_task", {"task_id": tid_1})
    assert f"Completed {tid_1}" in res_comp_1
    assert "Unblocked: 执行测试" in res_comp_1

    # 6. 顺畅认领 tid_2
    res_claim_2 = await registry.execute("claim_task", {"task_id": tid_2})
    assert f"Claimed {tid_2}" in res_claim_2

    # 7. 查看单任务详情 get_task
    res_get = await registry.execute("get_task", {"task_id": tid_2})
    task_data = json.loads(res_get)
    assert task_data["id"] == tid_2
    assert task_data["status"] == DAGTaskStatus.IN_PROGRESS.value
    assert task_data["owner"] == "agent"
