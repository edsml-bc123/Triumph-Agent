"""
triumph-agent 三层记忆系统 (Memory Operating Engine) 自动化测试
===============================================================
验证目标：
1. Storage 层：YAML frontmatter 解析与序列化、安全防逃逸、MEMORY.md 索引自动重构；
2. 门禁过滤器 should_store_memory：临时词过滤、非 persistent 作用域阻断、多维查重；
3. Recall 召回管线：两阶段选号、字符预算限制、关键词重合度降级兜底、Prompt 注入与优先级声明；
4. Consolidation 整理引擎：阈值触发合并、快照备份与崩溃原子回滚；
5. AOP 切面插件 MemoryHook：UserPromptSubmit 召回注入、Stop 回合结束知识沉淀。
"""

from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock
import pytest

from memory import (
    CandidateMemory,
    MemoryHook,
    MemoryManager,
    MemoryRecord,
    MemoryScope,
    MemoryStorage,
    MemoryType,
    format_document,
    memory_slug,
    parse_frontmatter,
    should_store_memory,
)
from runtime.hooks import HookManager
from runtime.state import AgentState, AgentStatus


# ----------------------------------------------------------------------
# 1. 存储层与路径安全测试
# ----------------------------------------------------------------------

def test_parse_frontmatter_and_format_document():
    doc = format_document(
        name="User Code Style",
        mem_type="user",
        description="User prefers 4 spaces indentation",
        body="Always use 4 spaces for python code.",
    )
    assert doc.startswith("---\n")
    metadata, body = parse_frontmatter(doc)
    assert metadata["name"] == "User Code Style"
    assert metadata["type"] == "user"
    assert metadata["description"] == "User prefers 4 spaces indentation"
    assert body == "Always use 4 spaces for python code."

    # 容错：无 frontmatter
    meta, plain = parse_frontmatter("Just pure markdown")
    assert meta == {}
    assert plain == "Just pure markdown"


def test_storage_write_read_and_rebuild_index(tmp_path):
    storage = MemoryStorage(workdir=tmp_path)
    storage.write_memory(
        name="Docker Sandbox Architecture",
        mem_type="project",
        description="Sandbox uses gVisor runtime",
        body="MicroVM isolates all non-trusted bash executions.",
    )

    records = storage.list_memories()
    assert len(records) == 1
    assert records[0].name == "Docker Sandbox Architecture"
    assert records[0].type == "project"

    # 验证 MEMORY.md 索引目录已生成
    index_content = storage.read_index()
    assert "- [Docker Sandbox Architecture](docker-sandbox-architecture.md) -" in index_content
    assert "Sandbox uses gVisor runtime" in index_content

    # 验证单条读取
    content = storage.read_memory("docker-sandbox-architecture.md")
    assert content is not None
    assert "MicroVM isolates" in content

    # 验证删除后索引自动同步
    assert storage.delete_memory("docker-sandbox-architecture.md") is True
    assert len(storage.list_memories()) == 0
    assert storage.read_index() == ""


def test_storage_path_traversal_protection(tmp_path):
    storage = MemoryStorage(workdir=tmp_path)

    # 禁止路径包含目录层级
    with pytest.raises(ValueError, match="非法记忆文件名"):
        storage.resolve_path("../escape.md")

    with pytest.raises(ValueError, match="非法记忆文件名"):
        storage.resolve_path("sub/folder.md")

    # 默认禁止直接读写 MEMORY.md 索引文件
    with pytest.raises(ValueError, match="MEMORY.md 为系统专用索引文件"):
        storage.resolve_path("MEMORY.md", allow_index=False)


# ----------------------------------------------------------------------
# 2. 门禁过滤器 should_store_memory 测试
# ----------------------------------------------------------------------

def test_should_store_memory_filters():
    existing = [
        MemoryRecord(
            name="Tab Indentation",
            type="user",
            description="Use tabs for indentation",
            body="Tabs only.",
            filename="tab-indentation.md",
        )
    ]

    # 1. 成功持久化放行
    valid_candidate = {
        "name": "Strict Typing",
        "type": "feedback",
        "scope": "persistent",
        "description": "Always annotate python functions",
        "body": "Use mypy compatible type hints for public APIs.",
    }
    assert should_store_memory(valid_candidate, existing) is True

    # 2. 拒绝临时任务作用域
    temp_scope_candidate = dict(valid_candidate, scope="current_task")
    assert should_store_memory(temp_scope_candidate, existing) is False

    # 3. 拒绝非法类型
    bad_type_candidate = dict(valid_candidate, type="temporary_fact")
    assert should_store_memory(bad_type_candidate, existing) is False

    # 4. 拒绝重复（同 slug、同 description 或同 body）
    duplicate_slug = dict(valid_candidate, name="tab-indentation")
    assert should_store_memory(duplicate_slug, existing) is False

    duplicate_desc = dict(valid_candidate, name="Different Name", description="Use tabs for indentation")
    assert should_store_memory(duplicate_desc, existing) is False

    duplicate_body = dict(valid_candidate, name="Another Name", body="Tabs only.")
    assert should_store_memory(duplicate_body, existing) is False

    # 5. 支持显式 CandidateMemory 实体传入
    candidate_obj = CandidateMemory.from_raw(valid_candidate)
    assert candidate_obj is not None
    assert should_store_memory(candidate_obj, existing) is True

    # 6. 非法原始输入由 CandidateMemory.from_raw 安全拦截
    assert CandidateMemory.from_raw("not-a-dict") is None
    assert CandidateMemory.from_raw({"name": "missing fields"}) is None
    assert CandidateMemory.from_raw(dict(valid_candidate, type="invalid_type")) is None


@pytest.mark.asyncio
async def test_judge_candidate_durability_evaluator(tmp_path):
    """
    验证基于 EVALUATOR_MODEL (qwen3.6-flash) 的裁判小模型二分类裁决
    """
    manager = MemoryManager(workdir=tmp_path)
    candidate = CandidateMemory(
        name="pytest rule",
        type=MemoryType.FEEDBACK,
        scope=MemoryScope.PERSISTENT,
        description="Run pytest before submit",
        body="Always execute pytest.",
    )

    mock_client = MagicMock()

    # 1. 裁判判定为长期通用知识 (DURABLE) -> 放行
    mock_client.chat_completion = AsyncMock(return_value=MagicMock(content="DURABLE"))
    assert await manager.judge_candidate_durability(candidate, mock_client) is True

    # 2. 裁判判定为临时单次指令 (TRANSIENT) -> 拦截
    mock_client.chat_completion = AsyncMock(return_value=MagicMock(content="TRANSIENT"))
    assert await manager.judge_candidate_durability(candidate, mock_client) is False

    # 3. 裁判调用异常时优雅容灾放行
    mock_client.chat_completion = AsyncMock(side_effect=RuntimeError("API error"))
    assert await manager.judge_candidate_durability(candidate, mock_client) is True


# ----------------------------------------------------------------------
# 3. 召回管线测试 (两阶段模型选号 + 关键词降级)
# ----------------------------------------------------------------------

def test_keyword_memory_selection(tmp_path):
    records = [
        MemoryRecord(
            name="PostgreSQL Config",
            type="project",
            description="Database runs on port 5432 with schema public",
            body="",
            filename="postgresql-config.md",
        ),
        MemoryRecord(
            name="Redis Cache Strategy",
            type="project",
            description="Cache expired after 300 seconds",
            body="",
            filename="redis-cache.md",
        ),
        MemoryRecord(
            name="代码缩进偏好",
            type="user",
            description="用户偏好在 Python 中使用 4 个空格缩进",
            body="",
            filename="python-indent.md",
        ),
    ]

    # 英文关键词匹配
    selected_pg = MemoryManager.keyword_memory_selection(records, "How to connect to PostgreSQL database?", max_items=2)
    assert selected_pg == ["postgresql-config.md"]

    # 中文关键词匹配
    selected_cn = MemoryManager.keyword_memory_selection(records, "请帮我调整代码缩进格式", max_items=2)
    assert selected_cn == ["python-indent.md"]


@pytest.mark.asyncio
async def test_two_phase_model_recall_and_budget(tmp_path):
    storage = MemoryStorage(workdir=tmp_path)
    storage.write_memory(
        name="Auth Strategy",
        mem_type="project",
        description="JWT tokens expire in 24h",
        body="Secret key is stored in vault.",
    )
    storage.write_memory(
        name="Docker Setup",
        mem_type="project",
        description="Container runs unprivileged user",
        body="UID 1000 is used.",
    )

    manager = MemoryManager(workdir=tmp_path, storage=storage, recall_char_limit=500)

    # Mock 客户端返回模型选号 JSON: [0]
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "Based on your request, I select [0] because it covers auth."
    mock_client.chat_completion = AsyncMock(return_value=mock_resp)

    messages = [{"role": "user", "content": "Tell me about authentication"}]
    loaded = await manager.load_memories(messages, mock_client)

    assert len(loaded) == 1
    assert loaded[0]["source"] == "auth-strategy.md"
    assert "Secret key is stored in vault." in loaded[0]["content"]

    # 验证注入 System Prompt 的内容包含优先级契约
    prompt_section = manager.build_memory_prompt_section(loaded)
    assert "### 长期记忆索引目录" in prompt_section
    assert "### 针对当前请求召回的相关记忆正文" in prompt_section
    assert "【记忆优先级契约】" in prompt_section
    assert "最高优先级" in prompt_section


# ----------------------------------------------------------------------
# 4. 快照备份与崩溃原子回滚测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consolidation_atomic_rollback_on_failure(tmp_path):
    storage = MemoryStorage(workdir=tmp_path)
    # 创建 10 条初始记忆以达到 consolidate_threshold
    for i in range(10):
        storage.write_memory(
            name=f"Rule {i}",
            mem_type="feedback",
            description=f"Description for rule {i}",
            body=f"Body content {i}",
        )

    assert len(storage.list_memories()) == 10
    manager = MemoryManager(workdir=tmp_path, storage=storage, consolidate_threshold=10)

    # 模拟模型给出了包含异常数据的响应（例如写入第二条时触发底层异常）
    mock_client = MagicMock()
    mock_resp = MagicMock()
    # 返回合法的 JSON，但在 write_memory 模拟引发 I/O 错误
    mock_resp.content = (
        '[{"name": "Merged Rule 1", "type": "feedback", "description": "d1", "body": "b1"}, '
        '{"name": "Merged Rule 2", "type": "feedback", "description": "d2", "body": "b2"}]'
    )
    mock_client.chat_completion = AsyncMock(return_value=mock_resp)

    original_write = storage.write_memory
    call_count = {"count": 0}

    def flaky_write(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise OSError("Simulated disk write failure during consolidation")
        return original_write(*args, **kwargs)

    storage.write_memory = flaky_write

    # 执行整理，发生异常应被安全捕获或回滚
    with pytest.raises(OSError, match="Simulated disk write failure"):
        await manager.consolidate_memories(mock_client)

    # 恢复 write_memory 引用
    storage.write_memory = original_write

    # 验证原子回滚：原先的 10 条记忆必须 100% 完整无损！
    restored_records = storage.list_memories()
    assert len(restored_records) == 10
    assert any(r.name == "Rule 0" for r in restored_records)
    assert any(r.name == "Rule 9" for r in restored_records)


# ----------------------------------------------------------------------
# 5. MemoryHook AOP 切面全生命周期集成测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_hook_lifecycle(tmp_path):
    storage = MemoryStorage(workdir=tmp_path)
    manager = MemoryManager(workdir=tmp_path, storage=storage)

    # 预置一条长期偏好
    storage.write_memory(
        name="User Tone",
        mem_type="user",
        description="User prefers concise responses",
        body="Do not include fluff, get straight to the point.",
    )

    mock_client = MagicMock()
    # 模拟 UserPromptSubmit 阶段两阶段选号
    mock_recall_resp = MagicMock()
    mock_recall_resp.content = "[0]"

    # 模拟 Stop 阶段自动提炼新记忆
    mock_extract_resp = MagicMock()
    mock_extract_resp.content = (
        '[{"name": "No Mock DB", "type": "feedback", "scope": "persistent", '
        '"description": "Never mock the real postgres database", "body": "Always test against actual dockerized DB."}]'
    )

    # 模拟裁判小模型二分类判定
    mock_evaluator_resp = MagicMock()
    mock_evaluator_resp.content = "DURABLE"

    mock_client.chat_completion = AsyncMock(side_effect=[mock_recall_resp, mock_extract_resp, mock_evaluator_resp])

    hook = MemoryHook(manager=manager, client=mock_client)
    hook_mgr = HookManager()
    hook.register_to(hook_mgr)

    # 1. 模拟 UserPromptSubmit 切面
    state = AgentState(max_steps=5)
    state.messages.append({"role": "system", "content": "You are a coding agent."})
    state.messages.append({"role": "user", "content": "Please review my PR."})

    await hook_mgr.trigger("UserPromptSubmit", prompt="Please review my PR.", state=state)

    # 验证 System Prompt 已注入记忆与契约
    sys_content = state.messages[0]["content"]
    assert "User Tone" in sys_content
    assert "【记忆优先级契约】" in sys_content

    # 2. 模拟任务执行完毕并正常达到终态
    state.mark_success("PR review completed.")
    await hook_mgr.trigger("Stop", state=state)

    # 验证通过 Stop 切面自动提炼并持久化了新记忆
    current_memories = storage.list_memories()
    assert len(current_memories) == 2
    assert any(m.name == "No Mock DB" for m in current_memories)


def test_session_multi_turn_continuation():
    """
    验证 Outer Loop 会话级多轮上下文延续机制：
    任务 1 产生对话历史后，任务 2 能够平滑继承历史消息，保证上下文不丢失。
    """
    session_messages = []

    # 模拟第一轮会话
    state1 = AgentState(max_steps=5)
    state1.add_user_message("帮我编写一个计算斐波那契数列的函数")
    # 模拟模型回复
    mock_resp_content = "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"
    state1.messages.append({"role": "assistant", "content": mock_resp_content})
    state1.mark_success(mock_resp_content)

    # 任务 1 成功后，同步至 session_messages
    session_messages = list(state1.messages)
    assert len(session_messages) == 2

    # 模拟第二轮会话：用户基于前文继续提问
    state2 = AgentState(max_steps=5)
    state2.messages.extend(session_messages)
    state2.add_user_message("把它改写成动态规划的迭代版本")

    # 验证第二轮状态机拥有前文所有上下文（含第 1 轮 user、第 1 轮 assistant 以及第 2 轮 user）
    assert len(state2.messages) == 3
    assert state2.messages[0]["role"] == "user"
    assert "斐波那契" in state2.messages[0]["content"]
    assert state2.messages[1]["role"] == "assistant"
    assert "def fib" in state2.messages[1]["content"]
    assert state2.messages[2]["role"] == "user"
    assert "动态规划" in state2.messages[2]["content"]

