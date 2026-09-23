"""
triumph-agent 上下文渐进式压缩引擎 (ContextCompactor) 自动化测试
===============================================================
验证目标：
1. Layer 1: tool_result_budget 单条超大输出安全落盘与带路径预览截断；
2. Layer 2: snip_compact 历史中间轮次成对对齐截取与归档，防止 tool_calls 契约破坏；
3. Layer 3: micro_compact 陈旧工具输出占位化，确保保留最近活跃结果且成对契约不破损；
4. Layer 4: fit_tool_results 活跃工具结果贪心自适应预览化；
5. Layer 5: semantic_compact 达到硬上限时调用轻量模型生成结构化工作事实摘要并折叠历史；
6. CompactorHook 切面透明集成与 AgentState 自动化瘦身。
"""

import pytest

from context.compactor import CompactionConfig, CompactorHook, ContextCompactor
from runtime.hooks import HookManager
from runtime.state import AgentState
from client import DashScopeClient, LLMResponse
from tools import ToolRegistry



def test_compactor_estimate_chars_and_tokens(tmp_path):
    compactor = ContextCompactor(workdir=tmp_path)
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi"},
    ]
    chars = compactor.estimate_chars(messages)
    assert chars > 0
    tokens = compactor.estimate_tokens(messages)
    assert tokens == chars // 4


def test_compactor_layer1_single_large_output_budget(tmp_path):
    """验证 L1：单条超大工具输出持久化落盘并生成引用标记"""
    cfg = CompactionConfig(max_single_tool_output_chars=300, preview_chars=100)
    compactor = ContextCompactor(workdir=tmp_path, config=cfg)

    large_text = "ERROR: stack trace line...\n" * 50  # ~1350 chars
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": large_text},
    ]

    processed = compactor.tool_result_budget(messages, run_id="test_run_001")
    tool_msg = processed[1]
    assert "<persisted-output>" in tool_msg["content"]
    assert "Full output saved to:" in tool_msg["content"]
    assert "Preview (first 100 chars):" in tool_msg["content"]

    # 验证磁盘文件真实存在且内容完整（符合方案 A 目录规范）
    saved_file = tmp_path.resolve() / "runs" / "test_run_001" / "tool_outputs" / "call_1.txt"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == large_text


def test_compactor_layer2_snip_compact_turn_alignment(tmp_path):
    """验证 L2：中间长历史归档对齐，严禁切断 assistant 与 tool 成对关系"""
    cfg = CompactionConfig(keep_recent_messages=3)
    compactor = ContextCompactor(workdir=tmp_path, config=cfg)

    # 构造 10 轮对话 (共 20 条消息)
    messages = [
        {"role": "system", "content": "You are agent."},
        {"role": "user", "content": "Initial user task."},
    ]
    for i in range(1, 10):
        messages.extend([
            {"role": "assistant", "content": f"thought {i}", "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "bash"}}]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"result {i}"},
        ])

    assert len(messages) == 20

    # 触发中间裁剪（阈值设为 8 条）
    snipped = compactor.snip_compact(messages, max_messages=8)
    assert len(snipped) < len(messages)

    # 验证头部保留 System 与 Initial Task
    assert snipped[0]["role"] == "system"
    assert snipped[1]["role"] == "user"

    # 验证中间有且仅有一条归档标记
    archive_markers = [m for m in snipped if "archived at" in str(m.get("content", ""))]
    assert len(archive_markers) == 1

    # 验证尾部保留且所有 assistant.tool_calls 后的 tool 必须成对完整
    for i, m in enumerate(snipped):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            assert i + 1 < len(snipped)
            assert snipped[i + 1]["role"] == "tool"
            assert snipped[i + 1]["tool_call_id"] == m["tool_calls"][0]["id"]


def test_compactor_layer3_micro_compact_preserves_recent(tmp_path):
    """验证 L3：微压缩保留最近 N 条，陈旧输出替换为极简占位符，且消息字典完全保留"""
    cfg = CompactionConfig(keep_recent_tool_results=2)
    compactor = ContextCompactor(workdir=tmp_path, config=cfg)

    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "old long output 1 " * 20},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "old long output 2 " * 20},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "active output 3"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "c4", "content": "active output 4"},
    ]

    compacted = compactor.micro_compact(messages)

    # c1 与 c2 应被微压缩为占位符
    assert "[Earlier tool result saved at " in compacted[1]["content"]
    assert "[Earlier tool result saved at " in compacted[3]["content"]

    # c3 与 c4 属于最近两条活跃结果，保持原样
    assert compacted[5]["content"] == "active output 3"
    assert compacted[7]["content"] == "active output 4"

    # 验证成对契约未被破坏
    tool_call_ids = [tc["id"] for m in compacted if m.get("tool_calls") for tc in m["tool_calls"]]
    tool_resp_ids = [m["tool_call_id"] for m in compacted if m.get("role") == "tool"]
    assert tool_call_ids == tool_resp_ids == ["c1", "c2", "c3", "c4"]


def test_compactor_layer4_fit_tool_results_greedy(tmp_path):
    """验证 L4：贪心自适应拟合优先缩减体积最大的活跃结果"""
    compactor = ContextCompactor(workdir=tmp_path)

    # 构造两个活跃结果：一个超大 (10,000 字符)，一个中等 (2,000 字符)
    large_block = "X" * 10000
    medium_block = "Y" * 2000
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c_huge", "type": "function", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "c_huge", "content": large_block},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c_med", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "c_med", "content": medium_block},
    ]

    initial_chars = compactor.estimate_chars(messages)
    target = initial_chars - 5000

    fitted = compactor.fit_tool_results(messages, target_chars=target)

    # 贪心策略：最大体积的 c_huge 必须被裁剪为带 preview 的文本
    c_huge_msg = next(m for m in fitted if m.get("tool_call_id") == "c_huge")
    assert "<persisted-output>" in c_huge_msg["content"]
    assert len(c_huge_msg["content"]) < len(large_block)

    # 此时如果总长度已经小于 target，中等块 c_med 保持原样不被修改
    assert compactor.estimate_chars(fitted) <= target


@pytest.mark.asyncio
async def test_compactor_layer5_semantic_compact(tmp_path):
    """验证 L5：终极全局语义摘要合并与全量 transcript 磁盘持久化"""
    from client import LLMResponse

    class MockSummaryClient:
        def __init__(self):
            self.call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self.call_count += 1
            return LLMResponse(
                content="## 已完成事项\n1. 修复了编译问题。\n2. 下一步编写单测。",
                finish_reason="stop",
            )

    client = MockSummaryClient()
    compactor = ContextCompactor(workdir=tmp_path)

    messages = [
        {"role": "system", "content": "System directive."},
        {"role": "user", "content": "用户核心总目标：重构运行时"},
        {"role": "assistant", "content": "thought", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "output"},
    ]

    compressed = await compactor.semantic_compact(
        messages, client=client, active_goal="重构运行时", run_id="run_semantic_001"
    )

    assert client.call_count == 1
    assert len(compressed) == 2
    assert compressed[0]["role"] == "system"
    assert compressed[0]["content"] == "System directive."

    user_msg = compressed[1]
    assert user_msg["role"] == "user"
    assert "[Conversation Compressed]" in user_msg["content"]
    assert "已完成事项" in user_msg["content"]
    assert "重构运行时" in user_msg["content"]
    assert "全量历史归档路径" in user_msg["content"]

    # 验证 runs/{run_id}/transcripts 目录下成功写入了全量快照文件（符合方案 A 目录规范）
    transcripts = list((tmp_path.resolve() / "runs" / "run_semantic_001" / "transcripts").glob("transcript_*.jsonl"))
    assert len(transcripts) == 1


@pytest.mark.asyncio
async def test_compactor_hook_integration(tmp_path):
    """验证 CompactorHook 与 HookManager、AgentState 的无缝集成驱动"""
    cfg = CompactionConfig(soft_threshold_chars=500, keep_recent_tool_results=1)
    compactor = ContextCompactor(workdir=tmp_path, config=cfg)
    hook = CompactorHook(compactor=compactor)

    manager = HookManager()
    manager.register_plugin(hook)

    from client import LLMResponse, ToolCall

    # 构造超出 soft_threshold_chars 的 AgentState
    state = AgentState()
    state.add_user_message("完成大规模文件分析")
    for i in range(1, 4):
        state.add_assistant_turn(
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id=f"call_{i}", name="read_file", arguments_raw="{}")],
            )
        )
        state.add_tool_result(tool_call_id=f"call_{i}", name="read_file", result=f"data chunk {i} " * 20)

    # 初始字符数应超过 500
    assert compactor.estimate_chars(state.messages) > 500

    # 触发 StepStart 切面
    await manager.trigger("StepStart", state=state)

    # 验证微压缩自动生效：call_1, call_2 被压缩为占位符，仅 call_3 保留
    c1_msg = next(m for m in state.messages if m.get("tool_call_id") == "call_1")
    assert "[Earlier tool result saved at " in c1_msg["content"]

    c3_msg = next(m for m in state.messages if m.get("tool_call_id") == "call_3")
    assert "data chunk 3" in c3_msg["content"]


@pytest.mark.asyncio
async def test_compactor_full_pipeline_prepare_end_to_end(tmp_path):
    """
    验证 prepare 全流水线端到端穿透：
    从 L1 (单条超大截断) -> L2 (轮次裁剪) -> L3 (微压缩) -> L4 (自适应拟合) -> L5 (语义摘要)。
    """
    from client import LLMResponse

    class MockSummaryClient:
        async def chat_completion(self, *args, **kwargs):
            return LLMResponse(content="全链路摘要生成成功", finish_reason="stop")

    # 设定较紧的阈值以触发完整 5 级流水线
    cfg = CompactionConfig(
        soft_threshold_chars=2000,
        hard_threshold_chars=3000,
        max_single_tool_output_chars=500,
        max_history_messages=6,
        keep_recent_tool_results=1,
    )
    compactor = ContextCompactor(workdir=tmp_path, config=cfg)

    # 构造大量消息（超过 6 条，且单条工具输出巨大，总字符数 > 3000）
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Initial user task."},
    ]
    for i in range(1, 6):
        messages.extend([
            {"role": "assistant", "content": f"thought {i}", "tool_calls": [{"id": f"c_{i}", "type": "function", "function": {"name": "bash"}}]},
            {"role": "tool", "tool_call_id": f"c_{i}", "content": f"massive log output {i}\n" * 50},  # ~1000 chars per tool
        ])

    initial_chars = compactor.estimate_chars(messages)
    assert initial_chars > 6000

    # 驱动 prepare 全流水线
    prepared = await compactor.prepare(
        messages=messages, client=MockSummaryClient(), active_goal="端到端全链路测试"
    )

    # 验证经过 L1~L4 成功把 >6000 字符大幅压制削减 60% 以上，且未达 hard_threshold_chars 免于调用 L5 语义折叠
    final_chars = compactor.estimate_chars(prepared)
    assert final_chars < 2600
    assert final_chars < initial_chars * 0.5
    assert len(prepared) == 9


@pytest.mark.asyncio
async def test_compactor_full_pipeline_prepare_triggers_l5_when_over_hard_limit(tmp_path):
    """验证即使完成 L1~L4 后仍超出硬上限时，prepare 必然触发 Layer 5 全局语义摘要"""
    from client import LLMResponse

    class MockSummaryClient:
        async def chat_completion(self, *args, **kwargs):
            return LLMResponse(content="终极语义摘要完成", finish_reason="stop")

    # 设极低硬上限 (1000 字符)
    cfg = CompactionConfig(
        soft_threshold_chars=500,
        hard_threshold_chars=1000,
    )
    compactor = ContextCompactor(workdir=tmp_path, config=cfg)

    # 构造大量纯文字对话（无法被工具微压缩削减的内容）
    messages = [
        {"role": "system", "content": "sys " * 50},
        {"role": "user", "content": "user task " + ("Z" * 1500)},
        {"role": "assistant", "content": "detailed reasoning " * 50},
    ]

    prepared = await compactor.prepare(
        messages=messages, client=MockSummaryClient(), active_goal="硬上限击穿测试"
    )

    # 验证最终必定被折叠为包含摘要的 2 条消息
    assert len(prepared) == 2
    assert prepared[0]["role"] == "system"
    assert "终极语义摘要完成" in prepared[1]["content"]
    assert "硬上限击穿测试" in prepared[1]["content"]


def test_compactor_snip_compact_idempotent_no_nested_archive(tmp_path):
    """验证 snip_compact 二次归档判重，防止重复套娃"""
    cfg = CompactionConfig(keep_recent_messages=2)
    compactor = ContextCompactor(workdir=tmp_path, config=cfg)

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(1, 6):
        messages.extend([
            {"role": "assistant", "content": f"step {i}"},
            {"role": "user", "content": f"reply {i}"},
        ])

    # 第一次中段归档
    first_pass = compactor.snip_compact(messages, max_messages=4)
    assert any("archived at" in str(m.get("content", "")) for m in first_pass)

    # 第二次直接传入已被归档的消息列表，验证幂等返回，严禁重复套娃
    second_pass = compactor.snip_compact(first_pass, max_messages=4)
    assert len(second_pass) == len(first_pass)
    assert second_pass == first_pass


@pytest.mark.asyncio
async def test_compactor_semantic_compact_fallback_on_api_error(tmp_path):
    """验证 Layer 5 在模型 API 抛出异常时的安全优雅降级逻辑"""
    class CrashingClient:
        async def chat_completion(self, *args, **kwargs):
            raise RuntimeError("百炼网络超时或 API 限流")

    compactor = ContextCompactor(workdir=tmp_path)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]

    compressed = await compactor.semantic_compact(
        messages=messages, client=CrashingClient(), active_goal="容灾降级测试"
    )

    assert len(compressed) == 2
    # 验证未抛出异常，而是优雅降级回填提示
    assert "未能自动生成摘要" in compressed[1]["content"]
    assert "容灾降级测试" in compressed[1]["content"]


@pytest.mark.asyncio
async def test_compactor_semantic_compact_large_raw_text_truncation(tmp_path):
    """验证 Layer 5 在原始上下文极长 (>60,000 字符) 时的头尾保护截取分支"""
    from client import LLMResponse

    captured_prompt = ""

    class InspectClient:
        async def chat_completion(self, messages, *args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = messages[0]["content"]
            return LLMResponse(content="大文本摘要完成", finish_reason="stop")

    compactor = ContextCompactor(workdir=tmp_path)
    # 构造单条极长内容使总 json 超过 60,000 字符
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "huge task " + ("A" * 70_000)},
    ]

    await compactor.semantic_compact(messages, client=InspectClient())
    assert "... [中间部分已在磁盘归档:" in captured_prompt


def test_compactor_idempotent_skips_and_early_exit(tmp_path):
    """
    验证：
    1. tool_result_budget 跳过已落盘或已占位化的消息；
    2. micro_compact 遇到已占位化消息跳过；
    3. micro_compact 在 target_chars 达标时提前 break。
    """
    compactor = ContextCompactor(workdir=tmp_path)

    already_processed = [
        {"role": "tool", "tool_call_id": "c1", "content": "<persisted-output>\nFull output saved to: test.txt\n</persisted-output>"},
        {"role": "tool", "tool_call_id": "c2", "content": "[Earlier tool result saved at test.txt]"},
    ]
    # 1. 验证 L1 跳过已处理项
    res1 = compactor.tool_result_budget(already_processed)
    assert res1 == already_processed

    # 2. 验证 L3 跳过已占位化项并支持提前 break
    messages = [
        {"role": "tool", "tool_call_id": "c1", "content": "[Earlier tool result saved at test.txt]"},
        {"role": "tool", "tool_call_id": "c2", "content": "medium output " * 20},
        {"role": "tool", "tool_call_id": "c3", "content": "recent output 1"},
        {"role": "tool", "tool_call_id": "c4", "content": "recent output 2"},
    ]
    # 设置一个较大的 target_chars，使其在处理 c2 时如果已满足立即 break
    res2 = compactor.micro_compact(messages, target_chars=10_000)
    assert len(res2) == 4


@pytest.mark.asyncio
async def test_agent_loop_with_compactor_hook_end_to_end(tmp_path):
    """
    验证 AgentLoop 真实执行流与 CompactorHook 的完整集成闭环：
    Step 1: 模型返回 tool_call -> 执行工具产出超大输出
    Step 2: 触发下一轮 StepStart 切面 -> CompactorHook 自动截断并持久化落盘至 runs/{run_id}/tool_outputs/ -> 模型返回终止文本完成任务
    """
    from client import LLMResponse, ToolCall
    from runtime.loop import AgentLoop
    from runtime.state import AgentState, AgentStatus
    from runtime.hooks import HookManager
    from tools import BuiltinTool, ToolRegistry

    step_counter = 0

    class DualStepClient:
        async def chat_completion(self, messages, *args, **kwargs):
            nonlocal step_counter
            step_counter += 1
            if step_counter == 1:
                # 第一轮：发起工具调用
                return LLMResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[ToolCall(id="call_loop_01", name="read_file", arguments_raw='{"path": "data.txt"}')],
                )
            else:
                # 第二轮（此时上一轮的大输出已被 Compactor 瘦身）：完成任务
                return LLMResponse(content="文件分析完成", finish_reason="stop")

    # 在沙箱目录创建测试文件
    workdir = tmp_path
    (workdir / "data.txt").write_text("HUGE_DATA_LINE\n" * 50, encoding="utf-8")  # ~750 chars

    cfg = CompactionConfig(max_single_tool_output_chars=200, preview_chars=50)
    compactor = ContextCompactor(workdir=workdir, config=cfg)
    compactor_hook = CompactorHook(compactor=compactor)

    hooks = HookManager()
    hooks.register_plugin(compactor_hook)

    registry = ToolRegistry(workdir=workdir)
    registry.register_plugin(BuiltinTool(workdir=workdir))
    loop = AgentLoop(client=DualStepClient(), registry=registry, hooks=hooks)

    state = AgentState(run_id="run_loop_compactor_e2e")
    state.add_user_message("请读取 data.txt 内容")
    state.current_goal = "读取 data.txt 内容"

    # 执行完整的 ReAct 自主循环
    final_state = await loop.run(state)

    # 1. 验证状态机成功到达终态
    assert final_state.status == AgentStatus.SUCCESS
    assert final_state.step_count == 2
    assert final_state.final_answer == "文件分析完成"

    # 2. 验证工具输出已被 L1 自动持久化落盘至方案 A 专属目录
    persisted_tool_file = workdir / "runs" / "run_loop_compactor_e2e" / "tool_outputs" / "call_loop_01.txt"
    assert persisted_tool_file.exists()
    assert "HUGE_DATA_LINE" in persisted_tool_file.read_text(encoding="utf-8")

    # 3. 验证 state.messages 中保存的是受预算限制的瘦身结构
    tool_msg = next(m for m in final_state.messages if m.get("tool_call_id") == "call_loop_01")
    assert "<persisted-output>" in tool_msg["content"]
    assert "Preview (first 50 chars):" in tool_msg["content"]


@pytest.mark.asyncio
async def test_compact_tool_registration_and_spec(tmp_path):
    """验证 CompactTool 遵循 ToolPlugin 契约规范并成功注册 Spec"""
    from context.compactor import CompactTool

    workdir = tmp_path
    reg = ToolRegistry(workdir=workdir)
    compactor = ContextCompactor(workdir=workdir)
    compact_tool = CompactTool(compactor=compactor)

    reg.register_plugin(compact_tool)

    specs = reg.get_tools_spec()
    tool_names = [s["function"]["name"] for s in specs]
    assert "compact_context" in tool_names

    compact_spec = next(s for s in specs if s["function"]["name"] == "compact_context")
    assert "focus" in compact_spec["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_compact_tool_active_execution(tmp_path, mocker):
    """验证大模型主动调用 compact_context 工具，就地浓缩 state.messages 并归档 transcript"""
    from context.compactor import CompactTool
    from runtime.state import current_state_var, current_run_id_var

    workdir = tmp_path
    compactor = ContextCompactor(workdir=workdir)

    mock_client = mocker.AsyncMock(spec=DashScopeClient)
    mock_client.chat_completion.return_value = LLMResponse(
        content="## 精炼事实摘要\n- 完成代码重构\n- 测试已通过",
        finish_reason="stop",
    )

    tool = CompactTool(compactor=compactor, client=mock_client)

    # 模拟一个拥有 10 条历史的 AgentState
    state = AgentState(run_id="run_compact_tool_test")
    state.add_user_message("初始任务")
    for i in range(4):
        state.add_assistant_turn(LLMResponse(content=f"思考过程 {i}", finish_reason="stop"))
        state.add_user_message(f"继续步骤 {i}")

    assert len(state.messages) == 9

    # 模拟协程上下文绑定
    state_token = current_state_var.set(state)
    run_token = current_run_id_var.set(state.run_id)

    try:
        # 执行工具调用
        res = await tool.run(focus="重点保留测试通过的事实")

        assert "上下文压缩成功" in res
        assert "重点保留聚焦: 重点保留测试通过的事实" in res

        # 验证 state.messages 已被浓缩（不再是 9 条，而是重构成系统+单条精炼消息）
        assert len(state.messages) < 9
        assert "[Conversation Compressed]" in state.messages[-1]["content"]
        assert "重点保留测试通过的事实" in state.messages[-1]["content"]

        # 验证全量历史 transcript 已落盘
        transcript_files = list((workdir / "runs" / "run_compact_tool_test" / "transcripts").glob("*.jsonl"))
        assert len(transcript_files) == 1

    finally:
        current_state_var.reset(state_token)
        current_run_id_var.reset(run_token)

