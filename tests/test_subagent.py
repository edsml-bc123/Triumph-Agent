"""
triumph-agent 子智能体调度与物理上下文隔离 (Subagent Engine) 自动化测试
========================================================================
验证目标：
1. 统一工具插件化装配：SubAgentTool 符合 register_to 协议，支持 registry.register_plugin(...)；
2. 工具剥离与防递归：父注册表包含 subagent，派生的子注册表严格剥离 subagent，杜绝套娃；
3. 纯净上下文隔离：子任务从全新的 messages=[system, user] 启动，父级海量对话不被复制；
4. 执行闭环与浓缩回填：子智能体自主调用工具并完成总结，最终结论作为工具结果无缝回填；
5. 容错防御与进程保活：子任务发生大模型通信异常或超时时，安全转化为自愈文本返回，不击穿父循环；
6. 层级溯源与轨迹黑匣子：SubAgentState 显式绑定 parent_run_id，独立落盘 JSONL 轨迹。
"""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from client import LLMResponse, ToolCall
from orchestration import SubAgentTool
from runtime.state import current_run_id_var
from tools import BuiltinTool, ToolRegistry


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.default_model = "qwen3.8-flash"
    client.evaluator_model = "qwen3.6-flash"
    client.chat_completion = AsyncMock()
    return client


@pytest.fixture
def base_registry(tmp_path):
    """标准测试夹具：预装配 BuiltinTool 的干净工具注册表"""
    reg = ToolRegistry(workdir=tmp_path)
    reg.register_plugin(BuiltinTool(workdir=tmp_path))
    return reg


# ----------------------------------------------------------------------
# 1. 统一插件注册与工具剥离防递归测试
# ----------------------------------------------------------------------

def test_subagent_tool_registration_and_stripping(mock_client, base_registry):
    # 采用优雅的统一插件装配协议 (无任何冗余传参)
    base_registry.register_plugin(SubAgentTool(client=mock_client))

    # 1. 父注册表包含 subagent 工具
    parent_specs = base_registry.get_tools_spec()
    tool_names = [s["function"]["name"] for s in parent_specs]
    assert "subagent" in tool_names
    assert "bash" in tool_names
    assert "read_file" in tool_names

    # 2. 通过通用 fork 派生的子注册表严格剥离了 subagent（仅排除自身 1 个工具），杜绝嵌套递归
    sub_registry = base_registry.fork(exclude={"subagent"})
    sub_specs = sub_registry.get_tools_spec()
    sub_tool_names = [s["function"]["name"] for s in sub_specs]
    assert "subagent" not in sub_tool_names
    assert "bash" in sub_tool_names
    assert "read_file" in sub_tool_names


# ----------------------------------------------------------------------
# 2. 纯净上下文隔离测试 (Fresh Context)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_context_isolation(mock_client, base_registry):
    tool = SubAgentTool(client=mock_client)
    base_registry.register_plugin(tool)

    captured_messages = []

    async def mock_completion(messages, **kwargs):
        captured_messages.extend(messages)
        return LLMResponse(
            content="子任务分析完成：无任何调用链异常。",
            finish_reason="stop",
            tool_calls=[],
        )

    mock_client.chat_completion.side_effect = mock_completion

    result = await tool.run(
        prompt="请分析项目核心依赖",
        parent_run_id="parent_run_12345",
    )

    assert result == "子任务分析完成：无任何调用链异常。"
    # 验证子智能体的上下文是纯净的：仅包含一条系统消息和一条用户 prompt
    assert len(captured_messages) == 2
    assert captured_messages[0]["role"] == "system"
    assert "你是由主智能体委派的专属子智能体" in captured_messages[0]["content"]
    assert captured_messages[1]["role"] == "user"
    assert captured_messages[1]["content"] == "请分析项目核心依赖"


# ----------------------------------------------------------------------
# 3. 工具执行与结果浓缩回填测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_execution_success_and_summary_returned(mock_client, tmp_path, base_registry):
    test_file = tmp_path / "target.py"
    test_file.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    tool = SubAgentTool(client=mock_client)
    base_registry.register_plugin(tool)

    turn1_response = LLMResponse(
        content="我来阅读 target.py 文件",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="call_read_1",
                name="read_file",
                arguments_raw=json.dumps({"path": "target.py"}),
            )
        ],
    )
    turn2_response = LLMResponse(
        content="target.py 中定义了 add 函数，接收 a 和 b 并返回两数之和。",
        finish_reason="stop",
        tool_calls=[],
    )

    mock_client.chat_completion.side_effect = [turn1_response, turn2_response]

    summary = await tool.run(
        prompt="查看 target.py 实现了什么功能",
        parent_run_id="run_root_001",
    )

    assert "target.py 中定义了 add 函数" in summary
    assert mock_client.chat_completion.call_count == 2


# ----------------------------------------------------------------------
# 4. 容错防御与进程保活测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_resilience_on_failure(mock_client, base_registry):
    tool = SubAgentTool(client=mock_client)
    base_registry.register_plugin(tool)

    mock_client.chat_completion.side_effect = RuntimeError("网络对端拒绝连接")

    result = await tool.run(
        prompt="抓取远程日志",
        parent_run_id="run_root_002",
    )

    assert "Subagent execution ended without success" in result
    assert "网络对端拒绝连接" in result


# ----------------------------------------------------------------------
# 5. 层级溯源与轨迹黑匣子落盘测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_parent_run_id_and_trajectory(mock_client, tmp_path, base_registry):
    runs_dir = tmp_path / "runs"
    # 支持可选显式覆盖 runs_dir 用于受控测试
    tool = SubAgentTool(client=mock_client, runs_dir=runs_dir)
    base_registry.register_plugin(tool)

    mock_client.chat_completion.return_value = LLMResponse(
        content="探查完毕",
        finish_reason="stop",
        tool_calls=[],
    )

    parent_id = "parent_task_888"
    await tool.run(prompt="快速检测", parent_run_id=parent_id)

    # 验证树状分层结构: runs/{parent_id}/subagents/{sub_id}/trajectory.jsonl
    trajectory_files = list(runs_dir.rglob("trajectory.jsonl"))
    assert len(trajectory_files) == 1

    sub_trajectory_file = trajectory_files[0]
    # 父目录关系: trajectory.jsonl -> sub_xxxx -> subagents -> parent_task_888
    assert sub_trajectory_file.parent.parent.name == "subagents"
    assert sub_trajectory_file.parent.parent.parent.name == parent_id

    lines = sub_trajectory_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    first_event = json.loads(lines[0])
    assert "run_id" in first_event


# ----------------------------------------------------------------------
# 6. 异步 ContextVar 自动穿透与子任务 Compactor 树状落盘集成测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_contextvar_inheritance_and_compactor(mock_client, tmp_path, base_registry):
    """
    验证：
    1. 子任务通过 current_run_id_var 自动无感获取主任务 run_id；
    2. 子任务生命周期树状收拢至 runs/{main_run_id}/subagents/{sub_id}/；
    3. 子任务触发超大工具输出时，Compactor L1 自动落盘至树状子目录 tool_outputs/。
    """
    runs_dir = tmp_path / "runs"
    tool = SubAgentTool(client=mock_client, runs_dir=runs_dir)
    base_registry.register_plugin(tool)

    # 创建一个 30,000 字符的真实大文件，由内置 read_file 工具读取，专门触发 Compactor L1 (阈值 20,000 字符)
    big_file = tmp_path / "big_data.txt"
    big_file.write_text("DATA_LINE_" * 3000, encoding="utf-8")

    # 模拟子智能体调用 read_file 工具读取该大文件，随后给出最终答案
    mock_client.chat_completion.side_effect = [
        LLMResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="call_big_data_123",
                    name="read_file",
                    arguments_raw=json.dumps({"path": "big_data.txt"}),
                )
            ],
        ),
        LLMResponse(
            content="大文件数据已成功提取完毕",
            finish_reason="stop",
            tool_calls=[],
        ),
    ]

    main_run_id = "main_active_run_999"
    # 模拟主任务处于执行循环中，绑定了 current_run_id_var
    token = current_run_id_var.set(main_run_id)
    try:
        result = await base_registry.execute("subagent", {"prompt": "提取大数据"})
        assert "大文件数据已成功提取完毕" in result
    finally:
        current_run_id_var.reset(token)

    # 1. 验证轨迹文件落入 runs/{main_run_id}/subagents/{sub_id}/trajectory.jsonl
    sub_trajectories = list(runs_dir.glob(f"{main_run_id}/subagents/*/trajectory.jsonl"))
    assert len(sub_trajectories) == 1
    sub_dir = sub_trajectories[0].parent

    # 2. 验证 Compactor L1 自动截断落盘至 runs/{main_run_id}/subagents/{sub_id}/tool_outputs/call_big_data_123.txt
    tool_output_file = sub_dir / "tool_outputs" / "call_big_data_123.txt"
    assert tool_output_file.exists()
    saved_content = tool_output_file.read_text(encoding="utf-8")
    assert len(saved_content) == 30000
    assert "DATA_LINE_" in saved_content
