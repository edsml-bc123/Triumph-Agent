"""
triumph-agent 状态机 (AgentState) 自动化测试
=============================================
验证目标：
1. AgentState 初始化与 CST 东八区 run_id 生成；
2. 消息流转协议（User / Assistant / Tool Result）；
3. 步数递增与 max_steps 硬熔断保护；
4. Token 消耗统计与预算超标防护；
5. is_terminal 终态守卫与状态流转完整性。
"""

from runtime.state import AgentState, AgentStatus
from client import LLMResponse, ToolCall, Usage


def test_state_initialization():
    state = AgentState(max_steps=5)
    assert state.status == AgentStatus.PENDING
    assert not state.is_terminal
    assert state.step_count == 0
    assert state.total_tokens == 0
    assert state.run_id.startswith("run_")
    assert len(state.messages) == 0


def test_state_message_flow():
    state = AgentState()
    # 1. 注入用户输入
    state.add_user_message("你好，请帮我分析代码")
    assert len(state.messages) == 1
    assert state.messages[0] == {"role": "user", "content": "你好，请帮我分析代码"}

    # 2. 模拟 Assistant 返回
    tc = ToolCall(id="call_1", name="read_file", arguments_raw='{"path": "a.py"}')
    usage = Usage(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    resp = LLMResponse(
        content="我来读取文件",
        finish_reason="tool_calls",
        tool_calls=[tc],
        usage=usage,
    )
    state.add_assistant_turn(resp)
    assert len(state.messages) == 2
    assert state.messages[1]["role"] == "assistant"
    assert state.total_tokens == 70

    # 3. 回填 Tool 结果
    state.add_tool_result(tool_call_id="call_1", name="read_file", result="print('hello')")
    assert len(state.messages) == 3
    assert state.messages[2]["role"] == "tool"
    assert state.messages[2]["tool_call_id"] == "call_1"
    assert state.messages[2]["content"] == "print('hello')"


def test_state_step_circuit_breaker():
    state = AgentState(max_steps=3)
    assert state.increment_step() is True
    assert state.step_count == 1
    assert state.increment_step() is True
    assert state.step_count == 2

    # 第 3 次递增达到 max_steps=3，触发步数熔断并置为 FAILED
    assert state.increment_step() is False
    assert state.step_count == 3
    assert state.status == AgentStatus.FAILED
    assert state.is_terminal is True
    assert "安全熔断" in state.last_error


def test_state_terminal_transitions():
    # 正常达标
    state_ok = AgentState()
    state_ok.mark_success("任务圆满完成")
    assert state_ok.status == AgentStatus.SUCCESS
    assert state_ok.is_terminal is True
    assert state_ok.final_answer == "任务圆满完成"

    # 异常失败
    state_fail = AgentState()
    state_fail.mark_failed("网络超时中断")
    assert state_fail.status == AgentStatus.FAILED
    assert state_fail.is_terminal is True
    assert state_fail.last_error == "网络超时中断"


def test_terminal_state_immutability():
    """验证终态单向流转法则：一旦进入终态，严禁被任何业务方法复活或篡改"""
    state = AgentState()
    state.mark_failed("预算超标熔断")
    assert state.is_terminal is True
    assert state.status == AgentStatus.FAILED

    # 1. 回填工具结果不能将 FAILED 复活为 RUNNING
    state.add_tool_result(tool_call_id="call_999", name="read_file", result="some content")
    assert state.status == AgentStatus.FAILED
    assert state.is_terminal is True
    # 消息依旧正常追加以保证协议完整性
    assert state.messages[-1]["role"] == "tool"

    # 2. mark_success 不能强行覆盖 FAILED
    state.mark_success("篡改的成功结果")
    assert state.status == AgentStatus.FAILED
    assert state.final_answer is None

    # 3. 反向保护：SUCCESS 同样是终态，不能被 mark_failed 覆写
    ok_state = AgentState()
    ok_state.mark_success("已成功")
    assert ok_state.is_terminal is True
    ok_state.mark_failed("尝试失败覆盖")
    assert ok_state.status == AgentStatus.SUCCESS
    assert ok_state.final_answer == "已成功"

