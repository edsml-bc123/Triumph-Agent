"""
triumph-agent 阿里云百炼客户端 (DashScopeClient) 自动化测试
=========================================================
验证目标：
1. ToolCall 数据契约与 json_repair 畸变参数自动修复；
2. LLMResponse 规范化数据结构与 has_tool_calls 特性断言；
3. DashScopeClient 初始化防御性校验 (缺失 API Key / 占位符拦截)；
4. chat_completion HTTP 请求装配、Headers 与 Payload 协议校验；
5. 大模型返回体解析 (choices, tool_calls, usage)；
6. 极端防御性容错 (HTTP 4xx/5xx, 空 choices, 畸变非 JSON, 网络断联)；
7. 真实环境可选冒烟集成测试 (当检测到有效真实 Key 时执行，否则自动跳过)。
"""

import os
import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from client import DashScopeClient, ToolCall, LLMResponse, Usage, LLMAPIError


# ----------------------------------------------------------------------
# 1. ToolCall 与 json_repair 畸变修复测试
# ----------------------------------------------------------------------

def test_tool_call_parse_normal_json():
    """验证标准 JSON 参数解析"""
    tc = ToolCall(id="call_001", name="read_file", arguments_raw='{"path": "test.py", "limit": 10}')
    args = tc.parse_arguments()
    assert args == {"path": "test.py", "limit": 10}


def test_tool_call_parse_empty_or_whitespace():
    """验证空字符串或纯空白安全返回空字典"""
    tc_empty = ToolCall(id="call_002", name="list_dir", arguments_raw="")
    assert tc_empty.parse_arguments() == {}

    tc_ws = ToolCall(id="call_003", name="list_dir", arguments_raw="   \n\t  ")
    assert tc_ws.parse_arguments() == {}


def test_tool_call_json_repair_malformed():
    """验证 json_repair 对大模型常见畸变 JSON 的自动纠偏与自愈"""
    # 场景 1: 单引号包裹与尾随逗号 (Python 字典风格)
    raw_single_quote = "{'path': 'sub/file.txt', 'count': 5,}"
    tc1 = ToolCall(id="call_004", name="bash", arguments_raw=raw_single_quote)
    assert tc1.parse_arguments() == {"path": "sub/file.txt", "count": 5}

    # 场景 2: 缺失闭合右括号
    raw_unclosed = '{"command": "pytest tests/'
    tc2 = ToolCall(id="call_005", name="bash", arguments_raw=raw_unclosed)
    parsed = tc2.parse_arguments()
    assert "command" in parsed

    # 场景 3: 字符串被双重 JSON 序列化 (Double-encoded string)
    raw_double = json.dumps('{"nested_key": "nested_val"}')
    tc3 = ToolCall(id="call_006", name="test", arguments_raw=raw_double)
    assert tc3.parse_arguments() == {"nested_key": "nested_val"}


def test_tool_call_parse_invalid_raises_value_error():
    """验证极端彻底无法解析的输入抛出 ValueError"""
    with patch("client.json_repair.loads", side_effect=Exception("syntax error")):
        tc = ToolCall(id="call_err", name="bad_tool", arguments_raw="###INVALID###")
        with pytest.raises(ValueError) as exc_info:
            tc.parse_arguments()
        assert "无法被修复或解析" in str(exc_info.value)


# ----------------------------------------------------------------------
# 2. LLMResponse 状态与 has_tool_calls 判定
# ----------------------------------------------------------------------

def test_llm_response_has_tool_calls_flag():
    """验证 has_tool_calls 的各种分支触发条件"""
    # 纯文本回答
    resp_text = LLMResponse(content="hello", finish_reason="stop")
    assert not resp_text.has_tool_calls

    # finish_reason 为 tool_calls
    resp_tool_reason = LLMResponse(content="", finish_reason="tool_calls")
    assert resp_tool_reason.has_tool_calls

    # tool_calls 列表非空
    resp_tool_list = LLMResponse(
        content="",
        finish_reason="stop",
        tool_calls=[ToolCall(id="1", name="calc", arguments_raw="{}")],
    )
    assert resp_tool_list.has_tool_calls


# ----------------------------------------------------------------------
# 3. 客户端初始化与配置校验
# ----------------------------------------------------------------------

def test_client_init_missing_key():
    """验证缺失 API Key 时阻断初始化并提示配置"""
    with patch.dict(os.environ, {"ALIBABACLOUD_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError) as exc_info:
            DashScopeClient(api_key="")
        assert "未找到有效的 ALIBABACLOUD_API_KEY" in str(exc_info.value)

    # 验证占位符也被硬拦截
    with pytest.raises(ValueError) as exc_info:
        DashScopeClient(api_key="your_dashscope_api_key_here")
    assert "未找到有效的 ALIBABACLOUD_API_KEY" in str(exc_info.value)


def test_client_headers_and_endpoints():
    """验证请求端点与 Header 规范"""
    client = DashScopeClient(
        api_key="sk-test-valid-key",
        base_url="https://test.dashscope.api/v1/",
        default_model="qwen-custom",
    )
    assert client.completions_url == "https://test.dashscope.api/v1/chat/completions"
    assert client.default_model == "qwen-custom"
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer sk-test-valid-key"
    assert headers["Content-Type"] == "application/json"


# ----------------------------------------------------------------------
# 4. chat_completion 请求与响应解析单测 (Mock HTTP)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completion_text_response_success():
    """验证纯文本对话请求 Payload 组装与 LLMResponse 解析"""
    mock_response_json = {
        "id": "chatcmpl-123",
        "model": "qwen3.8-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "这是一个纯文本解答。",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }

    client = DashScopeClient(api_key="sk-test-fake")

    mock_http_response = httpx.Response(
        status_code=200,
        json=mock_response_json,
        request=httpx.Request("POST", client.completions_url),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http_response

        messages = [{"role": "user", "content": "你好"}]
        res = await client.chat_completion(messages=messages)

        assert res.content == "这是一个纯文本解答。"
        assert res.finish_reason == "stop"
        assert not res.has_tool_calls
        assert len(res.tool_calls) == 0
        assert res.usage.prompt_tokens == 12
        assert res.usage.completion_tokens == 8
        assert res.usage.total_tokens == 20
        assert res.model == "qwen3.8-flash"

        # 检查发往后端的 payload 是否干净 (未传 tools 时不包含 tools 字段)
        call_kwargs = mock_post.call_args.kwargs
        assert "tools" not in call_kwargs["json"]
        assert call_kwargs["json"]["model"] == "qwen3.8-flash"
        assert call_kwargs["json"]["messages"] == messages

    await client.close()


@pytest.mark.asyncio
async def test_chat_completion_tool_calls_response_success():
    """验证包含工具调用的响应解析与参数装配"""
    mock_response_json = {
        "id": "chatcmpl-456",
        "model": "qwen3.8-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "config.json"}',
                            },
                        },
                        {
                            "id": "call_abc_2",
                            "type": "function",
                            "function": {
                                "name": "glob",
                                "arguments": '{"pattern": "*.py"}',
                            },
                        },
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80,
        },
    }

    client = DashScopeClient(api_key="sk-test-fake")

    mock_http_response = httpx.Response(
        status_code=200,
        json=mock_response_json,
        request=httpx.Request("POST", client.completions_url),
    )

    tools_spec = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        {"type": "function", "function": {"name": "glob", "parameters": {}}},
    ]

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http_response

        messages = [{"role": "user", "content": "帮我看看文件"}]
        res = await client.chat_completion(messages=messages, tools=tools_spec)

        assert res.has_tool_calls is True
        assert res.finish_reason == "tool_calls"
        assert len(res.tool_calls) == 2
        assert res.tool_calls[0].name == "read_file"
        assert res.tool_calls[0].parse_arguments() == {"path": "config.json"}
        assert res.tool_calls[1].name == "glob"
        assert res.tool_calls[1].parse_arguments() == {"pattern": "*.py"}

        # 检查 payload 注入了 tools 与 tool_choice
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["tools"] == tools_spec
        assert call_kwargs["json"]["tool_choice"] == "auto"

    await client.close()


# ----------------------------------------------------------------------
# 5. 防御性容错与异常转换单测
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completion_http_error_status():
    """验证 HTTP 4xx/5xx 状态码抛出明确的 LLMAPIError"""
    client = DashScopeClient(api_key="sk-test-fake")

    mock_http_response = httpx.Response(
        status_code=401,
        text='{"error": {"message": "Invalid API key provided."}}',
        request=httpx.Request("POST", client.completions_url),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http_response

        with pytest.raises(LLMAPIError) as exc_info:
            await client.chat_completion(messages=[{"role": "user", "content": "hi"}])

        assert exc_info.value.status_code == 401
        assert "401" in str(exc_info.value)
        assert "Invalid API key" in (exc_info.value.response_text or "")

    await client.close()


@pytest.mark.asyncio
async def test_chat_completion_network_request_error():
    """验证网络断连或读取超时抛出 LLMAPIError"""
    client = DashScopeClient(api_key="sk-test-fake")

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectTimeout("Connection timed out.")

        with pytest.raises(LLMAPIError) as exc_info:
            await client.chat_completion(messages=[{"role": "user", "content": "hi"}])

        assert "网络请求异常 [ConnectTimeout]" in str(exc_info.value)

    await client.close()


@pytest.mark.asyncio
async def test_chat_completion_empty_choices():
    """验证大模型返回空 choices 时安全抛出异常而不是 IndexError"""
    client = DashScopeClient(api_key="sk-test-fake")

    mock_http_response = httpx.Response(
        status_code=200,
        json={"choices": [], "usage": {}},
        request=httpx.Request("POST", client.completions_url),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http_response

        with pytest.raises(LLMAPIError) as exc_info:
            await client.chat_completion(messages=[{"role": "user", "content": "hi"}])

        assert "返回空 choices" in str(exc_info.value)

    await client.close()


@pytest.mark.asyncio
async def test_client_context_manager_lifecycle():
    """验证 async with 上下文管理器的连接池自动关闭"""
    client_instance = None
    async with DashScopeClient(api_key="sk-test-fake") as client:
        client_instance = client
        http_client = await client.get_http_client()
        assert not http_client.is_closed

    assert client_instance._http_client.is_closed


# ----------------------------------------------------------------------
# 6. 真实线上连通性测试 (仅在本地存在真实 API Key 时执行)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_dashscope_roundtrip():
    """
    真实连通性测试 (集成测试)：
    若检测到真实密钥，执行轻量探活，确保百炼后端协议握手与模型推理全通。
    """
    api_key = os.getenv("ALIBABACLOUD_API_KEY", "").strip()
    if not api_key or api_key == "your_dashscope_api_key_here":
        pytest.skip("未检测到真实的 ALIBABACLOUD_API_KEY，跳过线上网络探活。")

    async with DashScopeClient() as client:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "请回复单字：OK"}],
            temperature=0.0,
        )
        assert response.content != ""
        assert response.finish_reason == "stop"
        assert response.usage.total_tokens > 0
