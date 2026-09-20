"""
triumph-agent 阿里云百炼 (DashScope) 客户端封装
===================================================
核心设计思想：
1. 纯原生 HTTPX 实现，不依赖 OpenAI SDK，完全掌控 HTTP 握手与超时机制；
2. 结构化解析大模型响应，暴露干净的强类型数据模型（LLMResponse, ToolCall, Usage）；
3. 原生支持长连接复用 (Connection Pooling) 与优雅资源回收；
4. 全面防御性异常处理，将网络闪断、HTTP 4xx/5xx 转为具象的业务异常。
"""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import httpx

# 自动加载当前工程根目录下的 .env
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)


import json_repair


# ----------------------------------------------------------------------
# 1. 响应结构封装 (Data Contracts)
# ----------------------------------------------------------------------

@dataclass
class ToolCall:
    """大模型返回的单个工具调用对象"""
    id: str
    name: str
    arguments_raw: str
    type: str = "function"

    def parse_arguments(self) -> Dict[str, Any]:
        """
        防御性解析大模型返回的 arguments 字符串。
        使用 json_repair 自动纠偏与修复大模型畸变 JSON (如缺少括号、单引号、未转义字符等)。
        """
        if not self.arguments_raw or not self.arguments_raw.strip():
            return {}
        try:
            parsed = json_repair.loads(self.arguments_raw)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                nested = json_repair.loads(parsed)
                if isinstance(nested, dict):
                    return nested
            return {"_raw": parsed}
        except Exception as e:
            raise ValueError(
                f"大模型返回的工具 [{self.name}] 参数无法被修复或解析: {self.arguments_raw}"
            ) from e


@dataclass
class Usage:
    """Token 开销度量"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """大模型统一规范化响应"""
    content: str
    finish_reason: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用（触发工具分支的关键判断）"""
        return self.finish_reason == "tool_calls" or len(self.tool_calls) > 0


# ----------------------------------------------------------------------
# 2. 异常定义
# ----------------------------------------------------------------------

class LLMAPIError(Exception):
    """百炼 API 通信与服务端业务异常基类"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


# ----------------------------------------------------------------------
# 3. 百炼客户端核心实现
# ----------------------------------------------------------------------

class DashScopeClient:
    """
    阿里云百炼原生 Async HTTP 客户端
    - 兼容 OpenAI 协议端点
    - 支持会话级连接池复用
    - 支持上下文管理器 (async with)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = 180.0,
        connect_timeout: float = 15.0,
    ):
        self.api_key = (api_key or os.getenv("ALIBABACLOUD_API_KEY") or "").strip()
        if not self.api_key or self.api_key == "your_dashscope_api_key_here":
            raise ValueError(
                "未找到有效的 ALIBABACLOUD_API_KEY。请在 .env 文件中配置真实的密钥。"
            )

        raw_base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.completions_url = f"{raw_base_url}/chat/completions"

        self.default_model = (default_model or os.getenv("DEFAULT_MODEL") or "qwen3.8-flash").strip()
        self.evaluator_model = (os.getenv("EVALUATOR_MODEL") or "qwen3.6-flash").strip()

        # 初始化 HTTPX 异步客户端（连接池复用）
        self.timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def get_http_client(self) -> httpx.AsyncClient:
        """获取或懒加载底层长连接客户端"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def close(self):
        """显式关闭底层 HTTP 连接池"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.7,
        tool_choice: str = "auto",
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        向阿里云百炼发送推理请求核心方法
        
        :param messages: 遵循 OpenAI 格式的消息列表
        :param tools: 工具定义列表 (JSON Schema)
        :param model: 模型名称，默认使用配置的 DEFAULT_MODEL (qwen3.8-flash)
        :param temperature: 采样温度
        :param top_p: 核采样概率
        :param tool_choice: 工具选择策略 ('auto', 'none', 或具体函数名)
        :param enable_thinking: 是否开启思考链 (如适用)
        :return: 规范化解析后的 LLMResponse 对象
        """
        target_model = model or self.default_model

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "enable_thinking": enable_thinking,
            "n": 1,
            **kwargs,
        }

        # 仅在传入 tools 时注入工具字段，保持 Payload 干净
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = self._get_headers()
        client = await self.get_http_client()

        try:
            response = await client.post(self.completions_url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            err_detail = str(exc).strip() or "读取超时 (ReadTimeout) 或对端连接断开"
            raise LLMAPIError(f"百炼 API 网络请求异常 [{type(exc).__name__}]: {err_detail}") from exc

        if response.status_code != 200:
            raise LLMAPIError(
                message=f"百炼 API 请求失败，状态码: {response.status_code}",
                status_code=response.status_code,
                response_text=response.text,
            )

        try:
            res_json = response.json()
        except Exception as e:
            raise LLMAPIError(f"百炼 API 返回体无法解析为 JSON: {response.text}") from e

        return self._parse_response(res_json)

    def _parse_response(self, res_json: Dict[str, Any]) -> LLMResponse:
        """将原始 API JSON 报文转化为严谨的强类型 LLMResponse 对象"""
        choices = res_json.get("choices", [])
        if not choices:
            raise LLMAPIError(f"百炼 API 返回空 choices: {res_json}")

        first_choice = choices[0]
        finish_reason = first_choice.get("finish_reason") or "stop"
        message_data = first_choice.get("message", {})
        content = message_data.get("content") or ""

        # 解析工具调用
        parsed_tool_calls: List[ToolCall] = []
        raw_tool_calls = message_data.get("tool_calls") or []
        for item in raw_tool_calls:
            func_data = item.get("function", {})
            parsed_tool_calls.append(
                ToolCall(
                    id=item.get("id", ""),
                    name=func_data.get("name", ""),
                    arguments_raw=func_data.get("arguments", "{}"),
                    type=item.get("type", "function"),
                )
            )

        # 解析 Token 账单
        raw_usage = res_json.get("usage", {})
        usage = Usage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            finish_reason=finish_reason,
            tool_calls=parsed_tool_calls,
            usage=usage,
            model=res_json.get("model", self.default_model),
            raw_response=res_json,
        )


# ----------------------------------------------------------------------
# 4. 模块独立快速冒烟自测
# ----------------------------------------------------------------------

async def _smoke_test():
    print("启动 client.py 冒烟自测...")
    
    async with DashScopeClient() as client:
        # 测试 1: 普通直接对话
        print("\n[测试 1] 发送纯文本问答...")
        res = await client.chat_completion(
            messages=[{"role": "user", "content": "请用一句话介绍你自己。"}]
        )
        print(f"Model:         {res.model}")
        print(f"Finish Reason: {res.finish_reason}")
        print(f"Answer:        {res.content}")
        print(f"Usage:         {res.usage.total_tokens} tokens")

        # 测试 2: 带工具调用的对话
        print("\n[测试 2] 发送带工具的问答 (探查 Tool Calling 解析)...")
        demo_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_sum",
                    "description": "计算两个整数的和",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        res_tool = await client.chat_completion(
            messages=[{"role": "user", "content": "帮我算一下 1234 + 5678 等于多少？"}],
            tools=demo_tools,
        )
        print(f"Has Tool Calls: {res_tool.has_tool_calls}")
        print(f"Finish Reason:  {res_tool.finish_reason}")
        for tc in res_tool.tool_calls:
            print(f" -> 触发工具: {tc.name} | 参数: {tc.parse_arguments()} | ID: {tc.id}")

    print("\n✅ client.py 冒烟自测全部通过！封装健壮可用。")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
