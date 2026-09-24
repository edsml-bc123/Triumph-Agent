"""
triumph-agent MCP 协议定义与数据契约 (Protocol Specifications)
=============================================================
遵循 JSON-RPC 2.0 规范与 Model Context Protocol (2024-11-05 版本) 核心原语契约。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

# 允许的工具命名规范字符集 (与 OpenAI / 百炼规范对齐)
_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_mcp_name(name: str) -> str:
    """
    规范化服务名或工具名，替换非 [a-zA-Z0-9_-] 字符为下划线。
    若清洗后全为空或仅剩下划线，则视为非法名称抛出 ValueError。
    """
    normalized = _DISALLOWED_CHARS.sub("_", name.strip())
    if not normalized or not normalized.strip("_"):
        raise ValueError(f"MCP 名称规范化后不可为空或仅包含下划线: '{name}'")
    return normalized


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 错误结构"""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 请求对象 (若 id 为 None 则为单向通知 Notification)"""
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        payload: Dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params is not None:
            payload["params"] = self.params
        if self.id is not None:
            payload["id"] = self.id
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 响应对象"""
    id: Optional[Union[int, str]]
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JSONRPCResponse:
        err = None
        if "error" in data and isinstance(data["error"], dict):
            err_dict = data["error"]
            err = JSONRPCError(
                code=err_dict.get("code", -32603),
                message=err_dict.get("message", "Internal JSON-RPC error"),
                data=err_dict.get("data"),
            )
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=err,
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


@dataclass
class MCPToolDefinition:
    """MCP 远程工具标准声明"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPToolDefinition:
        name = str(data.get("name", "")).strip()
        desc = str(data.get("description", "")).strip()
        schema = data.get("inputSchema") or data.get("input_schema") or {}
        if not isinstance(schema, dict) or schema.get("type") != "object":
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            schema = {"type": "object", "properties": props}
        return cls(
            name=name,
            description=desc,
            input_schema=schema,
            metadata=data.get("annotations", {}),
        )


@dataclass
class MCPServerInfo:
    """MCP 服务端信息元数据"""
    name: str
    version: str = "1.0.0"
    protocol_version: str = "2024-11-05"
    capabilities: Dict[str, Any] = field(default_factory=dict)
