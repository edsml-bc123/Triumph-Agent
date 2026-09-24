"""
triumph-agent MCP 模块入口
==========================
导出 MCP 核心客户端、适配插件与传输协议实现。
"""

from __future__ import annotations

from mcp.client import (
    InProcessTransport,
    MCPClient,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTransport,
    StdioTransport,
)
from mcp.protocol import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPServerInfo,
    MCPToolDefinition,
    normalize_mcp_name,
)
from mcp.tool import MCPTool

__all__ = [
    "MCPClient",
    "MCPTool",
    "MCPTransport",
    "StdioTransport",
    "InProcessTransport",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    "MCPServerInfo",
    "MCPToolDefinition",
    "MCPError",
    "MCPConnectionError",
    "MCPProtocolError",
    "normalize_mcp_name",
]
