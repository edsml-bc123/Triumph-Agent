"""
triumph-agent MCP 客户端核心引擎 (MCP Client & Transports)
===========================================================
支持标准 STDIO 子进程管道传输 (StdioTransport) 与进程内传输 (InProcessTransport)。
实现生命周期握手协商 (initialize)、动态工具发现 (tools/list) 与安全工具执行 (tools/call)。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from loguru import logger

from mcp.protocol import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPServerInfo,
    MCPToolDefinition,
    normalize_mcp_name,
)


class MCPError(Exception):
    """MCP 异常基类"""
    pass


class MCPConnectionError(MCPError):
    """MCP 传输连接异常"""
    pass


class MCPProtocolError(MCPError):
    """MCP 协议响应异常"""
    pass


# ----------------------------------------------------------------------
# 1. 传输层抽象 (Transports)
# ----------------------------------------------------------------------

class MCPTransport(ABC):
    """MCP 通信传输抽象层"""

    @abstractmethod
    async def start(self) -> None:
        """启动或建立物理连接"""
        pass

    @abstractmethod
    async def send_request(self, request: JSONRPCRequest, timeout: float = 60.0) -> JSONRPCResponse:
        """发送 JSON-RPC 请求并等待对应响应"""
        pass

    @abstractmethod
    async def send_notification(self, request: JSONRPCRequest) -> None:
        """发送单向通知 (无 id，不期望响应)"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭传输与清理底层句柄"""
        pass


class StdioTransport(MCPTransport):
    """
    基于异步子进程管道的标准 STDIO 传输层
    通过子进程 stdin 发送报文，通过 stdout 读取响应。
    """

    def __init__(
        self,
        command: Union[str, Sequence[str]],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        if isinstance(command, str):
            self.command = [command]
        else:
            self.command = list(command)

        self.cwd = str((cwd or Path.cwd()).resolve())
        self.env = env
        self.process: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return

        child_env = os.environ.copy()
        if self.env:
            child_env.update(self.env)

        # 确保当前 Python 所在目录置顶于子进程 PATH
        current_bin = str(Path(sys.executable).parent)
        child_env["PATH"] = f"{current_bin}:{child_env.get('PATH', '')}"

        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=child_env,
            )
            logger.debug(f"[MCP Stdio] 成功拉起子进程 PID: {self.process.pid} | 命令: {' '.join(self.command)}")
        except Exception as e:
            raise MCPConnectionError(f"启动 MCP 子进程失败 [{' '.join(self.command)}]: {e}") from e

    async def send_request(self, request: JSONRPCRequest, timeout: float = 60.0) -> JSONRPCResponse:
        if self.process is None or self.process.returncode is not None:
            await self.start()

        assert self.process is not None
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        line = request.to_json() + "\n"
        async with self._lock:
            try:
                # 写入请求报文
                self.process.stdin.write(line.encode("utf-8"))
                await self.process.stdin.drain()

                # 读取对应响应
                resp_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
                if not resp_line:
                    returncode = self.process.returncode
                    raise MCPConnectionError(f"MCP 子进程管道已关闭 (ReturnCode: {returncode})")

                raw_text = resp_line.decode("utf-8").strip()
                data = json.loads(raw_text)
                return JSONRPCResponse.from_dict(data)
            except asyncio.TimeoutError:
                raise MCPConnectionError(f"MCP 请求执行超时 ({timeout}s) [method={request.method}]")
            except Exception as e:
                if isinstance(e, MCPError):
                    raise
                raise MCPConnectionError(f"MCP 管道通信异常: {e}") from e

    async def send_notification(self, request: JSONRPCRequest) -> None:
        if self.process is None or self.process.returncode is not None:
            await self.start()

        assert self.process is not None
        assert self.process.stdin is not None

        line = request.to_json() + "\n"
        async with self._lock:
            try:
                self.process.stdin.write(line.encode("utf-8"))
                await self.process.stdin.drain()
            except Exception as e:
                logger.warning(f"[MCP Stdio] 发送通知异常 [method={request.method}]: {e}")

    async def close(self) -> None:
        if self.process is None:
            return

        try:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        except Exception as e:
            logger.debug(f"[MCP Stdio] 关闭进程句柄时忽略正常退出异常: {e}")
        finally:
            self.process = None


class InProcessTransport(MCPTransport):
    """
    进程内轻量传输层 (In-Process Transport)
    直接通过内部调度函数响应，专用于单元测试、高并发环境与进程内自建工具。
    """

    def __init__(self, router: Callable[[str, Optional[Dict[str, Any]]], Any]):
        self.router = router
        self._is_open = True

    async def start(self) -> None:
        self._is_open = True

    async def send_request(self, request: JSONRPCRequest, timeout: float = 60.0) -> JSONRPCResponse:
        if not self._is_open:
            raise MCPConnectionError("InProcessTransport 连接已关闭")

        try:
            handler_result = self.router(request.method, request.params)
            if asyncio.iscoroutine(handler_result):
                handler_result = await asyncio.wait_for(handler_result, timeout=timeout)
            return JSONRPCResponse(id=request.id, result=handler_result)
        except Exception as e:
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(code=-32603, message=str(e)),
            )

    async def send_notification(self, request: JSONRPCRequest) -> None:
        if not self._is_open:
            return
        try:
            res = self.router(request.method, request.params)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            logger.warning(f"[InProcessTransport] 通知处理异常: {e}")

    async def close(self) -> None:
        self._is_open = False


# ----------------------------------------------------------------------
# 2. MCPClient 核心客户端
# ----------------------------------------------------------------------

class MCPClient:
    """
    标准 Model Context Protocol (MCP) 客户端
    封装生命周期握手、工具发现与远程工具调用。
    """

    def __init__(self, name: str, transport: MCPTransport):
        self.name = normalize_mcp_name(name)
        self.transport = transport
        self.server_info: Optional[MCPServerInfo] = None
        self.tools: Dict[str, MCPToolDefinition] = {}
        self._req_counter = 0

    def _next_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    async def connect(self) -> None:
        """
        连接 MCP 服务端并完成标准双阶段握手协商 (Handshake):
        1. 发送 initialize 请求；
        2. 确认服务返回 capabilities；
        3. 发送 notifications/initialized 通知。
        """
        await self.transport.start()

        # Step 1: initialize
        req = JSONRPCRequest(
            id=self._next_id(),
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "triumph-agent",
                    "version": "0.4.0",
                },
            },
        )
        resp = await self.transport.send_request(req)
        if resp.error:
            raise MCPProtocolError(f"MCP 初始化协商失败 [{self.name}]: {resp.error.message}")

        result_data = resp.result or {}
        server_info_dict = result_data.get("serverInfo", {})
        self.server_info = MCPServerInfo(
            name=server_info_dict.get("name", self.name),
            version=server_info_dict.get("version", "1.0.0"),
            protocol_version=result_data.get("protocolVersion", "2024-11-05"),
            capabilities=result_data.get("capabilities", {}),
        )

        # Step 2: 发送 initialized 通知
        await self.transport.send_notification(
            JSONRPCRequest(method="notifications/initialized")
        )
        logger.debug(f"[MCPClient] 服务端 '{self.name}' 握手成功 (Server: {self.server_info.name} v{self.server_info.version})")

    async def list_tools(self) -> List[MCPToolDefinition]:
        """
        向 MCP 服务端查询所有可用的远程工具列表 (tools/list)
        """
        req = JSONRPCRequest(
            id=self._next_id(),
            method="tools/list",
            params={},
        )
        resp = await self.transport.send_request(req)
        if resp.error:
            raise MCPProtocolError(f"拉取工具列表失败 [{self.name}]: {resp.error.message}")

        raw_tools = (resp.result or {}).get("tools", [])
        self.tools.clear()

        for item in raw_tools:
            tool_def = MCPToolDefinition.from_dict(item)
            if tool_def.name:
                self.tools[tool_def.name] = tool_def

        return list(self.tools.values())

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        调用远程工具并提取文本执行结果 (tools/call)
        安全返回结果或异常文本，绝不崩溃主进程。
        """
        req = JSONRPCRequest(
            id=self._next_id(),
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments,
            },
        )

        try:
            resp = await self.transport.send_request(req)
        except Exception as e:
            return f"Error: MCP connection failed during tools/call [{self.name}/{name}]: {e}"

        if resp.error:
            return f"Error: MCP tool call rejected [{self.name}/{name}]: {resp.error.message}"

        result_dict = resp.result or {}
        is_error = result_dict.get("isError", False)
        content_items = result_dict.get("content", [])

        # 解析 content 数组：优先提取 text 字段
        extracted_texts: List[str] = []
        for item in content_items:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    extracted_texts.append(str(item["text"]))
                elif "text" in item:
                    extracted_texts.append(str(item["text"]))
                else:
                    extracted_texts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                extracted_texts.append(item)

        output_text = "\n".join(extracted_texts).strip()
        if not output_text:
            output_text = "(empty tool result)"

        if is_error:
            return f"Error: {output_text}"
        return output_text

    async def close(self) -> None:
        """安全释放与关闭客户端连接"""
        await self.transport.close()
