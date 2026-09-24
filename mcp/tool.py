"""
triumph-agent MCP 工具插件系统 (MCP Tool Adapter)
=================================================
统一遵循 ToolPlugin 协议契约，将任意第三方 MCP 远程工具动态转译并注册进 ToolRegistry。
提供名称空间物理隔离 (mcp__{server}__{tool})、防重名冲突防御与动态连接发现机制。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from loguru import logger

from mcp.client import MCPClient, StdioTransport
from mcp.protocol import normalize_mcp_name
from tools.registry import ToolRegistry


class MCPTool:
    """
    MCP 远程工具聚合与挂载插件 (符合 ToolPlugin 契约协议)
    全面采用按需动态连接模型 (connect_mcp)，杜绝静态预载子进程开销。
    """

    def __init__(
        self,
        factories: Optional[Dict[str, Callable[[], MCPClient]]] = None,
        descriptions: Optional[Dict[str, str]] = None,
    ):
        """
        :param factories: 懒加载工厂映射（供 connect_mcp 动态发现与实时拉起）
        :param descriptions: 各 MCP 服务的文字能力摘要（用于向模型展示服务目录）
        """
        self.factories: Dict[str, Callable[[], MCPClient]] = factories or {}
        self.descriptions: Dict[str, str] = descriptions or {}
        self.clients: Dict[str, MCPClient] = {}  # 运行时已建立连接的客户端缓存
        self._mounted_tools: Dict[str, str] = {}  # prefixed_name -> server_name
        self._registry: Optional[ToolRegistry] = None

    def register_to(self, registry: ToolRegistry) -> None:
        """
        向 ToolRegistry 挂载动态连接发现工具 connect_mcp 并注入当前就绪的服务目录。
        """
        self._registry = registry

        # 动态构建服务目录与参数候选枚举
        candidate_servers = sorted(self.factories.keys())
        if candidate_servers:
            server_list_desc = "\n".join(
                f"- {name}: {self.descriptions.get(name, '外部MCP服务')}"
                for name in candidate_servers
            )
            desc = (
                "动态连接到指定的外部 MCP 服务端，并将其提供的所有工具挂载进当前工具箱。\n"
                f"当前系统已就绪的候选 MCP 服务列表：\n{server_list_desc}"
            )
            server_prop: Dict[str, Any] = {
                "type": "string",
                "description": f"要连接的 MCP 服务名称，可选范围: {', '.join(candidate_servers)}",
                "enum": candidate_servers,
            }
        else:
            desc = "动态连接到指定的外部 MCP 服务端，并将其提供的所有工具挂载进当前工具箱。"
            server_prop = {
                "type": "string",
                "description": "要连接的 MCP 服务名称",
            }

        # 注册核心动态连接发现工具 connect_mcp
        registry.register(
            name="connect_mcp",
            description=desc,
            parameters={
                "type": "object",
                "properties": {
                    "server_name": server_prop,
                },
                "required": ["server_name"],
            },
            handler=self._run_connect_mcp,
        )

    def _register_client_tools(self, registry: ToolRegistry, client: MCPClient) -> List[str]:
        """将某个已拉取工具声明的 MCPClient 工具注册进 ToolRegistry"""
        safe_server = normalize_mcp_name(client.name)
        newly_mounted: List[str] = []

        for raw_name, tool_def in client.tools.items():
            safe_tool = normalize_mcp_name(raw_name)
            prefixed = f"mcp__{safe_server}__{safe_tool}"

            if len(prefixed) > 64:
                raise ValueError(f"MCP 工具完整名称超限 (64 字符上限): {prefixed}")

            existing_names = {t["function"]["name"] for t in registry.get_tools_spec()}
            if prefixed in existing_names:
                raise ValueError(f"MCP 工具命名冲突: 工具 '{prefixed}' 已在注册表中存在，严禁覆盖！")

            desc = f"[MCP: {client.name}] {tool_def.description or '无描述'}"
            schema = tool_def.input_schema

            def _create_handler(c: MCPClient, orig_tool: str):
                async def _handler(**kwargs) -> str:
                    return await c.call_tool(orig_tool, kwargs)
                return _handler

            registry.register(
                name=prefixed,
                description=desc,
                parameters=schema,
                handler=_create_handler(client, raw_name),
            )
            self._mounted_tools[prefixed] = client.name
            newly_mounted.append(prefixed)
            logger.debug(f"[MCPTool] 已挂载远程工具: {prefixed}")

        self.clients[client.name] = client
        return newly_mounted

    async def _mount_client(self, client: MCPClient) -> List[str]:
        """
        内部异步初始化并挂载一个新的 MCPClient 到当前 ToolRegistry。
        返回挂载成功的工具名称列表。
        """
        if self._registry is None:
            raise RuntimeError("MCPTool 尚未向 ToolRegistry 挂载注册 (register_to)")

        # 握手并拉取工具
        await client.connect()
        await client.list_tools()

        return self._register_client_tools(self._registry, client)

    async def _run_connect_mcp(self, server_name: str) -> str:
        """执行 connect_mcp 工具处理逻辑"""
        clean_name = normalize_mcp_name(server_name)

        if clean_name in self.clients:
            existing_tools = [t for t, s in self._mounted_tools.items() if s == clean_name]
            return f"MCP 服务端 '{clean_name}' 已连接，已挂载工具: {', '.join(existing_tools) or '无'}"

        factory = self.factories.get(clean_name)
        if not factory:
            available = ", ".join(sorted(self.factories.keys())) or "none"
            return f"Error: 未知的 MCP 服务 '{server_name}'。当前可动态连接的候选列表: {available}"

        try:
            client = factory()
            mounted = await self._mount_client(client)
            return (
                f"成功连接至 MCP 服务端 '{clean_name}'！\n"
                f"已动态发现并挂载 {len(mounted)} 个远程工具: {', '.join(mounted)}"
            )
        except Exception as e:
            logger.error(f"[MCPTool] 动态连接 MCP 服务 '{server_name}' 失败: {e}")
            return f"Error: 连接 MCP 服务端 '{server_name}' 失败: {e}"

    async def close(self) -> None:
        """关闭所有已连接的客户端"""
        for client in self.clients.values():
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"[MCPTool] 关闭 client '{client.name}' 时异常被忽略: {e}")
        self.clients.clear()
        self._mounted_tools.clear()

    @classmethod
    def from_config(cls, config_path: Union[str, Path], workdir: Optional[Path] = None) -> MCPTool:
        """
        从标准声明式配置文件 (如 mcp.json) 自动解析构建 MCPTool 及懒加载工厂集合。
        对齐 Claude Desktop / Cursor 官方配置规范。
        """
        cfg_file = Path(config_path)
        base_dir = (workdir or Path.cwd()).resolve()

        if not cfg_file.is_absolute():
            cfg_file = (base_dir / cfg_file).resolve()

        if not cfg_file.exists():
            logger.debug(f"[MCPTool] MCP 配置文件未找到 ({cfg_file})，初始化为空插件")
            return cls()

        try:
            raw_text = cfg_file.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except Exception as e:
            logger.error(f"[MCPTool] 读取或解析 MCP 配置文件失败 ({cfg_file}): {e}")
            return cls()

        server_map = data.get("mcpServers") or data.get("servers") or {}
        if not isinstance(server_map, dict):
            logger.warning(f"[MCPTool] 配置文件中的 mcpServers 不是有效字典对象: {cfg_file}")
            return cls()

        factories: Dict[str, Callable[[], MCPClient]] = {}
        descriptions: Dict[str, str] = {}

        for raw_name, s_cfg in server_map.items():
            if not isinstance(s_cfg, dict):
                continue
            try:
                server_name = normalize_mcp_name(raw_name)
            except ValueError:
                continue

            raw_command = s_cfg.get("command")
            if not raw_command:
                continue

            # 如果 command 是 "python" 或 "python3"，统一对齐为当前运行环境的 sys.executable
            if raw_command in ("python", "python3"):
                exec_cmd = sys.executable
            else:
                exec_cmd = raw_command

            args = s_cfg.get("args", [])
            expanded_args: List[str] = []
            for a in args:
                exp = os.path.expandvars(str(a))
                script_path = base_dir / exp
                if script_path.exists():
                    expanded_args.append(str(script_path))
                else:
                    expanded_args.append(exp)

            full_command = [exec_cmd] + expanded_args
            custom_env = s_cfg.get("env")
            if custom_env and isinstance(custom_env, dict):
                custom_env = {k: os.path.expandvars(str(v)) for k, v in custom_env.items()}

            def _create_factory(name: str, cmd: List[str], env: Optional[Dict[str, str]]):
                return lambda: MCPClient(
                    name=name,
                    transport=StdioTransport(command=cmd, cwd=base_dir, env=env),
                )

            factories[server_name] = _create_factory(server_name, full_command, custom_env)
            desc = str(s_cfg.get("description", "")).strip()
            descriptions[server_name] = desc or f"外部 MCP 服务: {server_name}"

        logger.info(f"[MCPTool] 成功从 {cfg_file.name} 载入 {len(factories)} 个 MCP 服务配置: {list(factories.keys())}")
        return cls(factories=factories, descriptions=descriptions)
