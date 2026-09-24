"""
triumph-agent MCP 模块单元与集成测试 (Tests for MCP)
=====================================================
覆盖协议解析、规范化、进程内与子进程管道通信、插件适配器与 AgentLoop 端到端。
"""

from __future__ import annotations

import json
import sys
import pytest
from pathlib import Path
from typing import Any, Dict, Optional

from client import LLMResponse, ToolCall
from mcp import (
    InProcessTransport,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPClient,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTool,
    MCPToolDefinition,
    StdioTransport,
    normalize_mcp_name,
)
from runtime.hooks import HookManager
from runtime.loop import AgentLoop
from runtime.state import AgentState, AgentStatus
from tools.registry import ToolRegistry


# ----------------------------------------------------------------------
# 1. 协议与命名测试 (Protocol & Normalization Tests)
# ----------------------------------------------------------------------

def test_normalize_mcp_name():
    """测试名称规范化与非法字符转译"""
    assert normalize_mcp_name("my-server") == "my-server"
    assert normalize_mcp_name("my_server") == "my_server"
    assert normalize_mcp_name("my.server:8080") == "my_server_8080"
    assert normalize_mcp_name("  tool@test#1  ") == "tool_test_1"

    with pytest.raises(ValueError):
        normalize_mcp_name("   ")

    with pytest.raises(ValueError):
        normalize_mcp_name("@#$%^&*")


def test_jsonrpc_request_serialization():
    """测试 JSON-RPC 2.0 请求报文序列化"""
    req = JSONRPCRequest(method="test_method", params={"a": 1}, id=101)
    data = json.loads(req.to_json())
    assert data["jsonrpc"] == "2.0"
    assert data["method"] == "test_method"
    assert data["params"] == {"a": 1}
    assert data["id"] == 101

    # 通知 (id 为 None)
    notif = JSONRPCRequest(method="notifications/test", params=None, id=None)
    data_notif = json.loads(notif.to_json())
    assert "id" not in data_notif
    assert "params" not in data_notif
    assert data_notif["method"] == "notifications/test"


def test_jsonrpc_response_parsing():
    """测试 JSON-RPC 2.0 响应报文反序列化"""
    # 成功响应
    success_dict = {"jsonrpc": "2.0", "id": 1, "result": {"key": "val"}}
    resp = JSONRPCResponse.from_dict(success_dict)
    assert resp.id == 1
    assert resp.result == {"key": "val"}
    assert resp.error is None

    # 错误响应
    error_dict = {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": -32600, "message": "Invalid Request", "data": "extra"},
    }
    resp_err = JSONRPCResponse.from_dict(error_dict)
    assert resp_err.id == 2
    assert resp_err.result is None
    assert isinstance(resp_err.error, JSONRPCError)
    assert resp_err.error.code == -32600
    assert resp_err.error.message == "Invalid Request"
    assert resp_err.error.data == "extra"
    assert resp_err.error.to_dict() == {
        "code": -32600,
        "message": "Invalid Request",
        "data": "extra",
    }


def test_mcp_exception_hierarchy():
    """测试 MCP 异常继承契约体系"""
    assert issubclass(MCPConnectionError, MCPError)
    assert issubclass(MCPProtocolError, MCPError)


def test_mcp_tool_definition():
    """测试远程工具声明解析与 schema 默认补齐"""
    d1 = {
        "name": "calc_add",
        "description": "加法计算",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }
    tool_def = MCPToolDefinition.from_dict(d1)
    assert tool_def.name == "calc_add"
    assert tool_def.description == "加法计算"
    assert "properties" in tool_def.input_schema

    # 缺失 schema
    d2 = {"name": "test_empty"}
    t2 = MCPToolDefinition.from_dict(d2)
    assert t2.input_schema == {"type": "object", "properties": {}}


# ----------------------------------------------------------------------
# 2. 进程内传输与 MCPClient 生命周期 (In-Process Transport Tests)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_in_process_client_lifecycle():
    """测试通过 InProcessTransport 模拟的完整客户端生命周期"""
    initialized_notified = False

    def mock_router(method: str, params: Optional[Dict[str, Any]]) -> Any:
        nonlocal initialized_notified
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-server", "version": "1.2.3"},
            }
        elif method == "notifications/initialized":
            initialized_notified = True
            return None
        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "greet",
                        "description": "问候工具",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            tool_name = params.get("name") if params else None
            args = params.get("arguments", {}) if params else {}
            if tool_name == "greet":
                return {
                    "content": [{"type": "text", "text": f"Hello, {args.get('name')}!"}],
                    "isError": False,
                }
            raise ValueError(f"未知工具: {tool_name}")
        raise ValueError(f"未支持的方法: {method}")

    transport = InProcessTransport(router=mock_router)
    client = MCPClient(name="mock_svc", transport=transport)

    # 1. 握手
    await client.connect()
    assert client.server_info is not None
    assert client.server_info.name == "mock-server"
    assert client.server_info.version == "1.2.3"
    assert initialized_notified is True

    # 2. 列举工具
    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "greet"
    assert "greet" in client.tools

    # 3. 执行工具
    res = await client.call_tool("greet", {"name": "Triumph"})
    assert res == "Hello, Triumph!"

    # 4. 远程报错
    err_res = await client.call_tool("not_exist", {})
    assert "Error:" in err_res

    # 5. 关闭
    await client.close()


@pytest.mark.asyncio
async def test_mcp_protocol_error_on_handshake_and_list():
    """测试服务端返回协议错误报文时正确抛出 MCPProtocolError"""
    # 1. 握手阶段服务端报错
    def fail_init_router(method: str, params: Optional[Dict[str, Any]]) -> Any:
        if method == "initialize":
            raise RuntimeError("Version handshake rejected")
        return {}

    client_bad_init = MCPClient(name="bad_init", transport=InProcessTransport(router=fail_init_router))
    with pytest.raises(MCPProtocolError) as exc_info:
        await client_bad_init.connect()
    assert "初始化协商失败" in str(exc_info.value)

    # 2. 列举工具阶段服务端报错
    def fail_list_router(method: str, params: Optional[Dict[str, Any]]) -> Any:
        if method == "initialize":
            return {"serverInfo": {"name": "test"}}
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            raise RuntimeError("Tools registry corrupted")
        return {}

    client_bad_list = MCPClient(name="bad_list", transport=InProcessTransport(router=fail_list_router))
    await client_bad_list.connect()
    with pytest.raises(MCPProtocolError) as exc_info2:
        await client_bad_list.list_tools()
    assert "拉取工具列表失败" in str(exc_info2.value)


# ----------------------------------------------------------------------
# 3. STDIO 管道子进程传输测试 (StdioTransport Tests)
# ----------------------------------------------------------------------

# 用于子进程运行的完整真实 MCP Server 微型脚本
_MOCK_SERVER_SCRIPT = """
import sys
import json

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        method = req.get("method")
        msg_id = req.get("id")
        params = req.get("params", {})

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stdio-mock-server", "version": "0.1.0"}
                }
            }
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()
        elif method == "notifications/initialized":
            # 通知无响应
            pass
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "multiply",
                            "description": "乘法计算",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"}
                                },
                                "required": ["x", "y"]
                            }
                        }
                    ]
                }
            }
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()
        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if tool_name == "multiply":
                x = args.get("x", 0)
                y = args.get("y", 0)
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": str(x * y)}],
                        "isError": False
                    }
                }
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Tool {tool_name} not found"}
                }
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
"""


@pytest.mark.asyncio
async def test_stdio_transport_live_process(tmp_path: Path):
    """测试通过标准 STDIO 管道拉起真实 Python 子进程进行双向通信"""
    server_py = tmp_path / "server.py"
    server_py.write_text(_MOCK_SERVER_SCRIPT, encoding="utf-8")

    transport = StdioTransport(
        command=[sys.executable, str(server_py)],
        cwd=tmp_path,
    )
    client = MCPClient(name="stdio_calc", transport=transport)

    try:
        # 1. 握手协商
        await client.connect()
        assert client.server_info is not None
        assert client.server_info.name == "stdio-mock-server"

        # 2. 拉取工具
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "multiply"

        # 3. 调用工具 (6 * 7 = 42)
        out = await client.call_tool("multiply", {"x": 6, "y": 7})
        assert out == "42"

        # 4. 调用不存在的工具
        err_out = await client.call_tool("unknown", {})
        assert "Error:" in err_out
    finally:
        await client.close()
        # 确认子进程已终止
        assert transport.process is None


@pytest.mark.asyncio
async def test_stdio_transport_timeout(tmp_path: Path):
    """测试子进程请求超时保护"""
    # 模拟一个挂起不响应的脚本
    hang_script = "import time; time.sleep(10)"
    script_path = tmp_path / "hang.py"
    script_path.write_text(hang_script, encoding="utf-8")

    transport = StdioTransport(
        command=[sys.executable, str(script_path)],
        cwd=tmp_path,
    )
    client = MCPClient(name="hang_client", transport=transport)

    try:
        # 发送 initialize 但设定很短的超时，验证引发超时异常
        await transport.start()
        req = JSONRPCRequest(method="initialize", id=1)
        with pytest.raises(MCPConnectionError) as exc_info:
            await transport.send_request(req, timeout=0.3)
        assert "超时" in str(exc_info.value)
    finally:
        await client.close()


# ----------------------------------------------------------------------
# 4. MCPTool 插件适配器与 ToolRegistry 集成测试
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_tool_adapter_mount(tmp_path: Path):
    """测试 MCPTool 挂载到 ToolRegistry，并遵循命名空间隔离"""
    registry = ToolRegistry(workdir=tmp_path)

    # 创建一个 mock in-process client
    def router(method: str, params: Optional[Dict[str, Any]]) -> Any:
        if method == "initialize":
            return {"serverInfo": {"name": "remote-db"}}
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "query_sql",
                        "description": "执行 SQL",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"sql": {"type": "string"}},
                            "required": ["sql"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            sql = (params or {}).get("arguments", {}).get("sql", "")
            return {"content": [{"type": "text", "text": f"Result for: {sql}"}]}
        return {}

    def client_factory() -> MCPClient:
        return MCPClient(name="db_service", transport=InProcessTransport(router=router))

    mcp_tool = MCPTool(factories={"db_service": client_factory})
    registry.register_plugin(mcp_tool)

    # 初始状态下仅注册了 connect_mcp 工具元数据
    specs_before = {t["function"]["name"]: t for t in registry.get_tools_spec()}
    assert "connect_mcp" in specs_before
    assert "mcp__db_service__query_sql" not in specs_before

    # 动态连接后触发工具挂载
    mount_res = await registry.execute("connect_mcp", {"server_name": "db_service"})
    assert "成功连接至 MCP 服务端 'db_service'" in mount_res

    specs_after = {t["function"]["name"]: t for t in registry.get_tools_spec()}
    assert "mcp__db_service__query_sql" in specs_after

    # 直接执行注册后的远程包装工具
    res = await registry.execute("mcp__db_service__query_sql", {"sql": "SELECT 1"})
    assert res == "Result for: SELECT 1"

    await registry.close()


@pytest.mark.asyncio
async def test_mcp_tool_name_conflict_prevention(tmp_path: Path):
    """测试工具名称冲突时防御性抛错"""
    registry = ToolRegistry(workdir=tmp_path)

    # 先注册一个已存在的工具
    async def dummy(**kwargs):
        return "ok"

    registry.register(
        name="mcp__demo__conflict",
        description="existing",
        parameters={"type": "object", "properties": {}},
        handler=dummy,
    )

    # 构造同名工具客户端
    def router(method: str, params: Optional[Dict[str, Any]]) -> Any:
        if method == "initialize":
            return {"serverInfo": {"name": "demo"}}
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return {
                "tools": [
                    {"name": "conflict", "description": "dup", "inputSchema": {"type": "object"}}
                ]
            }
        return {}

    def client_factory() -> MCPClient:
        return MCPClient(name="demo", transport=InProcessTransport(router=router))

    mcp_tool = MCPTool(factories={"demo": client_factory})
    registry.register_plugin(mcp_tool)

    # 通过 connect_mcp 挂载时，捕获异常并返回友好的错误提示
    connect_res = await registry.execute("connect_mcp", {"server_name": "demo"})
    assert "Error:" in connect_res
    assert "命名冲突" in connect_res

    # 同时测试底层 _mount_client 直接调用时抛出严格的 ValueError
    conflict_client = client_factory()
    with pytest.raises(ValueError) as exc:
        await mcp_tool._mount_client(conflict_client)
    assert "命名冲突" in str(exc.value)

    await registry.close()


@pytest.mark.asyncio
async def test_mcp_dynamic_connect(tmp_path: Path):
    """测试通过 connect_mcp 动态发现与挂载远程服务"""
    registry = ToolRegistry(workdir=tmp_path)

    def router(method: str, params: Optional[Dict[str, Any]]) -> Any:
        if method == "initialize":
            return {"serverInfo": {"name": "analytics"}}
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "calc_metrics",
                        "description": "计算指标",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        return {}

    def client_factory():
        return MCPClient(name="analytics", transport=InProcessTransport(router=router))

    mcp_tool = MCPTool(factories={"analytics": client_factory})
    registry.register_plugin(mcp_tool)

    # 1. 尝试连接未知服务
    unknown_res = await registry.execute("connect_mcp", {"server_name": "non_exist"})
    assert "Error:" in unknown_res
    assert "候选列表: analytics" in unknown_res

    # 2. 动态连接 analytics 服务
    connect_res = await registry.execute("connect_mcp", {"server_name": "analytics"})
    assert "成功连接至 MCP 服务端 'analytics'" in connect_res
    assert "mcp__analytics__calc_metrics" in connect_res

    # 3. 验证新工具已成功挂载到 ToolRegistry 并可调用
    specs = {t["function"]["name"] for t in registry.get_tools_spec()}
    assert "mcp__analytics__calc_metrics" in specs

    # 4. 重复连接提示已连接
    dup_res = await registry.execute("connect_mcp", {"server_name": "analytics"})
    assert "已连接" in dup_res

    await registry.close()


# ----------------------------------------------------------------------
# 5. AgentLoop 端到端集成调用 MCP 工具测试
# ----------------------------------------------------------------------

class MockMCPDashScopeClient:
    """模拟大模型：第一轮先调用 connect_mcp 挂载服务，第二轮调用 mcp__calc__sum，第三轮给出最终回答"""
    def __init__(self):
        self.call_count = 0
        self.default_model = "qwen3.8-flash"
        self.evaluator_model = "qwen3.6-flash"

    async def chat_completion(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                content="我需要连接 calc 服务来完成求和计算...",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_connect_1",
                        name="connect_mcp",
                        arguments_raw=json.dumps({"server_name": "calc"}),
                    )
                ],
            )
        elif self.call_count == 2:
            return LLMResponse(
                content="正在为您计算数字和...",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_mcp_123",
                        name="mcp__calc__sum",
                        arguments_raw=json.dumps({"nums": [10, 20, 30]}),
                    )
                ],
            )
        else:
            return LLMResponse(
                content="计算结果是 60，任务顺利完成！",
                finish_reason="stop",
                tool_calls=[],
            )


@pytest.mark.asyncio
async def test_agent_loop_e2e_mcp_tool(tmp_path: Path):
    """测试 AgentLoop 端到端执行调用远程 MCP 工具"""
    registry = ToolRegistry(workdir=tmp_path)

    def router(method: str, params: Optional[Dict[str, Any]]) -> Any:
        if method == "initialize":
            return {"serverInfo": {"name": "calc"}}
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "sum",
                        "description": "求和计算",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "nums": {"type": "array", "items": {"type": "number"}}
                            },
                            "required": ["nums"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            nums = (params or {}).get("arguments", {}).get("nums", [])
            total = sum(nums)
            return {"content": [{"type": "text", "text": f"SUM={total}"}]}
        return {}

    def client_factory() -> MCPClient:
        return MCPClient(name="calc", transport=InProcessTransport(router=router))

    mcp_tool = MCPTool(factories={"calc": client_factory})
    registry.register_plugin(mcp_tool)

    mock_llm = MockMCPDashScopeClient()
    loop = AgentLoop(client=mock_llm, registry=registry, hooks=HookManager())

    state = AgentState(max_steps=5)
    state.add_user_message("帮我计算 10, 20, 30 的和")

    await loop.run(state)

    assert state.status == AgentStatus.SUCCESS
    tool_messages = [m.get("content", "") for m in state.messages if m.get("role") == "tool"]
    assert any("成功连接至 MCP 服务端 'calc'" in m for m in tool_messages)
    assert any("SUM=60" in m for m in tool_messages)
    assert "计算结果是 60" in state.messages[-1]["content"]

    await registry.close()


@pytest.mark.asyncio
async def test_tool_registry_lifecycle_close(tmp_path: Path):
    """测试 ToolRegistry.close() 正确级联触发 ClosablePlugin 的 close() 协议方法"""
    from tools import ClosablePlugin, ToolPlugin

    registry = ToolRegistry(workdir=tmp_path)
    closed_plugins = []

    class DummyClosable:
        def register_to(self, reg: ToolRegistry) -> None:
            pass

        async def close(self) -> None:
            closed_plugins.append("closable_plugin")

    class DummyNonClosable:
        def register_to(self, reg: ToolRegistry) -> None:
            pass

    plugin1 = DummyClosable()
    plugin2 = DummyNonClosable()

    # 验证协议类型断言
    assert isinstance(plugin1, ToolPlugin)
    assert isinstance(plugin1, ClosablePlugin)
    assert isinstance(plugin2, ToolPlugin)
    assert not isinstance(plugin2, ClosablePlugin)

    registry.register_plugin(plugin1)
    registry.register_plugin(plugin2)

    await registry.close()

    assert "closable_plugin" in closed_plugins
    assert len(registry._plugins) == 0


@pytest.mark.asyncio
async def test_live_system_server_stdio():
    """测试真实的 mcp_servers/system_server.py 脚本通过 STDIO 管道的端到端调用"""
    project_root = Path(__file__).resolve().parent.parent
    server_path = project_root / "mcp_servers" / "system_server.py"
    assert server_path.exists()

    transport = StdioTransport(
        command=[sys.executable, str(server_path)],
        cwd=project_root,
    )
    client = MCPClient(name="system_info", transport=transport)

    try:
        await client.connect()
        assert client.server_info is not None
        assert client.server_info.name == "system_info"

        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert "get_system_info" in names
        assert "get_hardware_usage" in names

        # 执行系统信息工具
        sys_res = await client.call_tool("get_system_info", {})
        sys_data = json.loads(sys_res)
        assert "platform" in sys_data
        assert "python_version" in sys_data

        # 执行硬件使用工具
        hw_res = await client.call_tool("get_hardware_usage", {})
        hw_data = json.loads(hw_res)
        assert "cpu_logical_cores" in hw_data
        assert "disk_total_gb" in hw_data
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_from_config_loading(tmp_path: Path):
    """测试从声明式 mcp.json 加载服务配置与服务目录注入"""
    config_file = tmp_path / "mcp.json"
    dummy_server_script = tmp_path / "my_server.py"
    dummy_server_script.write_text("# dummy", encoding="utf-8")

    cfg_data = {
        "mcpServers": {
            "test_svc": {
                "command": "python",
                "args": ["my_server.py"],
                "description": "测试外部服务描述",
            }
        }
    }
    config_file.write_text(json.dumps(cfg_data), encoding="utf-8")

    # 1. 正常解析
    tool = MCPTool.from_config(config_file, workdir=tmp_path)
    assert "test_svc" in tool.factories
    assert tool.descriptions["test_svc"] == "测试外部服务描述"

    # 2. 挂载到注册表，校验 connect_mcp 是否自动包含候选 enum 和服务目录
    reg = ToolRegistry(workdir=tmp_path)
    reg.register_plugin(tool)

    specs = {t["function"]["name"]: t for t in reg.get_tools_spec()}
    connect_tool = specs["connect_mcp"]
    connect_desc = connect_tool["function"]["description"]
    assert "test_svc: 测试外部服务描述" in connect_desc

    server_param = connect_tool["function"]["parameters"]["properties"]["server_name"]
    assert server_param.get("enum") == ["test_svc"]

    # 3. 配置文件不存在时安全退化为空实例
    empty_tool = MCPTool.from_config(tmp_path / "not_exist.json", workdir=tmp_path)
    assert len(empty_tool.factories) == 0

    await reg.close()
