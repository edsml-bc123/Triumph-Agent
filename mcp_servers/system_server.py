"""
triumph-agent 示范 MCP 外部服务 (System Information MCP Server)
================================================================
通过跨进程标准 STDIO 管道向客户端提供系统平台与硬件负载监控工具。
遵循 JSON-RPC 2.0 与 MCP 2024-11-05 规范。
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from typing import Any, Dict, Optional


def get_system_info() -> str:
    """获取基础系统平台与环境信息"""
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "node_hostname": platform.node(),
    }
    return json.dumps(info, ensure_ascii=False, indent=2)


def get_hardware_usage() -> str:
    """获取硬件资源与磁盘使用概要"""
    cpu_count = os.cpu_count() or 1
    total, used, free = shutil.disk_usage(os.getcwd())
    disk_gb = 1024 ** 3

    usage = {
        "cpu_logical_cores": cpu_count,
        "disk_total_gb": round(total / disk_gb, 2),
        "disk_used_gb": round(used / disk_gb, 2),
        "disk_free_gb": round(free / disk_gb, 2),
        "disk_used_pct": f"{round((used / total) * 100, 1)}%",
    }
    return json.dumps(usage, ensure_ascii=False, indent=2)


TOOLS = [
    {
        "name": "get_system_info",
        "description": "查询当前宿主机的操作系统平台、内核版本、CPU架构与Python运行时版本信息。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_hardware_usage",
        "description": "查询当前工作区所在机器的CPU逻辑核心数与磁盘存储空间占用情况。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def handle_request(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = req.get("method")
    msg_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "system_info",
                    "version": "1.0.0",
                },
            },
        }

    elif method == "notifications/initialized":
        # 单向通知，无需回复响应报文
        return None

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS,
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        if tool_name == "get_system_info":
            res_text = get_system_info()
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": res_text}],
                    "isError": False,
                },
            }
        elif tool_name == "get_hardware_usage":
            res_text = get_hardware_usage()
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": res_text}],
                    "isError": False,
                },
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{tool_name}' not found",
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Method '{method}' not implemented",
        },
    }


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        raw_text = line.strip()
        if not raw_text:
            continue

        try:
            req = json.loads(raw_text)
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            sys.stdout.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
