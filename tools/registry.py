"""
triumph-agent 工具注册表与安全执行分发器 (Tool Registry)
==========================================================
核心设计思想：
1. 100% 纯净工具容器 (Pure Tool Container & Safe Dispatcher)：
   - 遵循单一职责（SRP）与开闭原则（OCP），容器初始化保持绝对纯粹白板状态；
   - 彻底消除构造函数标志位、死代码与全局可变单例状态；
   - 专司工具 Schema 注册、安全路径边界校验与防御性分发派发。
2. 统一插件装配契约 (Explicit Tool Plugin Protocol)：
   - 统一遵循 ToolPlugin 协议 (PEP 544 Protocol)；
   - 外部通过 register_plugin 显式装配 BuiltinToolsPlugin、SubAgentTool、BackgroundTaskTool。
3. 极端边界防御 (Defensive Dispatching)：
   - 工具名不存在时防崩溃（捕获 KeyError 转化为自愈提示）；
   - 参数解包异常时防崩溃（捕获 TypeError 转化为参数错误提示）；
   - 保证任何工具执行失败均以文本语义安全回填给模型，绝不崩坏主循环。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ToolPlugin(Protocol):
    """
    复合工具插件装配协议 (PEP 544 Protocol)
    所有需要向注册表注入工具集合的插件必须满足此契约。
    """

    def register_to(self, registry: ToolRegistry) -> None:
        """向 ToolRegistry 统一注册工具声明与执行 Handler"""
        ...


class ToolRegistry:
    """
    工业级工具注册与安全派发中心 (Pure Tool Container)
    初始化为一个纯净的空白工具容器，不暗中绑定任何具体业务插件。
    """

    def __init__(self, workdir: Optional[Path] = None):
        self.workdir = (workdir or Path.cwd()).resolve()
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._specs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 1. 物理沙箱与路径安全防逃逸
    # ------------------------------------------------------------------

    def safe_path(self, p: str) -> Path:
        """
        路径防逃逸检查：限制所有文件读写必须位于工作区内部，
        杜绝 ../../../../etc/passwd 等目录穿越破坏。
        """
        path = (self.workdir / p).resolve()
        if not path.is_relative_to(self.workdir):
            raise ValueError(f"安全违规：路径逃逸出工作区范围: {p}")
        return path

    # ------------------------------------------------------------------
    # 2. 百炼 / OpenAI 兼容 Schema 注册与动态管理
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """
        动态注册新工具。以 name 为唯一键自然映射，杜绝冗余扫描与重复。
        """
        self._specs[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._handlers[name] = handler

    def get_tools_spec(self) -> List[Dict[str, Any]]:
        """导出提供给大模型调用的标准工具协议声明列表"""
        return list(self._specs.values())

    def register_plugin(self, plugin: ToolPlugin) -> None:
        """
        统一工具插件装配协议 (对齐 HookManager.register_plugin)：
        挂载遵循 ToolPlugin 协议契约规范的复合工具插件。
        """
        plugin.register_to(self)

    def fork(self, exclude: Optional[Iterable[str]] = None) -> ToolRegistry:
        """
        通用派生/复制工具注册表：
        继承当前工作区沙箱与已注册工具，支持在派生时按名称排除特定工具集合。
        :param exclude: 需要在派生副本中排除的工具名称集合 (如 {"subagent"})
        :return: 独立隔离的 ToolRegistry 副本
        """
        excluded_names = set(exclude or [])
        forked = ToolRegistry(workdir=self.workdir)
        for tool_name, spec in self._specs.items():
            if tool_name in excluded_names:
                continue
            forked.register(
                name=tool_name,
                description=spec["function"]["description"],
                parameters=spec["function"]["parameters"],
                handler=self._handlers[tool_name],
            )
        return forked

    # ------------------------------------------------------------------
    # 3. 防御性分发执行 (Defensive Dispatcher)
    # ------------------------------------------------------------------

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        """
        分发并执行工具调用，实现全方位防御性隔离：
        1. 捕获不存在的工具名 (KeyError 防御)
        2. 捕获参数解包异常 (TypeError 防御)
        3. 捕获任何运行时未知错误 (Exception 防御)
        原生支持同步与异步工具 Handler，保证进程绝不崩溃，以自愈语义反馈模型。
        """
        if name not in self._handlers:
            available = list(self._handlers.keys())
            return (
                f"Error: Unknown tool '{name}'. "
                f"Available tools are: {available}. Please adjust your tool selection."
            )

        handler = self._handlers[name]
        try:
            if inspect.iscoroutinefunction(handler):
                return await handler(**args)
            return handler(**args)
        except TypeError as e:
            return (
                f"Error: Invalid arguments passed to tool '{name}'. "
                f"Details: {e}. Provided arguments: {args}"
            )
        except Exception as e:
            return f"Error: Unexpected failure while executing '{name}': {e}"
