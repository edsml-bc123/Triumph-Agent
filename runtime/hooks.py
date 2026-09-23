"""
triumph-agent 生命周期钩子调度引擎 (Unified Hook Manager)
=========================================================
核心设计思想（吸收 learn-claude-code s04 极简哲学）：
1. 统一注册与触发接口：
   - 全生命周期仅暴露单一注册方法 register(event, callback) 与单一触发方法 trigger(event, *args, **kwargs)；
   - 提供 @manager.on(event) 装饰器语法糖；
2. 异步与短路拦截语义 (Short-Circuit by Non-None Return)：
   - 兼容器普通同步函数与 async 协程回调；
   - 遵从教程核心契约：只要任一 Hook 返回非 None 值，立即判定为拦截并短路返回该结果；
3. 纯净基础设施：
   - 本模块仅维护生命周期注册表与事件调度，不硬编码绑定任何具体业务插件。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class HookPlugin(Protocol):
    """
    生命周期切面插件装配协议 (PEP 544 Protocol)
    所有扩展插件（如 TrajectoryHook, PermissionHook, MemoryHook 等）必须满足此契约。
    """

    def register_to(self, manager: HookManager) -> None:
        """向 HookManager 统一注册生命周期回调钩子"""
        ...


class HookManager:
    """
    极简统一的生命周期钩子管理器 (Event Bus)
    标准事件清单：
      - UserPromptSubmit: (prompt: str, state: AgentState)
      - StepStart:        (step: int, state: AgentState)
      - LLMResponse:      (response: LLMResponse, state: AgentState)
      - PreToolUse:       (tool_name: str, args: dict, state: AgentState) -> 返回非 None 即阻断工具执行
      - PostToolUse:      (tool_name: str, args: dict, output: str, duration_ms: float, tool_call_id: str, state: AgentState)
      - Stop:             (state: AgentState)
    """

    def __init__(self):
        # 内部维护事件与回调列表映射
        self._hooks: Dict[str, List[Callable[..., Any]]] = {}

    def register(self, event: str, callback: Callable[..., Any]) -> None:
        """
        统一注册钩子：向指定事件挂载一个回调函数（支持同步 def 与异步 async def）
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def on(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        统一注册装饰器语法糖：
        @manager.on("PreToolUse")
        async def my_hook(...): ...
        """
        def decorator(callback: Callable[..., Any]) -> Callable[..., Any]:
            self.register(event, callback)
            return callback
        return decorator

    def register_plugin(self, plugin: HookPlugin) -> None:
        """
        统一插件化装配 (Plugin Architecture)：
        挂载遵循 HookPlugin 契约规范的切面插件。
        """
        plugin.register_to(self)

    async def trigger(self, event: str, *args: Any, **kwargs: Any) -> Any:
        """
        统一触发钩子（教程核心逻辑）：
        - 顺序执行该事件下的所有回调；
        - 短路机制：一旦某个回调返回了非 None 结果，立即阻断后续钩子并直接返回此结果；
        - 若全部正常执行完毕且无拦截，返回 None。
        """
        callbacks = self._hooks.get(event, [])
        for callback in callbacks:
            if inspect.iscoroutinefunction(callback):
                result = await callback(*args, **kwargs)
            else:
                result = callback(*args, **kwargs)
                if inspect.iscoroutine(result):
                    result = await result

            # 核心契约：返回非 None 即代表短路阻断（如权限拒绝提示）
            if result is not None:
                return result

        return None


