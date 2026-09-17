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

import inspect
from typing import Any, Callable, Dict, List


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
        self._hooks: Dict[str, List[Callable]] = {}

    def register(self, event: str, callback: Callable) -> None:
        """
        统一注册钩子：向指定事件挂载一个回调函数（支持同步 def 与异步 async def）
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def on(self, event: str) -> Callable:
        """
        统一注册装饰器语法糖：
        @manager.on("PreToolUse")
        async def my_hook(...): ...
        """
        def decorator(callback: Callable) -> Callable:
            self.register(event, callback)
            return callback
        return decorator

    def register_plugin(self, plugin: Any) -> None:
        """
        统一插件化装配 (Plugin Architecture)：
        挂载实现了 register_to(manager) 契约的复合切面插件。
        """
        if hasattr(plugin, "register_to") and callable(plugin.register_to):
            plugin.register_to(self)
        else:
            raise TypeError(f"插件 {plugin} 必须实现 'register_to(manager)' 方法以供挂载")

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


# ----------------------------------------------------------------------
# 模块自测
# ----------------------------------------------------------------------

async def _smoke_test():
    print("启动 runtime/hooks.py 纯净钩子调度引擎自测...")

    manager = HookManager()

    # 1. 显式注册一个同步权限拦截回调
    def simple_permission_check(tool_name: str, args: dict):
        if tool_name == "bash" and "rm -rf" in args.get("command", ""):
            return "Permission denied: dangerous command blocked"
        return None

    manager.register("PreToolUse", simple_permission_check)

    # 2. 装饰器语法糖注册一个异步统计钩子
    counter = {"steps": 0}

    @manager.on("StepStart")
    async def track_steps(step: int):
        counter["steps"] = step

    # 验证 1: 正常工具放行 (返回 None)
    res_ok = await manager.trigger("PreToolUse", tool_name="read_file", args={"path": "a.txt"})
    assert res_ok is None
    print("1. 统一 trigger 放行测试通过 (返回 None)")

    # 验证 2: 危险工具拦截 (返回非 None 短路结果)
    res_block = await manager.trigger("PreToolUse", tool_name="bash", args={"command": "rm -rf /"})
    assert res_block == "Permission denied: dangerous command blocked"
    print("2. 统一 trigger 短路拦截测试通过 (返回拦截原因)")

    # 验证 3: 触发异步 StepStart
    await manager.trigger("StepStart", step=3)
    assert counter["steps"] == 3
    print("3. @manager.on 装饰器与异步事件触发通过")

    print("runtime/hooks.py 纯净 HookManager 自测全部通过！")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
