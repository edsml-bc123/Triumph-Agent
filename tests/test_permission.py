"""
triumph-agent 权限与安全审查机制自动化测试
=============================================
验证目标：
1. Auto / Ask / Deny 三级策略判定准确性；
2. 敏感凭据文件 (.env) 与工作区越界沙箱逃逸拦截；
3. rm -rf / 与提权命令的高危一票否决；
4. PreToolUse 钩子在 AgentLoop 中阻断物理执行并回填错误消息。
"""

import tempfile
from pathlib import Path
import pytest

from security.permission import PermissionAction, PermissionHook
from runtime.hooks import HookManager


@pytest.fixture
def sandbox_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        (workdir / "safe.txt").write_text("hello safe world", encoding="utf-8")
        yield workdir


def test_permission_auto_actions(sandbox_env):
    """验证只读操作自动放行 (AUTO)"""
    engine = PermissionHook(workdir=sandbox_env, interactive=False)

    # 1. 读文件放行
    act, reason = engine.evaluate("read_file", {"path": "safe.txt"})
    assert act == PermissionAction.AUTO

    # 2. 目录遍历放行
    act, reason = engine.evaluate("glob", {"pattern": "*.txt"})
    assert act == PermissionAction.AUTO

    # 3. 无害 shell 命令放行
    act, reason = engine.evaluate("bash", {"command": "ls -la"})
    assert act == PermissionAction.AUTO


def test_permission_deny_security_boundary(sandbox_env):
    """验证安全边界与一票否决 (DENY)"""
    engine = PermissionHook(workdir=sandbox_env, interactive=False)

    # 1. 跨越工作区逃逸拦截
    act, reason = engine.evaluate("read_file", {"path": "../../etc/passwd"})
    assert act == PermissionAction.DENY
    assert "超出工作区" in reason or "禁止访问" in reason

    # 2. 敏感凭据保护 (.env)
    act, reason = engine.evaluate("read_file", {"path": ".env"})
    assert act == PermissionAction.DENY
    assert "敏感" in reason

    # 3. 毁灭性命令拦截 (rm -rf /)
    act, reason = engine.evaluate("bash", {"command": "rm -rf /"})
    assert act == PermissionAction.DENY
    assert "黑名单" in reason

    # 4. sudo 提权拦截
    act, reason = engine.evaluate("bash", {"command": "sudo useradd test"})
    assert act == PermissionAction.DENY

    # 5. Fork 炸弹拦截
    act, reason = engine.evaluate("bash", {"command": ":(){ :|:& };:"})
    assert act == PermissionAction.DENY


def test_permission_ask_non_interactive_fallback(sandbox_env):
    """验证非交互模式下，潜在破坏性操作安全降级拦截 (ASK -> Deny Fallback)"""
    engine = PermissionHook(workdir=sandbox_env, interactive=False)

    # 1. 写文件在未授权时归入 ASK，非交互执行时阻断
    act, reason = engine.evaluate("write_file", {"path": "new.txt", "content": "data"})
    assert act == PermissionAction.ASK

    # check 接口直接阻断返回非 None
    err = engine.check("write_file", {"path": "new.txt", "content": "data"})
    assert err is not None
    assert "Permission Denied" in err

    # 2. 普通文件删除指令进入 ASK
    act, reason = engine.evaluate("bash", {"command": "rm safe.txt"})
    assert act == PermissionAction.ASK
    err = engine.check("bash", {"command": "rm safe.txt"})
    assert err is not None
    assert "Permission Denied" in err

    # 3. 验证 echo > file 重定向写操作被识别为 ASK
    act, reason = engine.evaluate("bash", {"command": 'echo "hello" > demo.txt'})
    assert act == PermissionAction.ASK
    err = engine.check("bash", {"command": 'echo "hello" > demo.txt'})
    assert err is not None
    assert "Permission Denied" in err

    # 4. 验证向 /dev/null 丢弃输出被允许 (AUTO)
    act, _ = engine.evaluate("bash", {"command": "python script.py > /dev/null 2>&1"})
    assert act == PermissionAction.AUTO


@pytest.mark.asyncio
async def test_permission_hook_integration(sandbox_env):
    """验证 PermissionHook 挂载到 HookManager 后对 PreToolUse 的短路拦截"""
    engine = PermissionHook(workdir=sandbox_env, interactive=False)
    hooks = HookManager()
    # 验证统一通过 register_plugin 挂载
    hooks.register_plugin(engine)

    # 1. 安全操作：触发 PreToolUse 应该返回 None (不阻断)
    res_allow = await hooks.trigger("PreToolUse", tool_name="read_file", args={"path": "safe.txt"})
    assert res_allow is None

    # 2. 高危操作：触发 PreToolUse 应该返回拦截原因 (短路阻断)
    res_block = await hooks.trigger("PreToolUse", tool_name="bash", args={"command": "rm -rf /"})
    assert res_block is not None
    assert "Permission Denied" in res_block
