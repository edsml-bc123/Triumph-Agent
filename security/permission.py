"""
triumph-agent 权限审计与策略拦截引擎 (Security & Permission Engine)
=====================================================================
核心设计思想（吸收 learn-claude-code s03/s04 安全哲学）：
1. 三级策略决策矩阵 (Auto / Ask / Deny)：
   - AUTO: 只读与安全操作自动放行，保障开发效率；
   - ASK:  潜在破坏性操作（写文件、删除、覆盖）在终端提示人类审计确认 [y/N]；
   - DENY: 绝对高危命令（提权、系统格式化、跨工作区逃逸）一票否决阻断，绝不询问；
2. 沙箱防逃逸边界检查 (Workspace Sandbox Guard)：
   - 严禁未经授权读写工作区之外的文件或敏感系统凭证（.env, ~/.ssh 等）；
3. 严格契约与短路机制 (PreToolUse Hook Integration)：
   - 作为 PreToolUse 钩子执行，审查通过返回 None，审查拦截返回具体拒绝原因字符串。
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 确保项目根目录在 sys.path 中
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from loguru import logger
from runtime.hooks import HookManager
from runtime.state import AgentState


class PermissionAction(str, Enum):
    """权限决策动作"""
    AUTO = "auto"  # 自动放行
    ASK = "ask"    # 需要人类确认 [y/N]
    DENY = "deny"  # 一票否决阻断


class PermissionHook:
    """
    工业级权限审计与安全拦截钩子 (PreToolUse Hook)
    提供多维度安全策略评估与人类确认交互能力。
    """

    # 1. 绝对高危黑名单 (命中则强制 DENY，绝不询问)
    DEFAULT_DENY_PATTERNS: List[str] = [
        r"\brm\s+-(?:r[fF]|f[rR])\s+(?:/|\*|/\*|~|~/)(\s|$)",  # rm -rf /, rm -rf *
        r"\bsudo\b",                                            # 提权命令
        r"\bsu\b(?:\s+-|\s+root|\s*$)",                         # 切换 root
        r"\b(?:shutdown|reboot|poweroff|init\s+0)\b",           # 停机重启
        r"\bmkfs(?:\.\w+)?\b",                                  # 格式化文件系统
        r"\bdd\s+if=",                                          # 裸磁盘镜像写入
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",           # Fork 炸弹
    ]

    # 2. 潜在破坏性命令模式 (命中则转为 ASK，需人类授权)
    DESTRUCTIVE_PATTERNS: List[str] = [
        r"(?i)(?:^|[;&|()\n])\s*(?:rm|del|unlink)\b",           # 删除文件/目录
        r"\bchmod\s+(?:-[R\w]+\s+)?777\b",                      # 宽泛权限赋予
        r"\bchown\b",                                           # 更改所有权
        r"\bgit\s+reset\s+--hard\b",                            # 丢弃代码改动
        r"\bgit\s+clean\s+-[fF]",                               # 强力清理未跟踪文件
        r">\s*/(?:etc|bin|sbin|usr|var)",                       # 重定向覆写系统目录
        r"(?:^|[^<>&|])\s*>{1,2}\s*(?!/dev/null\b)[^\s>&|]+",  # 重定向写/追加文件 (防 bash echo > file 绕过)
    ]

    # 3. 敏感文件保护黑名单 (严禁直接访问)
    SENSITIVE_FILES: Set[str] = {
        ".env",
        "id_rsa",
        "id_ed25519",
        "authorized_keys",
        "/etc/passwd",
        "/etc/shadow",
    }

    def __init__(
        self,
        workdir: Optional[Path] = None,
        interactive: bool = True,
        auto_approve_writes: bool = False,
    ):
        """
        :param workdir: 工作区根目录（防逃逸基准路径）
        :param interactive: 是否启用终端交互式人类提问（自动化测试设为 False）
        :param auto_approve_writes: 是否静默信任写文件操作
        """
        self.workdir = (workdir or Path.cwd()).resolve()
        self.interactive = interactive
        self.auto_approve_writes = auto_approve_writes

        # 编译正则提高匹配性能
        self._deny_regexes = [re.compile(p) for p in self.DEFAULT_DENY_PATTERNS]
        self._destructive_regexes = [re.compile(p) for p in self.DESTRUCTIVE_PATTERNS]

    def evaluate(self, tool_name: str, args: Dict[str, Any]) -> tuple[PermissionAction, str]:
        """
        核心静态安全评估矩阵：
        :return: (PermissionAction, reason)
        """
        # A. 文件路径沙箱越界与敏感文件审查
        if tool_name in ("read_file", "write_file", "edit_file"):
            raw_path = str(args.get("path", "")).strip()
            if not raw_path:
                return PermissionAction.DENY, "参数错误：文件路径不能为空"

            # 检查敏感文件名
            path_obj = Path(raw_path)
            if path_obj.name in self.SENSITIVE_FILES or any(raw_path.endswith(s) for s in self.SENSITIVE_FILES):
                return PermissionAction.DENY, f"安全阻断：禁止访问敏感配置文件或凭据文件 '{raw_path}'"

            # 跨工作区逃逸检测
            resolved = (self.workdir / path_obj).resolve()
            try:
                resolved.relative_to(self.workdir)
            except ValueError:
                return PermissionAction.DENY, f"安全阻断：路径 '{raw_path}' 超出工作区沙箱限制"

            # 写操作审查
            if tool_name in ("write_file", "edit_file"):
                if self.auto_approve_writes:
                    return PermissionAction.AUTO, "写操作信任模式已启用"
                return PermissionAction.ASK, f"请求修改文件: {raw_path}"

            return PermissionAction.AUTO, "只读文件操作放行"

        # B. 目录检索只读放行
        if tool_name == "glob":
            return PermissionAction.AUTO, "目录搜索只读操作放行"

        # C. Bash 命令行指令安全审查
        if tool_name == "bash":
            command = str(args.get("command", "")).strip()
            if not command:
                return PermissionAction.DENY, "参数错误：bash 命令不能为空"

            # 1. 优先命中一票否决黑名单
            for regex in self._deny_regexes:
                if regex.search(command):
                    return PermissionAction.DENY, f"安全一票否决：命令匹配绝对高危黑名单规则 '{regex.pattern}'"

            # 2. 命中破坏性操作，转入人工确认
            for regex in self._destructive_regexes:
                if regex.search(command):
                    return PermissionAction.ASK, f"潜在破坏性指令需要确认: {command}"

            # 默认 bash 命令放行 (如 ls, cat, git, python 等)
            return PermissionAction.AUTO, "无害命令自动放行"

        # D. 其它所有系统受控工具（如 subagent 任务编排、计算器、数据分析等）默认放行
        # 遵循开放世界假定 (Open World Assumption) 与风险驱动模型 (对齐 learn-claude-code s06)，彻底解除工具名强耦合
        return PermissionAction.AUTO, f"受控工具 '{tool_name}' 自动放行"

    def ask_human_confirmation(self, tool_name: str, args: Dict[str, Any], reason: str) -> bool:
        """
        在终端提示人类进行审计决策 [y/N]
        """
        if not self.interactive:
            # 非交互式环境（如无人值守单测/后台执行），默认保守拒绝
            logger.warning(f"[Permission Audit] 非交互模式下拦截操作: {reason}")
            return False

        print("\n" + "-" * 60)
        print("[安全审计提示] 检测到需要人类授权的工具调用:")
        print(f"  工具名称: {tool_name}")
        print(f"  调用参数: {args}")
        print(f"  审查原因: {reason}")
        print("-" * 60)

        # 清理终端输入流中可能残留的未消费脏输入，确保安全授权 100% 来自当下真实的物理敲击
        try:
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass

        try:
            choice = input("是否允许该操作执行? [y/N]: ").strip().lower()
            allowed = choice in ("y", "yes")
            if allowed:
                logger.info(f"[Permission Audit] 人类已授权执行: {tool_name}")
            else:
                logger.warning(f"[Permission Audit] 人类已拒绝执行: {tool_name}")
            return allowed
        except (EOFError, KeyboardInterrupt):
            print()
            logger.warning("[Permission Audit] 授权等待被中断，默认拒绝")
            return False

    def check(self, tool_name: str, args: Dict[str, Any], state: Optional[AgentState] = None) -> Optional[str]:
        """
        PreToolUse 钩子标准执行接口：
        - 放行返回 None
        - 阻断返回错误说明字符串（将直接回填给大模型）
        """
        action, reason = self.evaluate(tool_name, args)

        if action == PermissionAction.DENY:
            logger.error(f"[Permission Blocked] {tool_name} | {reason}")
            return f"Permission Denied: {reason}"

        if action == PermissionAction.ASK:
            allowed = self.ask_human_confirmation(tool_name, args, reason)
            if not allowed:
                return f"Permission Denied: Operation declined by user ({reason})"
            return None

        # AUTO 放行
        return None

    def register_to(self, manager: HookManager) -> None:
        """实现插件装配协议：将自身注册到 PreToolUse 钩子点"""
        manager.register("PreToolUse", self.check)


