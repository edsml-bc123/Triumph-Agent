"""
triumph-agent 工具注册表与安全执行分发器 (Tool Registry)
==========================================================
核心设计思想：
1. 继承 learn-claude-code s02 工业级设计：
   - safe_path 物理工作区防穿越逃逸 (is_relative_to 沙箱)；
   - run_bash 120s 强制超时 + 50000 字符物理截断 + 危险命令拦截；
   - run_read, run_write (自动建目录), run_edit (原子替换), run_glob (文件发现)。
2. 标准百炼 / OpenAI Function Calling 协议适配：
   - 自动生成遵循 OpenAI 规范的 tools 声明（function.parameters）。
3. 极端边界防御 (Defensive Dispatching)：
   - 工具名不存在时防崩溃（捕获 KeyError 转化为自愈提示）；
   - 参数解包异常时防崩溃（捕获 TypeError 转化为参数错误提示）；
   - 保证任何工具执行失败均以文本语义安全回填给模型，绝不崩坏主循环。
"""

import glob as g
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ToolPlugin(Protocol):
    """
    复合工具插件装配协议 (PEP 544 Protocol)
    所有需要向注册表注入工具集合的插件（如 SubAgentTool）必须满足此契约。
    """

    def register_to(self, registry: "ToolRegistry") -> None:
        """向 ToolRegistry 统一注册工具声明与执行 Handler"""
        ...


class ToolRegistry:
    """
    工业级工具注册与安全派发中心
    """

    def __init__(self, workdir: Optional[Path] = None):
        self.workdir = (workdir or Path.cwd()).resolve()
        self._handlers: Dict[str, Callable[..., str]] = {}
        self._specs: List[Dict[str, Any]] = []

        # 自动注册开箱即用的 5 大核心基础工具
        self._register_builtin_tools()

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
    # 2. 核心原子工具实现 (基于 s02 工业级实现)
    # ------------------------------------------------------------------

    def _run_bash(self, command: str) -> str:
        """在工作区安全执行终端命令"""
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked by safety policy"
        try:
            # 将当前运行 Agent 的 Python 解释器所在 bin 目录优先置顶于 PATH
            child_env = os.environ.copy()
            current_bin_dir = str(Path(sys.executable).parent)
            child_env["PATH"] = f"{current_bin_dir}:{child_env.get('PATH', '')}"

            r = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=120,
                env=child_env,
            )
            out = (r.stdout + r.stderr).strip()
            # 物理截断保护（50000字符），防止上下文被打爆
            if len(out) > 50000:
                return out[:50000] + "\n\n[Warning: Output truncated at 50,000 characters]"
            return out if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command execution timed out (120s limit reached)"
        except (FileNotFoundError, OSError) as e:
            return f"Error executing bash: {e}"

    def _run_read(self, path: str, limit: Optional[int] = None) -> str:
        """读取文件内容，支持行数限制与防逃逸"""
        try:
            target = self.safe_path(path)
            if not target.exists():
                return f"Error: File '{path}' does not exist."
            if not target.is_file():
                return f"Error: Path '{path}' is not a regular file."

            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            if limit and limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines omitted)"]
            return "\n".join(lines)
        except Exception as e:
            return f"Error reading file: {e}"

    def _run_write(self, path: str, content: str) -> str:
        """安全写入文件，自动创建父级目录"""
        try:
            target = self.safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _run_edit(self, path: str, old_text: str, new_text: str) -> str:
        """精确替换文件中的特定代码块"""
        try:
            target = self.safe_path(path)
            if not target.exists():
                return f"Error: File '{path}' does not exist."
            
            text = target.read_text(encoding="utf-8", errors="replace")
            if old_text not in text:
                return f"Error: Specified old_text not found in {path}. Please verify the target content."
            
            # 单次精确替换，防止误伤其他同名代码
            target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"

    def _run_glob(self, pattern: str) -> str:
        """在工作区内查找符合 glob 模式的文件"""
        try:
            matches = sorted({
                match for match in g.glob(
                    pattern, root_dir=self.workdir, recursive=True
                )
                if (self.workdir / match).resolve().is_relative_to(self.workdir)
            })
            shown = matches[:200]
            if len(matches) > 200:
                shown.append("... (more matches omitted; please narrow down your pattern)")
            return "\n".join(shown) if shown else "(no matches found)"
        except Exception as e:
            return f"Error running glob: {e}"

    # ------------------------------------------------------------------
    # 3. 百炼 / OpenAI 兼容 Schema 注册
    # ------------------------------------------------------------------

    def _register_builtin_tools(self):
        """注册内置工具及其标准 OpenAI Function Schema"""
        builtin = [
            (
                "bash",
                "在工作区安全执行终端命令（支持 pip/pytest/git 等常用命令）",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的 Shell 命令"}
                    },
                    "required": ["command"],
                },
                self._run_bash,
            ),
            (
                "read_file",
                "读取指定路径文本文件的内容",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对于工作区的文件路径"},
                        "limit": {"type": "integer", "description": "最大读取行数（可选）"},
                    },
                    "required": ["path"],
                },
                self._run_read,
            ),
            (
                "write_file",
                "写入内容到指定文件（若不存在则自动创建，若存在则完全覆盖）",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对于工作区的文件路径"},
                        "content": {"type": "string", "description": "要写入的完整文本内容"},
                    },
                    "required": ["path", "content"],
                },
                self._run_write,
            ),
            (
                "edit_file",
                "精确替换文件中的指定代码片段（将 old_text 替换为 new_text）",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "目标文件相对路径"},
                        "old_text": {"type": "string", "description": "文件中必须完全精确匹配的原文本"},
                        "new_text": {"type": "string", "description": "用于替换的新文本内容"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
                self._run_edit,
            ),
            (
                "glob",
                "在当前工作区按模式检索文件列表（支持 ** 递归匹配）",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "文件通配符模式，如 '**/*.py' 或 'tests/*'"}
                    },
                    "required": ["pattern"],
                },
                self._run_glob,
            ),
        ]

        for name, description, parameters, handler in builtin:
            self.register(name, description, parameters, handler)

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., str],
    ):
        """
        动态注册新工具（为后续 MCP / 自定义插件拓展预留）
        """
        # 封装为标准百炼/OpenAI Function Schema 结构
        spec = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._specs.append(spec)
        self._handlers[name] = handler

    def get_tools_spec(self) -> List[Dict[str, Any]]:
        """导出提供给大模型调用的标准工具协议声明列表"""
        return self._specs

    def register_plugin(self, plugin: ToolPlugin) -> None:
        """
        统一工具插件装配协议 (对齐 HookManager.register_plugin)：
        挂载遵循 ToolPlugin 协议契约规范的复合工具插件。
        """
        plugin.register_to(self)

    def fork(self, exclude: Optional[Any] = None) -> "ToolRegistry":
        """
        通用派生/复制工具注册表：
        继承当前工作区沙箱与已注册工具，支持在派生时按名称排除特定工具集合。
        :param exclude: 需要在派生副本中排除的工具名称集合 (如 {"subagent"})
        :return: 独立隔离的 ToolRegistry 副本
        """
        excluded_names = set(exclude or [])
        forked = ToolRegistry(workdir=self.workdir)
        for spec in self._specs:
            tool_name = spec["function"]["name"]
            if tool_name in excluded_names:
                continue
            if tool_name not in forked._handlers:
                forked.register(
                    name=tool_name,
                    description=spec["function"]["description"],
                    parameters=spec["function"]["parameters"],
                    handler=self._handlers[tool_name],
                )
        return forked

    # ------------------------------------------------------------------
    # 4. 防御性分发执行 (Defensive Dispatcher)
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


# 全局默认单例实例
default_registry = ToolRegistry()


# ----------------------------------------------------------------------
# 5. 模块独立冒烟自测
# ----------------------------------------------------------------------

def _smoke_test():
    import tempfile
    print("启动 tools/registry.py 冒烟自测...")

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ToolRegistry(workdir=Path(tmpdir))

        # 1. 检验协议规范 Schema
        specs = registry.get_tools_spec()
        assert len(specs) == 5
        assert specs[0]["type"] == "function"
        assert "parameters" in specs[0]["function"]
        print(f"1. 百炼/OpenAI 兼容 Schema 校验通过: {len(specs)} 个工具已就绪")

        # 2. 检验 write_file
        res_write = registry.execute("write_file", {"path": "sub/demo.txt", "content": "hello agent"})
        assert "Successfully wrote" in res_write
        print("2. write_file (自动创建父目录) 校验通过")

        # 3. 检验 read_file
        res_read = registry.execute("read_file", {"path": "sub/demo.txt"})
        assert res_read == "hello agent"
        print("3. read_file 校验通过")

        # 4. 检验 edit_file
        res_edit = registry.execute("edit_file", {"path": "sub/demo.txt", "old_text": "agent", "new_text": "world"})
        assert "Successfully edited" in res_edit
        assert registry.execute("read_file", {"path": "sub/demo.txt"}) == "hello world"
        print("4. edit_file 原子替换校验通过")

        # 5. 检验 glob
        res_glob = registry.execute("glob", {"pattern": "**/*.txt"})
        assert "sub/demo.txt" in res_glob
        print("5. glob 递归文件检索校验通过")

        # 6. 检验 bash 及其危险命令拦截
        res_bash = registry.execute("bash", {"command": "echo 'running bash'"})
        assert "running bash" in res_bash
        res_danger = registry.execute("bash", {"command": "rm -rf /"})
        assert "Dangerous command blocked" in res_danger
        print("6. bash 与危险命令拦截校验通过")

        # 7. 检验路径穿越沙箱拦截
        res_escape = registry.execute("read_file", {"path": "../../etc/passwd"})
        assert "安全违规：路径逃逸" in res_escape
        print("7. 路径防逃逸沙箱拦截校验通过")

        # 8. 检验大模型幻觉：不存在的工具与错误参数
        res_unknown = registry.execute("not_exist_tool", {})
        assert "Unknown tool 'not_exist_tool'" in res_unknown
        res_bad_args = registry.execute("read_file", {"wrong_arg": 123})
        assert "Invalid arguments passed" in res_bad_args
        print("8. 幻觉工具与畸变参数容错校验通过")

    print("\n✅ tools/registry.py 全部 8 项工业级特性冒烟测试通过！")


if __name__ == "__main__":
    _smoke_test()
