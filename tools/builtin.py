"""
triumph-agent 内置基础原子工具集 (Builtin Tools Plugin)
======================================================
核心设计思想（对标 learn-claude-code s02 & s11）：
1. 终极融合 bash：
   - 融合原生同步执行（120s 强制超时 + 50,000 字符截断 + 危险命令硬拦截）；
   - 依赖注入可选的 BackgroundTaskManager，支持 run_in_background 后台非阻塞托管；
2. 文件与沙箱操作（safe_path 边界检查）：
   - read_file (行数分页), write_file (自动建父目录), edit_file (原子替换), glob (递归检索)；
3. 遵循 ToolPlugin 契约协议，作为一个纯净的、高内聚的标准工具插件。
"""

from __future__ import annotations

import glob as g
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestration.job import JobManager
    from tools.registry import ToolRegistry


class BuiltinToolsPlugin:
    """
    内置原子工具插件 (符合 ToolPlugin 协议契约)
    提供原生融合的 bash 以及文件沙箱原子工具。
    """

    def __init__(
        self,
        workdir: Optional[Path] = None,
        job_manager: Optional[JobManager] = None,
    ):
        self.workdir = (workdir or Path.cwd()).resolve()
        self.job_manager = job_manager

    def register_to(self, registry: ToolRegistry) -> None:
        """统一向 ToolRegistry 挂载内置工具"""
        for name, description, parameters, handler in self._get_specs(registry):
            registry.register(name, description, parameters, handler)

    def _get_specs(
        self, registry: ToolRegistry
    ) -> List[Tuple[str, str, Dict[str, Any], Callable[..., Any]]]:
        """定义基础工具的标准规格与对应的物理 Handler"""

        def run_bash(command: str, run_in_background: bool = False) -> str:
            """在工作区安全执行终端命令（原生支持同步与后台托管执行）"""
            clean_cmd = command.strip()
            dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
            if any(d in clean_cmd for d in dangerous):
                return "Error: Dangerous command blocked by safety policy"

            # 1. 后台长作业非阻塞模式 (对标 s11)
            if run_in_background:
                if self.job_manager is not None:
                    try:
                        job = self.job_manager.start(clean_cmd)
                        return (
                            f"[Background job {job.job_id} started (PID: {job.pid})]\n"
                            f"- Command: {job.command}\n"
                            f"- Log file: {job.log_file}\n"
                            "The job is executing in background. You can use `check_job` to monitor it, "
                            "or wait for an automatic completion notification on later turns."
                        )
                    except Exception as e:
                        return f"Error starting background job: {e}"
                return "Error: Background job manager is not configured in this environment."

            # 2. 正常同步执行模式 (对标 s02)
            try:
                # 将当前运行 Agent 的 Python 解释器所在 bin 目录优先置顶于 PATH
                child_env = os.environ.copy()
                current_bin_dir = str(Path(sys.executable).parent)
                child_env["PATH"] = f"{current_bin_dir}:{child_env.get('PATH', '')}"

                r = subprocess.run(
                    clean_cmd,
                    shell=True,
                    cwd=registry.workdir,
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

        def run_read(path: str, limit: Optional[int] = None) -> str:
            """读取文件内容，支持行数限制与防逃逸"""
            try:
                target = registry.safe_path(path)
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

        def run_write(path: str, content: str) -> str:
            """安全写入文件，自动创建父级目录"""
            try:
                target = registry.safe_path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return f"Successfully wrote {len(content)} bytes to {path}"
            except Exception as e:
                return f"Error writing file: {e}"

        def run_edit(path: str, old_text: str, new_text: str) -> str:
            """精确替换文件中的特定代码块"""
            try:
                target = registry.safe_path(path)
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

        def run_glob(pattern: str) -> str:
            """在工作区内查找符合 glob 模式的文件"""
            try:
                matches = sorted({
                    match for match in g.glob(
                        pattern, root_dir=registry.workdir, recursive=True
                    )
                    if (registry.workdir / match).resolve().is_relative_to(registry.workdir)
                })
                shown = matches[:200]
                if len(matches) > 200:
                    shown.append("... (more matches omitted; please narrow down your pattern)")
                return "\n".join(shown) if shown else "(no matches found)"
            except Exception as e:
                return f"Error running glob: {e}"

        return [
            (
                "bash",
                "在工作区安全执行终端命令（支持 pip/pytest/git 等常用命令；支持 run_in_background 后台长任务模式）",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的 Shell 命令"},
                        "run_in_background": {
                            "type": "boolean",
                            "description": "是否在后台非阻塞异步执行长命令（适用于持续测试、服务启动、长脚本等）。若为 True 则立即返回任务 ID，任务执行完成后会自动通知。",
                        },
                    },
                    "required": ["command"],
                },
                run_bash,
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
                run_read,
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
                run_write,
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
                run_edit,
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
                run_glob,
            ),
        ]
