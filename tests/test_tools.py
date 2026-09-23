"""
triumph-agent 工具协议与沙箱执行器 (ToolRegistry) 自动化测试
=============================================================
验证目标：
1. ToolRegistry 纯白板容器特性（默认无任何暗中挂载）；
2. safe_path 沙箱路径防逃逸；
3. read_file / write_file / edit_file 文件操作精度与边界；
4. glob 模式扫描；
5. _run_bash 原生同步执行与输出截断；
6. ToolRegistry.execute 容错分发与 OpenAI Schema 转换。
"""

import tempfile
from pathlib import Path
import pytest

from tools import BuiltinTool, ToolRegistry


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        yield workdir


@pytest.fixture
def registry(temp_workspace):
    """标准测试夹具：预装配 BuiltinTool 的纯净注册表"""
    reg = ToolRegistry(workdir=temp_workspace)
    reg.register_plugin(BuiltinTool(workdir=temp_workspace))
    return reg


def test_tool_registry_pure_whiteboard(temp_workspace):
    """验证 ToolRegistry 初始化为一个 100% 纯净的空白容器"""
    reg = ToolRegistry(workdir=temp_workspace)
    assert len(reg.get_tools_spec()) == 0
    assert len(reg._handlers) == 0


@pytest.mark.asyncio
async def test_tools_file_lifecycle(temp_workspace, registry):
    # 1. 写入文件
    write_res = await registry.execute("write_file", {"path": "sub/test.txt", "content": "Line 1\nLine 2\nLine 3"})
    assert "Successfully wrote" in write_res
    assert (temp_workspace / "sub/test.txt").exists()

    # 2. 读取文件
    read_res = await registry.execute("read_file", {"path": "sub/test.txt"})
    assert "Line 1\nLine 2\nLine 3" in read_res

    # 读取限制行数
    limit_res = await registry.execute("read_file", {"path": "sub/test.txt", "limit": 2})
    assert "Line 1\nLine 2" in limit_res
    assert "more lines" in limit_res

    # 3. 编辑文件
    edit_res = await registry.execute("edit_file", {"path": "sub/test.txt", "old_text": "Line 2", "new_text": "Modified"})
    assert "Successfully edited" in edit_res
    new_content = (temp_workspace / "sub/test.txt").read_text(encoding="utf-8")
    assert "Line 1\nModified\nLine 3" == new_content

    # 编辑不存在的内容应返回明确错误
    edit_fail = await registry.execute("edit_file", {"path": "sub/test.txt", "old_text": "NonExist", "new_text": "X"})
    assert "Error:" in edit_fail


@pytest.mark.asyncio
async def test_tools_safe_path_sandbox_escape(registry):
    # 试图逃逸出沙箱
    escape_res = await registry.execute("read_file", {"path": "../../etc/passwd"})
    assert "Error reading file:" in escape_res
    assert "安全违规" in escape_res or "沙箱边界" in escape_res


@pytest.mark.asyncio
async def test_tools_glob(temp_workspace, registry):
    (temp_workspace / "a.py").write_text("# a", encoding="utf-8")
    (temp_workspace / "b.py").write_text("# b", encoding="utf-8")
    (temp_workspace / "c.txt").write_text("text", encoding="utf-8")

    glob_res = await registry.execute("glob", {"pattern": "*.py"})
    assert "a.py" in glob_res
    assert "b.py" in glob_res
    assert "c.txt" not in glob_res


@pytest.mark.asyncio
async def test_tools_bash_execution(registry):
    # 正常命令
    res_echo = await registry.execute("bash", {"command": "echo 'Hello Bash'"})
    assert "Hello Bash" in res_echo

    # 包含多余 kwargs 的防御性分发测试
    res_defensive = await registry.execute("bash", {"command": "echo 'Safe'", "redundant_arg": 123})
    assert "Safe" in res_defensive


def test_tools_spec_generation(registry):
    specs = registry.get_tools_spec()
    assert isinstance(specs, list)
    assert len(specs) == 5

    names = [s["function"]["name"] for s in specs]
    assert "bash" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "edit_file" in names
    assert "glob" in names
