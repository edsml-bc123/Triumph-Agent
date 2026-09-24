"""
triumph-agent 终端交互增强与 Bracketed Paste 自动化测试
=====================================================
验证目标：
1. DECSET 2004 转义字符协议握手 (enable / disable)；
2. 非 TTY 环境平滑降级兼容；
3. 异步终端单行键盘输入即时返回；
4. 异步多行 Bracketed Paste 捕获、转义码剥离与物理回车确认闭环；
5. 后台事件（如 Cron 到期）中断等待机制；
6. 安全审计终端输入流脏数据清理防御 (tcflush)。
"""

import pytest
from unittest.mock import MagicMock, patch

from runtime.terminal import (
    DECSET_2004_DISABLE,
    DECSET_2004_ENABLE,
    PASTE_END,
    PASTE_START,
    async_read_terminal_input,
    disable_bracketed_paste,
    enable_bracketed_paste,
)
from security.permission import PermissionHook


def test_bracketed_paste_protocol_handshake():
    """验证 DECSET 2004 终端转义序列输出"""
    mock_stdout = MagicMock()
    mock_stdout.isatty.return_value = True

    with patch("sys.stdout", mock_stdout):
        enable_bracketed_paste()
        mock_stdout.write.assert_called_with(DECSET_2004_ENABLE)
        mock_stdout.flush.assert_called()

        disable_bracketed_paste()
        mock_stdout.write.assert_called_with(DECSET_2004_DISABLE)


@pytest.mark.asyncio
async def test_async_read_terminal_input_non_tty_fallback():
    """验证在非 TTY 管道/单测环境中无缝降级为标准 input"""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False

    with patch("sys.stdin", mock_stdin), patch("builtins.input", return_value="hello agent"):
        res = await async_read_terminal_input(">> ")
        assert res == "hello agent"


@pytest.mark.asyncio
async def test_async_read_terminal_input_single_line():
    """验证手动敲击单行时，无需二次确认，立即返回"""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdout = MagicMock()
    mock_stdout.isatty.return_value = True

    with patch("sys.stdin", mock_stdin), patch("sys.stdout", mock_stdout):
        with patch("runtime.terminal._has_pending_input", side_effect=[True, False]):
            with patch("builtins.input", return_value="single line prompt"):
                res = await async_read_terminal_input(">> ")
                assert res == "single line prompt"


@pytest.mark.asyncio
async def test_async_read_terminal_input_bracketed_paste_multiline():
    """
    验证模拟 Cmd+V 粘贴多行内容：
    - 捕获包含换行的整块文本；
    - 剥离转义码；
    - 等待用户回车确认后作为单一 prompt 返回。
    """
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdout = MagicMock()
    mock_stdout.isatty.return_value = True

    first_input = f"{PASTE_START}1. 任务一：创建数据库表"
    subsequent_lines = [
        "2. 任务二：编写登录 API\n",
        f"3. 任务三：运行测试验证{PASTE_END}\n",
    ]
    mock_stdin.readline.side_effect = subsequent_lines

    # has_pending_input 模拟：
    # 第 1 次（async 循环检测回车）：True
    # 第 2、3 次（收集剩余多行）：True, True
    # 第 4 次（多行读尽）：False
    pending_side_effects = [True, True, True, False]

    input_calls = []

    def fake_input(prompt=""):
        input_calls.append(prompt)
        if len(input_calls) == 1:
            return first_input
        # 第二次 input 是等待用户敲击 Enter 确认
        return ""

    with patch("sys.stdin", mock_stdin), patch("sys.stdout", mock_stdout):
        with patch("runtime.terminal._has_pending_input", side_effect=pending_side_effects):
            with patch("builtins.input", side_effect=fake_input):
                result = await async_read_terminal_input("triumph >> ")

                # 验证两次 input 被调用：一次读首行，一次等待 Enter 确认
                assert len(input_calls) == 2
                expected_lines = [
                    "1. 任务一：创建数据库表",
                    "2. 任务二：编写登录 API",
                    "3. 任务三：运行测试验证",
                ]
                assert result == "\n".join(expected_lines)
                assert PASTE_START not in result
                assert PASTE_END not in result


def test_permission_tcflush_defense(tmp_path):
    """验证安全审计提示在终端环境下触发 tcflush 排空残留输入"""
    hook = PermissionHook(workdir=tmp_path, interactive=True)

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True

    with patch("sys.stdin", mock_stdin), patch("termios.tcflush") as mock_tcflush:
        with patch("builtins.input", return_value="y"):
            res = hook.ask_human_confirmation("bash", {"command": "rm file"}, "高危删除")
            assert res is True
            mock_tcflush.assert_called_once()


@pytest.mark.asyncio
async def test_async_read_terminal_input_interrupted():
    """验证当检测到后台事件中断时，async_read_terminal_input 立即返回 None"""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdout = MagicMock()
    mock_stdout.isatty.return_value = True

    checks = [False, True]
    with patch("sys.stdin", mock_stdin), patch("sys.stdout", mock_stdout):
        with patch("runtime.terminal._has_pending_input", return_value=False):
            res = await async_read_terminal_input(
                prompt=">> ",
                interrupt_check=lambda: checks.pop(0) if checks else True,
                poll_interval=0.01,
            )
            assert res is None


@pytest.mark.asyncio
async def test_async_read_terminal_input_normal():
    """验证正常键盘按 Enter 输入被顺利读取"""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdout = MagicMock()
    mock_stdout.isatty.return_value = True

    with patch("sys.stdin", mock_stdin), patch("sys.stdout", mock_stdout):
        with patch("runtime.terminal._has_pending_input", side_effect=[True, False]):
            with patch("builtins.input", return_value="hello agent"):
                res = await async_read_terminal_input(
                    prompt=">> ",
                    interrupt_check=lambda: False,
                    poll_interval=0.01,
                )
                assert res == "hello agent"
