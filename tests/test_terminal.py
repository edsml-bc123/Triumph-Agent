"""
triumph-agent 终端交互增强与 Bracketed Paste 自动化测试
=====================================================
验证目标：
1. DECSET 2004 转义字符协议握手 (enable / disable)；
2. 非 TTY 环境平滑降级兼容；
3. 单行键盘输入即时返回；
4. 多行 Bracketed Paste 捕获、转义码剥离与物理回车确认闭环；
5. 安全审计终端输入流脏数据清理防御 (tcflush)。
"""

from unittest.mock import MagicMock, patch

from runtime.terminal import (
    DECSET_2004_DISABLE,
    DECSET_2004_ENABLE,
    PASTE_END,
    PASTE_START,
    disable_bracketed_paste,
    enable_bracketed_paste,
    read_terminal_input,
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


def test_read_terminal_input_non_tty_fallback():
    """验证在非 TTY 管道/单测环境中无缝降级为标准 input"""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False

    with patch("sys.stdin", mock_stdin), patch("builtins.input", return_value="hello agent"):
        res = read_terminal_input(">> ")
        assert res == "hello agent"


def test_read_terminal_input_single_line():
    """验证手动敲击单行时，无需二次确认，立即返回"""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdout = MagicMock()
    mock_stdout.isatty.return_value = True

    with patch("sys.stdin", mock_stdin), patch("sys.stdout", mock_stdout):
        with patch("runtime.terminal._has_pending_input", return_value=False):
            with patch("builtins.input", return_value="single line prompt"):
                res = read_terminal_input(">> ")
                assert res == "single line prompt"


def test_read_terminal_input_bracketed_paste_multiline():
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

    # 模拟剪贴板灌入：第一行带 PASTE_START，随后的行带剩余内容与 PASTE_END
    first_input = f"{PASTE_START}1. 任务一：创建数据库表"
    subsequent_lines = [
        "2. 任务二：编写登录 API\n",
        f"3. 任务三：运行测试验证{PASTE_END}\n",
    ]
    mock_stdin.readline.side_effect = subsequent_lines

    # has_pending_input 模拟：前两次有待读行，第三次已读完
    pending_side_effects = [True, True, False]

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
                result = read_terminal_input("triumph >> ")

                # 验证两次 input 被调用：一次读首行，一次等待 Enter 确认
                assert len(input_calls) == 2
                # 验证返回的内容是完整合并的多行整体
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
