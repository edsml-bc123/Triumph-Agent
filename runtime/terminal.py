"""
triumph-agent 终端交互增强与括号粘贴模式 (Bracketed Paste Mode)
==============================================================
对齐 DECSET 2004 国际标准终端协议与现代 CLI 人机交互体验：
1. 解决终端多行复制粘贴时换行符被误当成多次提交的顽疾；
2. 捕获剪贴板整块多行输入，在用户手动敲击物理 Enter 回车键确认前，绝不抢跑执行；
3. 严格保护 sys.stdin 输入流干净，杜绝后续安全审计被未消费的残留输入污染；
4. 保证在非 TTY 环境（自动化测试/管道重定向）下无缝平滑降级。
"""

import atexit
import select
import sys

# DECSET 2004 标准控制转义序列
DECSET_2004_ENABLE = "\x1b[?2004h"
DECSET_2004_DISABLE = "\x1b[?2004l"
PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"

_bracketed_paste_enabled = False


def enable_bracketed_paste() -> None:
    """向终端发送转义码，开启括号粘贴模式 (DECSET 2004)"""
    global _bracketed_paste_enabled
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        try:
            sys.stdout.write(DECSET_2004_ENABLE)
            sys.stdout.flush()
            _bracketed_paste_enabled = True
        except Exception:
            pass


def disable_bracketed_paste() -> None:
    """向终端发送转义码，关闭括号粘贴模式，还原终端原始状态"""
    global _bracketed_paste_enabled
    if _bracketed_paste_enabled and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        try:
            sys.stdout.write(DECSET_2004_DISABLE)
            sys.stdout.flush()
            _bracketed_paste_enabled = False
        except Exception:
            pass


# 注册退出钩子，确保进程意外中断或退出时一定会恢复终端模式
atexit.register(disable_bracketed_paste)


def _has_pending_input(timeout_seconds: float = 0.03) -> bool:
    """非阻塞检测 sys.stdin 是否仍有未读数据 (适用于类 Unix/macOS)"""
    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        return False
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
        return bool(rlist)
    except Exception:
        return False


def read_terminal_input(prompt: str = "\ntriumph >> ") -> str:
    """
    智能终端输入读取器：
    - 手动打字敲回车：立即返回，保持 0 延迟原生交互；
    - 剪贴板多行粘贴：完整截获所有行，回显并等待用户按下真实键盘 Enter 确认后才返回；
    - 非 TTY 环境：直接回退到标准 input()。
    """
    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        # 非交互终端 (如单测/重定向管道)，直接标准读取
        return input(prompt).strip()

    # 确保当前终端已开启 Bracketed Paste
    enable_bracketed_paste()

    # 1. 读出第一行 (如果是手动输入，用户按 Enter 即返回；若是粘贴，第一行携带 PASTE_START)
    line = input(prompt)

    # 检查是否为 Bracketed Paste 或存在未消费的瞬时突发粘贴残留
    is_bracketed = PASTE_START in line
    has_burst = False if is_bracketed else _has_pending_input(timeout_seconds=0.04)

    if not is_bracketed and not has_burst:
        # 纯键盘单行手动输入，直接返回
        return line.strip()

    # 2. 属于多行粘贴行为：循环收集 sys.stdin 中所有剩余的行
    lines = [line]
    end_detected = PASTE_END in line

    while not end_detected and _has_pending_input(timeout_seconds=0.05):
        next_line = sys.stdin.readline()
        if not next_line:
            break
        # 移除行末换行符以便统一处理
        cleaned = next_line.rstrip("\r\n")
        lines.append(cleaned)
        if PASTE_END in cleaned:
            end_detected = True

    # 3. 拼接并清洗 Bracketed Paste 转义标记
    raw_content = "\n".join(lines)
    clean_content = (
        raw_content.replace(PASTE_START, "")
        .replace(PASTE_END, "")
        .strip()
    )

    if not clean_content:
        return ""

    # 4. 如果内容包含多行，执行人机确认门禁：绝不自动抢跑！
    content_lines = clean_content.splitlines()
    if len(content_lines) > 1:
        # 回显后续行（终端第一行已被 input 回显，这里清晰打印完整视图）
        print("-" * 50)
        print(f"[已捕获剪贴板多行内容: 共 {len(content_lines)} 行]")
        for idx, cl in enumerate(content_lines, start=1):
            print(f"  {idx:2d} | {cl}")
        print("-" * 50)
        print(">>> 请确认上述提示词无误，按 [Enter 回车键] 提交执行 (或按 Ctrl+C 取消)...", end="", flush=True)

        # 阻塞等待用户真实的物理按键确认
        try:
            input()
        except KeyboardInterrupt:
            print("\n[已取消本次多行执行]")
            return ""

    return clean_content
