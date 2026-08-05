"""一键启动：在 VS Code 里对着这个文件点运行就行，不用敲命令。

它只做一件事——把服务起起来，然后打印手机能打开的地址。建索引和查询
都在网页上做，命令行那三条（`puzzlefind index/query/stats`）留给调参。

存在的理由是**依赖装在项目自带的 `.venv` 里，没有装进 base Anaconda**
（PaddleOCR 会拉进 opencv-contrib-python，装进 base 会顶掉原有的 headless
OpenCV 5.0）。而 VS Code 的运行按钮用的是它自己选中的那个解释器，多半
不是 `.venv` 里那个，直接跑会以 `ModuleNotFoundError: paddleocr` 收场。
所以这个文件先把自己切到对的解释器上，再干正事。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 默认端口。不用 8000：本机 8000 已被另一个 python 进程占着，而 Windows
# 允许两个进程绑同一端口且不报错，请求会随机落到其中一个——这种故障
# 看起来像「代码没生效」，极难排查。被占时 find_free_port 会往上顺延。
PORT = 8791


def venv_python(project_root: Path) -> Path | None:
    """项目 `.venv` 里的解释器；没建过 venv 时返回 None。"""
    candidate = project_root / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else None


def needs_relaunch(current: str, target: Path) -> bool:
    """当前解释器不是 target，需要换一个重跑吗？

    用 `os.path.samefile` 而不是比较字符串：Windows 上同一个文件有很多种
    写法（盘符大小写、正反斜杠、8.3 短路径），而这个判断错了不是小事——
    判成「不是同一个」就会重启，子进程做同样的判断又重启，一路 fork 下去。
    samefile 比的是文件本身，不是路径的拼法。

    路径压根打不开时返回 True：宁可多切一次解释器，也不要在错的解释器上
    往下走然后炸在 import。
    """
    try:
        return not os.path.samefile(current, target)
    except OSError:
        return True


def main() -> int:
    target = venv_python(PROJECT_ROOT)
    if target is None:
        print("没找到 .venv。先建好虚拟环境再跑这个文件：\n")
        print(f"  cd {PROJECT_ROOT}")
        print("  python -m venv .venv")
        print('  .\\.venv\\Scripts\\python.exe -m pip install -e ".[dev]"\n')
        return 2

    if needs_relaunch(sys.executable, target):
        # flush 是必须的：stdout 不是终端时（重定向、被别的程序捕获）Python
        # 会块缓冲，这句话会一直压在缓冲区里，直到服务停掉才吐出来——而它
        # 恰恰是「为什么愣了几秒还没反应」的答案。
        print(f"当前解释器没装依赖，切到 {target}\n", flush=True)
        # 用 subprocess 而不是 os.execv：execv 在 Windows 上会让父 shell
        # 以为命令已经结束、抢先打出提示符，服务明明在跑却像是崩了。
        return subprocess.run([str(target), str(Path(__file__).resolve())]).returncode

    from puzzlefind import server

    port = server.find_free_port(PORT)
    if port != PORT:
        print(f"{PORT} 已经有人在听，改用 {port}。", flush=True)
    server.run(port=port)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C 是正常的停止方式，不该甩一屏 traceback
        print("\n已停止。")
