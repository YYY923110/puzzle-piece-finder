"""启动器的解释器判定。

这里只测两个纯函数。真正的重启动作是一句 subprocess.run，测它等于测标准库。
"""
from pathlib import Path

import main


class TestVenvPython:
    def test_finds_the_interpreter_in_this_project(self):
        found = main.venv_python(main.PROJECT_ROOT)
        assert found is not None
        assert found.exists()

    def test_returns_none_when_there_is_no_venv(self, tmp_path: Path):
        assert main.venv_python(tmp_path) is None


class TestNeedsRelaunch:
    def test_no_relaunch_when_already_running_the_venv_interpreter(self):
        target = main.venv_python(main.PROJECT_ROOT)
        assert target is not None
        assert main.needs_relaunch(str(target), target) is False

    def test_relaunch_when_running_some_other_interpreter(self):
        target = main.venv_python(main.PROJECT_ROOT)
        assert target is not None
        assert main.needs_relaunch(r"C:\Anaconda3\python.exe", target) is True

    def test_a_differently_spelled_path_to_the_same_file_is_not_a_relaunch(self):
        """Windows 上同一个文件有很多种写法，认错了会无限重启。

        VS Code 给出的 sys.executable 可能是 'd:\\...'，而我们算出来的是
        'D:\\...'。naive 字符串比较会判定「不是同一个解释器」→ 重启 →
        子进程做同样的判断 → 再重启，一路 fork 到系统卡死。
        """
        target = main.venv_python(main.PROJECT_ROOT)
        assert target is not None
        weird = str(target).replace("\\", "/")
        weird = weird[0].swapcase() + weird[1:]
        assert main.needs_relaunch(weird, target) is False
