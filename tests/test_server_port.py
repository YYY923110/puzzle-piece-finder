"""空闲端口挑选。

判据必须是「有没有人在听」而不是「能不能绑上」：Windows 允许两个进程
绑同一个端口且不报错，请求随机落到其中一个（README 记着本机 8000 就是
这么被悄悄抢走的）。所以 bind 探测恰好测不出这个坑，只有 connect 能。
"""
import socket
from contextlib import contextmanager

import pytest

from puzzlefind import server


@contextmanager
def serving_port():
    """占住一个端口并真的监听它，退出时释放。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        yield sock.getsockname()[1]
    finally:
        sock.close()


def free_port() -> int:
    """一个当下没人占的端口号。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestFindFreePort:
    def test_returns_the_preferred_port_when_nobody_is_listening(self):
        port = free_port()
        assert server.find_free_port(port) == port

    def test_steps_past_a_port_that_is_already_serving(self):
        with serving_port() as taken:
            picked = server.find_free_port(taken, attempts=8)
        assert picked != taken
        assert taken < picked < taken + 8

    def test_raises_when_every_candidate_is_taken(self):
        with serving_port() as taken:
            with pytest.raises(RuntimeError):
                server.find_free_port(taken, attempts=1)
