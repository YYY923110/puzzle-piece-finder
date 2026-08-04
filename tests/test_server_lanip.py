"""局域网 IP 选取。

单独一个文件是因为这跟 HTTP 无关，纯粹是网络地址挑选逻辑。
"""
from puzzlefind import server


class TestPickLanIp:
    def test_prefers_a_real_private_lan_address(self):
        # 198.18.x 是 RFC2544 基准测试段，Clash/Mihomo 这类代理的 TUN
        # 网卡常占用它。手机连不上这个地址。
        assert server._pick_lan_ip(["198.18.0.0", "192.168.2.119"]) == "192.168.2.119"

    def test_ranks_192_168_above_other_private_ranges(self):
        """家用路由器几乎都发 192.168.x，优先它命中率最高。"""
        picked = server._pick_lan_ip(["172.22.48.1", "10.0.0.5", "192.168.1.7"])
        assert picked == "192.168.1.7"

    def test_falls_back_to_ten_dot_when_no_192(self):
        assert server._pick_lan_ip(["172.22.48.1", "10.0.0.5"]) == "10.0.0.5"

    def test_skips_loopback_and_link_local(self):
        assert server._pick_lan_ip(["127.0.0.1", "169.254.9.9", "10.1.2.3"]) == "10.1.2.3"

    def test_returns_loopback_when_nothing_usable(self):
        assert server._pick_lan_ip(["198.18.0.0", "127.0.0.1"]) == "127.0.0.1"

    def test_empty_candidates_yield_loopback(self):
        assert server._pick_lan_ip([]) == "127.0.0.1"

    def test_wsl_virtual_adapter_loses_to_wlan(self):
        """本机实测的真实情形：WSL 的 172.22.48.1 和 WLAN 的 192.168.2.119 并存。"""
        assert server._pick_lan_ip(["172.22.48.1", "192.168.2.119"]) == "192.168.2.119"

    def test_172_outside_the_private_block_is_rejected(self):
        # 172.32.x 不在 172.16–31 私有段里，是公网地址
        assert server._pick_lan_ip(["172.32.0.1"]) == "127.0.0.1"


class TestLocalIpv4Addresses:
    def test_returns_parseable_ipv4_strings(self):
        import ipaddress

        for address in server._local_ipv4_addresses():
            ipaddress.IPv4Address(address)  # 非法地址会抛异常

    def test_finds_at_least_one_address(self):
        assert server._local_ipv4_addresses()
