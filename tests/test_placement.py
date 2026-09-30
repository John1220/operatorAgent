"""Stage 2 多卡放置的离线标定 (设计 §11 / §9.3 / §13.5)

五条硬纪律, 各一条方向检查:
  1. 同域串行, 异域并行 (未知域 → 保守串行)
  2. 同卡配对: 跨卡/跨窗加速比作废
  3. 竞争检测 → DEGRADED 而不是错误通过
  4. BuildKey ≠ PartitionKey (编译缓存可共享, 测量缓存绝不能)
  5. 异构卡分区不隐式共享
"""
import unittest

from operator_agent.placement import (BuildKey, DeviceKey, DeviceLease,
                                      PartitionKey, build_key, can_measure_parallel,
                                      lease_notes, pairing_conflict, partition_of,
                                      plan_leases, same_partition)


def _dev(uuid, domain="", arch="sm90"):
    return DeviceKey(uuid=uuid, node="n1", arch=arch, interference_domain=domain)


class TestInterferenceDomain(unittest.TestCase):
    def test_same_domain_cannot_measure_in_parallel(self):
        a, b = _dev("u1", "l2-a"), _dev("u2", "l2-a")
        self.assertFalse(can_measure_parallel(a, b), "同域并行会互相污染 cv")

    def test_different_domain_can_parallel(self):
        a, b = _dev("u1", "l2-a"), _dev("u2", "l2-b")
        self.assertTrue(can_measure_parallel(a, b))

    def test_unknown_domain_fails_closed_to_serial(self):
        """§1.3 歧义失败关闭: 不知道是不是同域就当同域"""
        a, b = _dev("u1", ""), _dev("u2", "l2-b")
        self.assertFalse(can_measure_parallel(a, b))
        self.assertFalse(can_measure_parallel(_dev("u1", ""), _dev("u2", "")))

    def test_plan_leases_puts_same_domain_in_one_wave(self):
        devs = [_dev("u1", "l2-a"), _dev("u2", "l2-a"),
                _dev("u3", "l2-b"), _dev("u4", "l2-b")]
        waves = plan_leases(devs, n=4)
        for wave in waves:
            domains = [d.interference_domain for d in wave]
            self.assertEqual(len(domains), len(set(domains)),
                             f"同域进了同一波并行: {wave}")

    def test_plan_direction_more_domains_more_parallelism(self):
        two_domains = plan_leases([_dev("a", "d1"), _dev("b", "d2")], n=2)
        one_domain = plan_leases([_dev("a", "d1"), _dev("b", "d1")], n=2)
        self.assertGreaterEqual(max(len(w) for w in two_domains),
                                max(len(w) for w in one_domain))


class TestSameDevicePairing(unittest.TestCase):
    def test_different_devices_refused(self):
        why = pairing_conflict("gpu-0", "gpu-1")
        self.assertIn("同一张卡", why)

    def test_same_device_different_window_refused(self):
        why = pairing_conflict("gpu-0", "gpu-0", "w1", "w2")
        self.assertIn("时间窗", why)

    def test_same_device_same_window_passes(self):
        self.assertEqual("", pairing_conflict("gpu-0", "gpu-0", "w1", "w1"))

    def test_direction_control_unity_not_refused(self):
        self.assertEqual("", pairing_conflict("a", "a"))


class TestCompetitionDetection(unittest.TestCase):
    def test_exclusive_clean_lease_is_not_degraded(self):
        ev, deg = lease_notes(DeviceLease(_dev("u1", "d1"), exclusive=True,
                                          external_load=0.0))
        self.assertFalse(deg)
        self.assertIn("exclusive", ev)

    def test_external_load_degrades_not_fails(self):
        """§11-4: 诚实降级优于静默污染 —— 不是 error 是 DEGRADED"""
        ev, deg = lease_notes(DeviceLease(_dev("u1", "d1"), exclusive=True,
                                          external_load=0.3))
        self.assertTrue(deg)
        self.assertIn("external_load", ev)

    def test_non_exclusive_degrades(self):
        ev, deg = lease_notes(DeviceLease(_dev("u1", "d1"), exclusive=False))
        self.assertTrue(deg)


class TestBuildKeyIsNotPartitionKey(unittest.TestCase):
    def test_same_source_different_arch_different_build_key(self):
        """§9.3: 编译产物按 arch 共享, 测量结果绝不共享"""
        a = build_key("triton", "sm90", "abc")
        b = build_key("triton", "sm80", "abc")
        self.assertNotEqual(a, b)

    def test_partition_includes_driver_runtime(self):
        p1 = partition_of(_dev("u1", "d1"), driver="550.54", runtime="cu12")
        p2 = partition_of(_dev("u1", "d1"), driver="550.90", runtime="cu12")
        self.assertNotEqual(p1.as_tuple(), p2.as_tuple(),
                            "驱动变了分区没变 —— 跨漂移记忆会被误用")

    def test_build_and_partition_are_different_types(self):
        self.assertNotIsInstance(build_key("t", "sm90", "h"), PartitionKey)
        self.assertNotIsInstance(partition_of(_dev("u1")), BuildKey)

    def test_heterogeneous_arch_not_same_partition(self):
        p1 = partition_of(_dev("u1", "d1", arch="sm90"))
        p2 = partition_of(_dev("u1", "d1", arch="sm80"))
        self.assertFalse(same_partition(p1, p2))


if __name__ == "__main__":
    unittest.main()
