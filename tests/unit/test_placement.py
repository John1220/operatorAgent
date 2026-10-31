"""Stage 2 多卡放置的离线标定 (设计 §11 / §9.3 / §13.5)

五条硬纪律, 各一条方向检查:
  1. 同域串行, 异域并行 (未知域 → 保守串行)
  2. 同卡配对: 跨卡/跨窗加速比作废
  3. 竞争检测: 本模块产出证据(三态), 降级判决只在 measurement.assess
  4. BuildKey ≠ PartitionKey (编译缓存可共享, 测量缓存绝不能)
  5. 异构卡分区不隐式共享
"""
import unittest

from operator_agent.execute.placement import (LEASE_CLEAN, LEASE_CONTENDED, LEASE_UNOBSERVED,
                                      BuildKey, DeviceKey, DeviceLease, PartitionKey,
                                      build_key, can_measure_parallel,
                                      device_of_payload, lease_evidence, lease_waves,
                                      observed_lease, pairing_conflict, partition_of,
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


class TestCompetitionEvidence(unittest.TestCase):
    """§11-4: 本模块只产出**证据**; 降级判决在 `measurement.assess` (读同一个 util)"""

    def test_an_unobserved_load_is_not_a_clean_bill(self):
        """`external_load=None` (没测) ≠ `0.0` (测到没占用) —— 缺键不是 0 的老账"""
        _, state = lease_evidence(DeviceLease(_dev("u1", "d1")))
        self.assertEqual(LEASE_UNOBSERVED, state,
                         "没观测到占用时不许声称干净")

    def test_a_measured_zero_is_clean(self):
        text, state = lease_evidence(DeviceLease(_dev("u1", "d1"), exclusive=None,
                                                 external_load=0.0, source="t"))
        self.assertEqual(LEASE_CLEAN, state)
        self.assertIn("external_load=0.00", text)
        self.assertIn("exclusive 未观测", text, "三态要逐格写出来, 不是只报一个结论")

    def test_an_external_load_is_contended(self):
        """§11-4: 租不到独占卡时记录 external_load —— 诚实降级优于静默污染"""
        text, state = lease_evidence(DeviceLease(_dev("u1", "d1"), external_load=0.3))
        self.assertEqual(LEASE_CONTENDED, state)
        self.assertIn("external_load=0.30", text)

    def test_a_non_exclusive_lease_is_contended_even_with_a_zero_load(self):
        """独占性被观测为假时, 瞬时 util=0 不能翻案 (它只是「这一刻没人」)"""
        _, state = lease_evidence(DeviceLease(_dev("u1", "d1"), exclusive=False,
                                              external_load=0.0))
        self.assertEqual(LEASE_CONTENDED, state)

    def test_exclusive_is_never_claimed_by_default(self):
        """默认必须是 `None` —— 「没观测」不许写成「独占」(本仓最贵的那种假话)"""
        lease = DeviceLease(_dev("u1", "d1"))
        self.assertIsNone(lease.exclusive)
        self.assertIsNone(lease.external_load)


class TestTheObservedLease(unittest.TestCase):
    """喂给者: 从这次测量的载荷装出租约 (读的键与档位判决同一份)"""

    def test_the_util_comes_from_the_same_payload_the_tier_reads(self):
        lease = observed_lease({"gpu_util_pct": 37.0, "selected_gpu": 3,
                                "device": "cuda:3"}, source="L3")
        self.assertAlmostEqual(0.37, lease.external_load)
        self.assertEqual(3, lease.device.index)
        self.assertEqual("cuda:3", lease.device.node)
        self.assertEqual(LEASE_CONTENDED, lease_evidence(lease)[1])
        self.assertEqual("L3", lease.source)

    def test_a_missing_util_is_unobserved_not_zero(self):
        for payload in ({}, {"selected_gpu": 0}, {"gpu_util_pct": None},
                        {"gpu_util_pct": "12"}):
            lease = observed_lease(payload)
            self.assertIsNone(lease.external_load, payload)
            self.assertEqual(LEASE_UNOBSERVED, lease_evidence(lease)[1], payload)

    def test_an_index_is_never_dressed_up_as_a_uuid(self):
        """§11: 「uuid 最稳的锚, 别用 index」—— 只有索引时如实记空 uuid"""
        key = device_of_payload({"selected_gpu": 2, "device": "cuda:2"})
        self.assertEqual("", key.uuid)
        self.assertEqual(2, key.index)
        self.assertIn("index=2", key.label())
        # 读失败 (没有 selected_gpu) ⇒ index 记 -1, 不许编一个 0 出来
        self.assertEqual(-1, device_of_payload({"device": "cuda:0"}).index)

    def test_unknown_interference_domain_still_fails_closed(self):
        """租约的身份来自观测, 而干扰域今天观测不到 ⇒ 空串 ⇒ 保守串行"""
        key = device_of_payload({"selected_gpu": 1})
        self.assertEqual("", key.interference_domain)
        self.assertFalse(can_measure_parallel(key, _dev("u2", "d2")))

    def test_lease_waves_carries_leases_with_the_observed_loads(self):
        """设计的 `Placement.lease(contract, n)` 形状: 波次里装租约"""
        waves = lease_waves([_dev("a", "d1"), _dev("b", "d2")], n=2,
                            loads={"a": 0.5})
        self.assertEqual(2, len(waves))
        first = waves[0][0]
        self.assertIsInstance(first, DeviceLease)
        self.assertIsNone(first.exclusive, "不租卡 ⇒ 不许声称独占")
        self.assertAlmostEqual(0.5, first.external_load)
        # 没有 load 记录的那张卡是「未观测」, 不是 0
        b = [l for w in waves for l in w if l.device.uuid == "b"][0]
        self.assertIsNone(b.external_load)
        self.assertEqual(LEASE_UNOBSERVED, lease_evidence(b)[1])

    def test_the_lease_serialises_its_own_tri_state(self):
        d = observed_lease({"gpu_util_pct": 12.0}, source="L3").as_dict()
        self.assertIsNone(d["exclusive"])
        self.assertAlmostEqual(0.12, d["external_load"])
        self.assertEqual("L3", d["source"])
        self.assertEqual("", d["device"]["uuid"], "序列化也要如实带出「只有索引」")
        self.assertIn("index=-1", d["device_label"])


    def test_harness_peers_make_the_lease_contended_even_at_zero_util(self):
        """I68: util 是**时刻**采样 (对手还在 import 时读到 0), claim 声明的生命期
        覆盖对手整个进程 —— 所以 0% util + 1 个 harness 声明仍是 contended。"""
        lease = observed_lease({"gpu_util_pct": 0.0, "selected_gpu": 2,
                                "gpu_harness_peers": 1}, source="L3")
        self.assertEqual(1, lease.harness_peers)
        self.assertEqual(0.0, lease.external_load)
        text, state = lease_evidence(lease)
        self.assertEqual(LEASE_CONTENDED, state)
        self.assertIn("同卡 harness 进程 ×1", text)
        self.assertIn("external_load=0.00", text, "util 那一格照实说它的 0")

    def test_missing_peers_key_is_unobserved_not_zero(self):
        """旧 runner 不报这个键 ⇒ None (未观测), 序列化也带得出去; 缺 ≠ 0 ≠ 有对手。"""
        lease = observed_lease({"gpu_util_pct": 0.0})
        self.assertIsNone(lease.harness_peers)
        self.assertIsNone(lease.as_dict()["harness_peers"])
        self.assertEqual(LEASE_CLEAN, lease_evidence(lease)[1])


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
