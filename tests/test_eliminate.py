"""构象淘汰（eliminate）与 allowed 候选映射的必要测试。

期望值来源：
- tests/reference.py 的独立全枚举与 verify_domination（不导入 rotamer_select）；
- 手工构造的边界实例（并列、硬冲突、全 None 列、非法 allowed）。
"""

import unittest

from rotamer_select import (
    eliminate,
    solve,
    optimize,
    RotamerProblem,
    RotamerInputError,
    SolveStatus,
    INF,
)
from reference import brute_best, all_optimal, verify_domination
from datagen import generate_instance


def _allowed_of(result):
    return {p: list(c) for p, c in enumerate(result.remaining)}


class TestEliminationPreservesOptima(unittest.TestCase):
    """小实例枚举：所有原最优构象的每个 rotamer 都必须幸存。"""

    def test_all_optima_survive(self):
        checked = 0
        for n in range(2, 7):
            for seed in range(25):
                with self.subTest(n=n, seed=seed):
                    singles, pairs = generate_instance(
                        n, seed=seed, extra_pairs=(seed % 3)
                    )
                    res = eliminate(singles, pairs)
                    optima = all_optimal(singles, pairs)
                    if not optima:
                        continue  # 原问题无可行构象，另行覆盖
                    self.assertEqual(res.status, "ok")
                    for assign in optima:
                        for p, r in enumerate(assign):
                            self.assertIn(r, res.remaining[p])
                    checked += 1
        self.assertGreater(checked, 100)

    def test_reduced_solve_matches_brute_best(self):
        for n in range(2, 7):
            for seed in range(15):
                with self.subTest(n=n, seed=seed):
                    singles, pairs = generate_instance(n, seed=300 + seed)
                    res = eliminate(singles, pairs)
                    ba, be = brute_best(singles, pairs)
                    if be is None:
                        continue
                    self.assertEqual(res.status, "ok")
                    out = solve(singles, pairs, allowed=_allowed_of(res))
                    self.assertIs(out.status, SolveStatus.EXACT)
                    self.assertEqual(out.energy, be)
                    self.assertEqual(tuple(out.assignment), ba)


class TestCertificateIndependentCheck(unittest.TestCase):
    """每条淘汰证书都用独立参考逐条复核。"""

    def test_certificates_hold(self):
        total = 0
        for n in range(2, 7):
            for seed in range(20):
                singles, pairs = generate_instance(
                    n, seed=500 + seed, extra_pairs=(seed % 2)
                )
                res = eliminate(singles, pairs)
                for cert in res.certificates:
                    with self.subTest(n=n, seed=seed, cert=cert.eliminated,
                                      pos=cert.position):
                        if cert.reason == "self_conflict":
                            self.assertIsNone(singles[cert.position][cert.eliminated])
                            self.assertIsNone(cert.dominator)
                            self.assertEqual(cert.lower_bound_diff, INF)
                        else:
                            ok, worst = verify_domination(
                                singles, pairs, cert.position,
                                cert.eliminated, cert.dominator, cert.context,
                            )
                            self.assertTrue(ok, f"证书不成立: {cert}")
                            if worst is not None and                                     cert.lower_bound_diff != INF:
                                # 证书声明的下界不得超过独立算得的真实最小差
                                self.assertLessEqual(cert.lower_bound_diff, worst)
                            self.assertTrue(
                                cert.lower_bound_diff == INF
                                or cert.lower_bound_diff > 0
                            )
                        total += 1
        self.assertGreater(total, 50)

    def test_certificate_terms_cover_all_pairs(self):
        singles, pairs = generate_instance(5, seed=42, extra_pairs=2)
        res = eliminate(singles, pairs)
        pair_end = {}
        for i, j, _ in pairs:
            pair_end.setdefault(i, set()).add((i, j))
            pair_end.setdefault(j, set()).add((i, j))
        for cert in res.certificates:
            if cert.reason != "dominated":
                continue
            kinds = [t.kind for t in cert.terms]
            self.assertEqual(kinds[0], "single")
            cert_pairs = {t.pair for t in cert.terms if t.kind == "pair"}
            self.assertEqual(cert_pairs, pair_end.get(cert.position, set()))


class TestEqualEnergyNotEliminated(unittest.TestCase):
    """相等能量不淘汰：并列最优的两个候选都必须保留。"""

    def test_tie_kept(self):
        # 两个位置的候选两两完全等价（单体与对能逐格相同）。
        singles = [[2, 2], [1, 1]]
        pairs = [(0, 1, [[1, 1], [1, 1]])]
        res = eliminate(singles, pairs)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.remaining, ((0, 1), (0, 1)))
        self.assertEqual(res.eliminated_count, 0)
        optima = all_optimal(singles, pairs)
        self.assertEqual(len(optima), 4)
        # 部分等价：位置 0 并列，位置 1 的 r1 被合法支配不影响并列保留
        singles2 = [[2, 2], [1, 3]]
        pairs2 = [(0, 1, [[1, 4], [1, 4]])]
        res2 = eliminate(singles2, pairs2)
        self.assertEqual(res2.remaining[0], (0, 1))
        for assign in all_optimal(singles2, pairs2):
            for p, r in enumerate(assign):
                self.assertIn(r, res2.remaining[p])


class TestHardConflictArithmetic(unittest.TestCase):
    """硬冲突按不可行组合处理：无穷大相减不得产生 NaN 或误判。"""

    def test_both_none_contributes_zero(self):
        # 对 (0,1)：r0 与 r1 对 x=0 都是 None（同不可行，差 0），
        # 对 x=1 差为 3；单体差 1 => LB = 1+0 = 1 > 0，r0 被 r1 支配。
        singles = [[5, 4], [0, 0]]
        pairs = [(0, 1, [[None, 8], [None, 5]])]
        res = eliminate(singles, pairs)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.remaining[0], (1,))
        cert = res.certificates[0]
        self.assertEqual(cert.reason, "dominated")
        self.assertEqual(cert.lower_bound_diff, 1)

    def test_none_minus_finite_is_pos_inf(self):
        # r0 对所有邻居选择都硬冲突、r1 有限：r0 恒不可行 => 差 +inf。
        singles = [[0, 0], [0, 0]]
        pairs = [(0, 1, [[None, None], [3, 0]])]
        res = eliminate(singles, pairs)
        self.assertEqual(res.remaining[0], (1,))
        cert = res.certificates[0]
        self.assertEqual(cert.lower_bound_diff, INF)

    def test_finite_minus_none_blocks_domination(self):
        # r0 有限而 r1 在该组合硬冲突：差 -inf，r1 不能支配 r0。
        singles = [[9, 0], [0, 0]]
        pairs = [(0, 1, [[0, 0], [None, 0]])]
        res = eliminate(singles, pairs)
        self.assertEqual(res.remaining[0], (0, 1))
        self.assertEqual(res.eliminated_count, 0)

    def test_self_conflict_single_none(self):
        singles = [[None, 3], [0, 1]]
        res = eliminate(singles, [])
        self.assertEqual(res.remaining[0], (1,))
        cert = res.certificates[0]
        self.assertEqual(cert.reason, "self_conflict")
        self.assertIsNone(cert.dominator)


class TestInfeasibleEvidence(unittest.TestCase):
    """某残基没有可行候选时返回无解证据，而不是崩溃或误报。"""

    def test_all_none_column(self):
        singles = [[None, None], [0, 1]]
        res = eliminate(singles, [])
        self.assertEqual(res.status, "infeasible")
        self.assertIsNone(res.remaining)
        self.assertEqual(res.infeasible_position, 0)
        self.assertEqual(len(res.infeasible_evidence), 2)
        for cert in res.infeasible_evidence:
            self.assertEqual(cert.position, 0)
            self.assertEqual(cert.reason, "self_conflict")
        # 求解器独立确认：证明无解
        self.assertIs(solve(singles, []).status, SolveStatus.INFEASIBLE)

    def test_pair_forces_empty(self):
        # 位置对全硬冲突：严格规则下互不支配（同不可行差为 0）。
        singles = [[0, 0], [1, 2]]
        pairs = [(0, 1, [[None, None], [None, None]])]
        # 行全 None：两个候选对彼此差都是 0（同不可行），严格规则不淘汰，
        # 但每个候选自身在任何构象中都不可行——由求解器证明无解。
        res = eliminate(singles, pairs)
        self.assertEqual(res.status, "ok")  # 淘汰器不做组合推理，如实保留
        self.assertIs(solve(singles, pairs).status, SolveStatus.INFEASIBLE)


class TestAllowedMapping(unittest.TestCase):
    """旧求解器接收剩余候选映射并还原原始 ID。"""

    def test_allowed_restores_original_ids(self):
        singles = [[3, 1, 5], [2, 7], [0, 4]]
        pairs = [(0, 1, [[1, 2], [0, 9], [3, 1]]), (1, 2, [[5, 0], [1, 1]])]
        res = eliminate(singles, pairs)
        allowed = _allowed_of(res)
        out = solve(singles, pairs, allowed=allowed)
        full = solve(singles, pairs)
        self.assertEqual(out.assignment, full.assignment)
        self.assertEqual(out.energy, full.energy)
        # 构象中的 ID 都是原始 ID（在幸存集内）
        for p, r in enumerate(out.assignment):
            self.assertIn(r, res.remaining[p])

    def test_allowed_partial_mapping(self):
        singles = [[3, 1, 5], [2, 7]]
        pairs = [(0, 1, [[1, 2], [0, 9], [3, 1]])]
        # 只给位置 0 的映射；位置 1 默认全保留
        out = solve(singles, pairs, allowed={0: [1, 2]})
        self.assertIs(out.status, SolveStatus.EXACT)
        self.assertIn(out.assignment[0], (1, 2))
        self.assertEqual(out.energy, 1 + 2)  # r1 + 位置1 r0 + 对能 0

    def test_allowed_with_fixed_and_problem_reuse(self):
        singles = [[3, 1, 5], [2, 7]]
        pairs = [(0, 1, [[1, 2], [0, 9], [3, 1]])]
        prob = RotamerProblem(singles, pairs, allowed={0: [1, 2]})
        out = prob.solve(fixed={0: 2})
        self.assertEqual(out.assignment[0], 2)
        # 固定到已淘汰候选 -> 直接 INFEASIBLE
        dead = prob.solve(fixed={0: 0})
        self.assertIs(dead.status, SolveStatus.INFEASIBLE)
        # energy_of 仍按原始 ID
        self.assertEqual(prob.energy_of(out.assignment), out.energy)

    def test_optimize_with_allowed(self):
        singles, pairs = generate_instance(5, seed=17, extra_pairs=2)
        res = eliminate(singles, pairs)
        ana = optimize(singles, pairs, allowed=_allowed_of(res))
        full = optimize(singles, pairs)
        self.assertEqual(ana.assignment, full.assignment)
        self.assertEqual(ana.energy, full.energy)
        for p, col in enumerate(ana.fixed_costs):
            for entry in col:
                if entry is not None:
                    self.assertIn(entry.rotamer, res.remaining[p])

    def test_invalid_allowed_rejected(self):
        singles = [[1, 2], [3, 4]]
        bad = [
            {0: []},               # 空候选
            {0: [0, 0]},           # 重复
            {0: [0, 2]},           # 越界
            {2: [0]},              # 位置越界
            {0: [True]},           # bool 冒充 int
            {"0": [0]},            # 非 int 键
            [0, 1],                # 非 dict
            {0: "ab"},             # 非 list/tuple
        ]
        for allowed in bad:
            with self.subTest(allowed=allowed):
                with self.assertRaises(RotamerInputError):
                    solve(singles, [], allowed=allowed)

    def test_eliminate_invalid_input_rejected(self):
        with self.assertRaises(RotamerInputError):
            eliminate([[1.5, 2], [3, 4]], [])       # float 能量
        with self.assertRaises(RotamerInputError):
            eliminate([[1, 2], [3, 4]], [(0, 0, [[1, 2], [3, 4]])])  # i==j
        with self.assertRaises(RotamerInputError):
            eliminate([[True, 2], [3, 4]], [])      # bool 能量


if __name__ == "__main__":
    unittest.main()
