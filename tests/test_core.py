"""针对 rotamer_pack 核心的必要测试。

期望值由 tests/reference_bruteforce.py 的独立全枚举产生（该参考不导入
rotamer_pack）；``evaluate`` 往返一致性只作为补充断言。
"""

import math
import random
import unittest

from rotamer_pack import RotamerProblem, ProblemValidationError, minimize_energy
from rotamer_pack.core import MAX_POSITIONS, MIN_ROTAMERS, MAX_ROTAMERS

from reference_bruteforce import all_feasible, true_optimum


def random_problem(rng, max_n=6):
    """确定性生成一个合法问题描述（list 结构，含负数与 None）。"""
    n = rng.randint(1, max_n)
    k = [rng.randint(MIN_ROTAMERS, MAX_ROTAMERS) for _ in range(n)]

    def cell(lo, hi, none_p):
        if rng.random() < none_p:
            return None
        return rng.randint(lo, hi)

    singles = [
        [cell(-20, 20, 0.12) for _ in range(k[i])]
        for i in range(n)
    ]
    pairs = []
    for i in range(n - 1):
        if rng.random() < 0.8:
            j = i + 1
            mat = [
                [cell(-12, 12, 0.12) for _ in range(k[j])]
                for _ in range(k[i])
            ]
            pairs.append(((i, j), mat))
    # 偶尔加入非相邻对
    for i in range(n):
        for j in range(i + 2, n):
            if rng.random() < 0.2:
                mat = [
                    [cell(-8, 8, 0.10) for _ in range(k[j])]
                    for _ in range(k[i])
                ]
                pairs.append(((i, j), mat))
    return singles, pairs


class TestAgainstBruteForce(unittest.TestCase):
    """跨 60 个确定性随机实例，把完整报告与独立全枚举逐项对照。"""

    def test_random_instances(self):
        rng = random.Random(20260923)
        for case in range(60):
            with self.subTest(case=case):
                singles, pairs = random_problem(rng)
                feats = all_feasible(singles, pairs)
                rep = RotamerProblem(singles, pairs).solve()

                if not feats:
                    self.assertEqual(rep.status, "infeasible")
                    self.assertIsNone(rep.energy)
                    self.assertIsNone(rep.assignment)
                    self.assertIsNone(rep.lower_bound)
                    self.assertFalse(rep.energy_proven)
                    self.assertIs(rep.feasible, False)
                    continue

                self.assertEqual(rep.status, "optimal")
                self.assertTrue(rep.energy_proven)
                self.assertEqual(rep.lower_bound, feats[0][1])

                # 最优
                best_seq, best_e = feats[0]
                self.assertEqual(rep.energy, best_e)
                self.assertEqual(rep.assignment, best_seq)
                self.assertEqual(
                    RotamerProblem(singles, pairs).evaluate(rep.assignment),
                    best_e,
                )

                # 次优（不同构象；并列按字典序）
                if len(feats) >= 2:
                    s_seq, s_e = feats[1]
                    self.assertEqual(rep.second_energy, s_e)
                    self.assertEqual(rep.second_assignment, s_seq)
                    self.assertNotEqual(rep.second_assignment, rep.assignment)
                    self.assertGreaterEqual(rep.second_energy, rep.energy)
                else:
                    self.assertIsNone(rep.second_energy)
                    self.assertIsNone(rep.second_assignment)

                # 每位置固定代价增量
                self.assertIsNotNone(rep.costs)
                fixed_table = {}
                for i in range(len(singles)):
                    for r in range(len(singles[i])):
                        vals = [e for seq, e in feats if seq[i] == r]
                        fixed_table[(i, r)] = min(vals) if vals else None
                for i, col in enumerate(rep.costs):
                    for r, delta in enumerate(col):
                        exp = fixed_table[(i, r)]
                        if exp is None:
                            self.assertIsNone(delta, msg=f"costs[{i}][{r}]")
                        else:
                            self.assertEqual(delta, exp - best_e,
                                             msg=f"costs[{i}][{r}]")
                            self.assertGreaterEqual(delta, 0)
                self.assertEqual(rep.costs[i][best_seq[i]], 0)

    def test_tie_lexicographic_explicit(self):
        # 两个构象能量完全相等：必须取字典序最小，次优为下一个序列。
        singles = [[0, 0], [0, 0], [0, 0]]
        pairs = [((0, 1), [[0, 0], [0, 0]]),
                 ((1, 2), [[0, 0], [0, 0]])]
        rep = RotamerProblem(singles, pairs).solve()
        self.assertEqual(rep.energy, 0)
        self.assertEqual(rep.assignment, (0, 0, 0))
        self.assertEqual(rep.second_energy, 0)
        self.assertEqual(rep.second_assignment, (0, 0, 1))
        # 全部 8 个构象并列，costs 全部为 0
        for i in range(3):
            self.assertEqual(rep.costs[i], (0, 0))

    def test_negative_pair_energy_dominates(self):
        # 负成对能压倒单体偏好（单体贪心短视会选错，见 demo.py）。
        singles = [[0, -1], [0, 0], [0, 0]]
        pairs = [
            ((0, 1), [[-5, None], [None, 0]]),
            ((1, 2), [[-6, -6], [0, 0]]),
        ]
        rep = RotamerProblem(singles, pairs).solve()
        self.assertEqual(rep.assignment, (0, 0, 0))
        self.assertEqual(rep.energy, -11)
        # 固定 pos0=1 时最好只能 -1
        self.assertEqual(rep.costs[0][1], 10)

    def test_hard_clashes(self):
        # 某位置全部 rotamer 单体硬冲突 => 无解（已证明）
        prob = RotamerProblem([[None, None], [0, 0]])
        rep = prob.solve()
        self.assertEqual(rep.status, "infeasible")
        self.assertIs(rep.feasible, False)

        # 成对矩阵全 None => 无解
        pairs = [((0, 1), [[None, None], [None, None]])]
        rep = RotamerProblem([[0, 0], [0, 0]], pairs).solve()
        self.assertEqual(rep.status, "infeasible")

        # 部分 None：只剩唯一可行构象
        pairs = [((0, 1), [[3, None], [None, None]])]
        rep = RotamerProblem([[1, 100], [1, 100]], pairs).solve()
        self.assertEqual(rep.status, "optimal")
        self.assertEqual(rep.assignment, (0, 0))
        self.assertEqual(rep.energy, 5)
        self.assertIsNone(rep.second_assignment)
        # 其余固定槽位全部无解
        self.assertEqual(rep.costs[0], (0, None))
        self.assertEqual(rep.costs[1], (0, None))

    def test_evaluate_rejects_clash(self):
        prob = RotamerProblem([[0, None], [0, 0]],
                              [((0, 1), [[0, None], [None, 0]])])
        with self.assertRaises(ProblemValidationError):
            prob.evaluate([0, 1])
        with self.assertRaises(ProblemValidationError):
            prob.evaluate([1, 0])
        with self.assertRaises(ProblemValidationError):
            prob.evaluate([0, 2])
        with self.assertRaises(ProblemValidationError):
            prob.evaluate([0])


class TestBudget(unittest.TestCase):
    def setUp(self):
        rng = random.Random(424242)
        self.cases = [random_problem(rng) for _ in range(12)]

    def test_budget_zero_is_exhaustion_not_infeasible(self):
        singles, pairs = self.cases[0]
        rep = RotamerProblem(singles, pairs).solve(budget=0)
        self.assertEqual(rep.status, "budget_exhausted")
        self.assertFalse(rep.energy_proven)
        self.assertIsNone(rep.costs)
        optimum = true_optimum(singles, pairs)
        if optimum is not None:
            self.assertLessEqual(rep.lower_bound, optimum)

    def test_lower_bound_never_above_optimum(self):
        # 预算 1..40 下，全局下界必须始终 <= 真实最优（可采纳性实测）。
        for singles, pairs in self.cases:
            optimum = true_optimum(singles, pairs)
            if optimum is None:
                continue
            for b in range(0, 31):
                rep = RotamerProblem(singles, pairs).solve(budget=b)
                self.assertIn(rep.status,
                              ("budget_exhausted", "optimal"))
                self.assertIsNotNone(rep.lower_bound)
                self.assertLessEqual(
                    rep.lower_bound, optimum,
                    msg=f"budget={b}, LB 超过真实最优",
                )
                if rep.energy is not None:
                    # 已找到的能量必须是真实可行叶的能量
                    self.assertEqual(
                        RotamerProblem(singles, pairs).evaluate(rep.assignment),
                        rep.energy,
                    )

    def test_phase2_exhaustion_keeps_proven_optimum_but_no_costs(self):
        # 找到一个恰好让阶段一完成、代价增量阶段耗尽的预算。
        for singles, pairs in self.cases:
            if true_optimum(singles, pairs) is None:
                continue
            unlimited = RotamerProblem(singles, pairs).solve()
            hit = None
            for b in range(1, unlimited.nodes_expanded + 1):
                rep = RotamerProblem(singles, pairs).solve(budget=b)
                if (rep.status == "budget_exhausted"
                        and rep.energy_proven
                        and rep.costs is None):
                    hit = rep
                    break
            self.assertIsNotNone(hit, "应存在阶段二耗尽的预算区间")
            self.assertEqual(hit.energy, unlimited.energy)
            self.assertEqual(hit.assignment, unlimited.assignment)
            self.assertEqual(hit.lower_bound, unlimited.energy)
            self.assertFalse(hit.costs_complete)

    def test_large_budget_is_optimal(self):
        singles, pairs = self.cases[0]
        rep = RotamerProblem(singles, pairs).solve(budget=10_000_000)
        self.assertEqual(rep.status, "optimal")

    def test_node_level_bound_admissibility(self):
        """对每个可达状态，g + _lower_bound 不得超过该子树真实最优。"""
        for singles, pairs in self.cases:
            prob = RotamerProblem(singles, pairs)
            feats = all_feasible(singles, pairs)
            n = len(singles)

            def subtree(prefix):
                d = len(prefix)
                vals = [e for seq, e in feats if seq[:d] == tuple(prefix)]
                return min(vals) if vals else None

            def walk(depth, assigned, g):
                true_best = subtree(assigned[:depth])
                lb = prob._lower_bound(depth, assigned)
                if true_best is not None:
                    # 可行子树：下界必须有限且不高估
                    self.assertIsNotNone(lb)
                    self.assertLessEqual(g + lb, true_best)
                # 不可行子树允许 lb 为 None 或有限值（乐观下界不检测一切死局）
                if depth == n:
                    return
                for r, g2 in prob._children(depth, assigned, g):
                    assigned[depth] = r
                    if prob._forward_ok(depth + 1, assigned):
                        walk(depth + 1, assigned, g2)
                    assigned[depth] = -1

            walk(0, [-1] * n, 0)


class TestValidation(unittest.TestCase):
    def _base(self):
        return [[0, 1], [2, 3]], [((0, 1), [[0, 0], [0, 0]])]

    def test_matrix_dimensions(self):
        singles, _ = self._base()
        with self.assertRaises(ProblemValidationError):  # 行数不足
            RotamerProblem(singles, [((0, 1), [[0, 0]])])
        with self.assertRaises(ProblemValidationError):  # 列数不足
            RotamerProblem(singles, [((0, 1), [[0, 0], [0]])])
        with self.assertRaises(ProblemValidationError):  # 行不是序列
            RotamerProblem(singles, [((0, 1), [123, 456])])

    def test_keys_and_duplicates(self):
        singles, _ = self._base()
        good = [[0, 0], [0, 0]]
        with self.assertRaises(ProblemValidationError):  # i >= j
            RotamerProblem(good, [((1, 1), good)])
        with self.assertRaises(ProblemValidationError):  # 越界
            RotamerProblem(good, [((0, 2), [[0, 0], [0, 0]])])
        with self.assertRaises(ProblemValidationError):  # 重复对
            RotamerProblem(
                [[0, 0], [0, 0], [0, 0]],
                [((0, 1), [[0, 0], [0, 0]]),
                 ((0, 1), [[0, 0], [0, 0]])],
            )

    def test_counts_out_of_range(self):
        with self.assertRaises(ProblemValidationError):  # 13 位置
            RotamerProblem([[0, 0]] * (MAX_POSITIONS + 1))
        with self.assertRaises(ProblemValidationError):  # 0 位置
            RotamerProblem([])
        with self.assertRaises(ProblemValidationError):  # 每位置仅 1 rotamer
            RotamerProblem([[0], [0, 0]])
        with self.assertRaises(ProblemValidationError):  # 每位置 6 rotamer
            RotamerProblem([[0] * (MAX_ROTAMERS + 1), [0, 0]])

    def test_non_finite_and_float_and_bool(self):
        with self.assertRaises(ProblemValidationError):
            RotamerProblem([[0, math.nan], [0, 0]])
        with self.assertRaises(ProblemValidationError):
            RotamerProblem([[0, math.inf], [0, 0]])
        with self.assertRaises(ProblemValidationError):
            RotamerProblem([[0, -math.inf], [0, 0]])
        with self.assertRaises(ProblemValidationError):  # 浮点不接受
            RotamerProblem([[0, 1.5], [0, 0]])
        with self.assertRaises(ProblemValidationError):  # bool 不算整数
            RotamerProblem([[0, True], [0, 0]])
        with self.assertRaises(ProblemValidationError):
            RotamerProblem(
                [[0, 0], [0, 0]],
                [((0, 1), [[0, math.nan], [0, 0]])],
            )
        with self.assertRaises(ProblemValidationError):  # 下标 bool
            RotamerProblem([[0, 0], [0, 0]],
                           [((False, 1), [[0, 0], [0, 0]])])

    def test_structure_and_budget(self):
        with self.assertRaises(ProblemValidationError):
            RotamerProblem("not-a-sequence")
        with self.assertRaises(ProblemValidationError):
            RotamerProblem([[0, 0], 7])
        with self.assertRaises(ProblemValidationError):
            RotamerProblem([[0, 0], [0, 0]], pairs=123)
        with self.assertRaises(ProblemValidationError):
            RotamerProblem([[0, 0], [0, 0]]).solve(budget=-1)
        with self.assertRaises(ProblemValidationError):
            RotamerProblem([[0, 0], [0, 0]]).solve(budget=True)

    def test_error_message_is_locatable(self):
        try:
            RotamerProblem([[0, 0], [0, math.nan]])
        except ProblemValidationError as e:
            self.assertIn("singles[1][1]", str(e))
        else:
            self.fail("应当拒绝 NaN")
        try:
            RotamerProblem([[0, 0], [0, 0]],
                           [((0, 1), [[0, 0], [0, 9.0]])])
        except ProblemValidationError as e:
            self.assertIn("pairs[0] matrix[1][1]", str(e))
        else:
            self.fail("应当拒绝浮点矩阵项")

    def test_at_upper_boundary_12_positions(self):
        # 12 位置、每位置 2 rotamer 可构造并精确求解
        singles = [[0, 1] for _ in range(12)]
        pairs = [((i, i + 1), [[0, 0], [0, -2]]) for i in range(11)]
        rep = RotamerProblem(singles, pairs).solve()
        self.assertEqual(rep.status, "optimal")
        self.assertEqual(rep.assignment, tuple([1] * 12))
        self.assertEqual(rep.energy, 12 - 22)


class TestDeterminismAndStatelessness(unittest.TestCase):
    def test_repeated_solves_identical(self):
        rng = random.Random(7)
        singles, pairs = random_problem(rng)
        prob = RotamerProblem(singles, pairs)
        r1 = prob.solve()
        r2 = prob.solve()
        self.assertEqual(r1, r2)
        r3 = minimize_energy(singles, pairs)
        self.assertEqual(r1.assignment, r3.assignment)
        self.assertEqual(r1.energy, r3.energy)


if __name__ == "__main__":
    unittest.main()
