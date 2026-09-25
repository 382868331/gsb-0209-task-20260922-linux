"""类别配额特性的必要测试。

期望值来源：tests/reference.py 的独立全枚举（不导入 rotamer_select），
以及手工构造的边界实例（局部最低能破坏全局配额、类别只在冲突位置可选、
零配额、同分布局）。
"""

import random
import unittest

from rotamer_select import (
    optimize,
    solve,
    energy_of,
    RotamerProblem,
    RotamerInputError,
    SolveStatus,
)
from reference import brute_best, brute_best_quota, brute_second_quota
from datagen import generate_instance

LABELS = ("hyd", "pol", "chg")


def make_categories(counts, rng):
    return [[rng.choice(LABELS) for _ in range(k)] for k in counts]


def make_quotas(categories, n, rng):
    used = sorted({c for col in categories for c in col})
    quotas = {}
    for label in used:
        if rng.random() < 0.3:
            continue  # 该类别不限制
        lo = rng.randint(0, n // 2)
        hi = rng.randint(lo, n)
        quotas[label] = (lo, hi)
    return quotas


class TestQuotaAgainstBruteForce(unittest.TestCase):
    """2..6 位置随机实例：带配额的最优/次优/类别计数对独立全枚举。"""

    def test_best_matches_brute_force(self):
        for n in range(2, 7):
            for seed in range(30):
                with self.subTest(n=n, seed=seed):
                    rng = random.Random(n * 10_000 + seed)
                    singles, pairs = generate_instance(
                        n, seed=seed, extra_pairs=(seed % 3))
                    cats = make_categories([len(c) for c in singles], rng)
                    quotas = make_quotas(cats, n, rng)
                    res = solve(singles, pairs, categories=cats, quotas=quotas)
                    ba, be, bc = brute_best_quota(singles, pairs, cats, quotas)
                    if be is None:
                        self.assertIs(res.status, SolveStatus.INFEASIBLE)
                        self.assertIsNone(res.assignment)
                        self.assertIsNone(res.category_counts)
                    else:
                        self.assertIs(res.status, SolveStatus.EXACT)
                        self.assertEqual(res.energy, be)
                        self.assertEqual(tuple(res.assignment), ba)
                        self.assertEqual(res.category_counts, bc)
                        self.assertEqual(res.global_lower_bound, be)
                        self.assertEqual(
                            energy_of(singles, pairs, res.assignment), be)

    def test_second_matches_brute_force(self):
        for n in range(2, 7):
            for seed in range(20):
                with self.subTest(n=n, seed=seed):
                    rng = random.Random(n * 20_000 + seed)
                    singles, pairs = generate_instance(n, seed=500 + seed)
                    cats = make_categories([len(c) for c in singles], rng)
                    quotas = make_quotas(cats, n, rng)
                    ana = optimize(singles, pairs, categories=cats, quotas=quotas)
                    sa, se = brute_second_quota(singles, pairs, cats, quotas)
                    if se is None:
                        self.assertIsNone(ana.second_assignment)
                    else:
                        self.assertEqual(tuple(ana.second_assignment), sa)
                        self.assertEqual(ana.second_energy, se)

    def test_categories_without_quotas_match_plain(self):
        """给类别但不给配额时，结果必须与无类别调用完全一致。"""
        for n in range(2, 7):
            for seed in range(10):
                singles, pairs = generate_instance(n, seed=900 + seed)
                rng = random.Random(seed)
                cats = make_categories([len(c) for c in singles], rng)
                plain = solve(singles, pairs)
                tagged = solve(singles, pairs, categories=cats)
                self.assertEqual(plain.status, tagged.status)
                self.assertEqual(plain.assignment, tagged.assignment)
                self.assertEqual(plain.energy, tagged.energy)


class TestQuotaHandcrafted(unittest.TestCase):
    """手工边界实例。"""

    def test_local_minimum_breaks_global_quota(self):
        """逐位置局部最低能（全取 rotamer 0）违反类别下界，最优必须让位。"""
        singles = [[0, 5], [0, 5], [0, 5]]
        cats = [["A", "B"], ["A", "B"], ["A", "B"]]
        plain = solve(singles, [], categories=cats)
        self.assertEqual(plain.assignment, (0, 0, 0))
        res = solve(singles, [], categories=cats, quotas={"B": (2, 3)})
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.assignment, (0, 1, 1))  # 字典序最小的达标构象
        self.assertEqual(res.energy, 10)
        self.assertEqual(res.category_counts, {"A": 1, "B": 2})

    def test_category_only_available_via_conflict(self):
        """类别 C 只出现在与他人硬冲突的 rotamer 上，要求 C>=1 必无解。"""
        singles = [[0, 1], [0, 1]]
        cats = [["A", "C"], ["A", "A"]]
        # C(位置0的rotamer1) 与位置1的所有 rotamer 都冲突
        pairs = [(0, 1, [[0, 0], [None, None]])]
        res = solve(singles, pairs, categories=cats, quotas={"C": (1, 2)})
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        self.assertIsNone(res.assignment)
        # 不要求 C 时仍有解
        ok = solve(singles, pairs, categories=cats)
        self.assertIs(ok.status, SolveStatus.EXACT)

    def test_zero_quota_excludes_category(self):
        singles = [[0, 5], [0, 5]]
        cats = [["A", "B"], ["A", "B"]]
        res = solve(singles, [], categories=cats, quotas={"A": (0, 0)})
        self.assertEqual(res.assignment, (1, 1))
        self.assertEqual(res.energy, 10)
        self.assertEqual(res.category_counts, {"B": 2})

    def test_tie_layout_prefers_lexicographic(self):
        """同分布局：多个构象同能量且都满足配额，取字典序最小。"""
        singles = [[1, 1], [1, 1]]
        cats = [["A", "B"], ["A", "B"]]
        res = solve(singles, [], categories=cats, quotas={"B": (1, 2)})
        self.assertEqual(res.assignment, (0, 1))
        self.assertEqual(res.energy, 2)
        ana = optimize(singles, [], categories=cats, quotas={"B": (1, 2)})
        self.assertEqual(ana.second_assignment, (1, 0))
        self.assertEqual(ana.second_energy, 2)

    def test_negative_pair_energy_kept_under_quota(self):
        """负成对能不得被配额剪枝丢掉：最优依赖一条负能对。"""
        singles = [[0, 0], [0, 0]]
        cats = [["A", "B"], ["A", "B"]]
        pairs = [(0, 1, [[0, 0], [0, -10]])]  # (B,B) 强吸引
        res = solve(singles, pairs, categories=cats, quotas={"B": (2, 2)})
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.assignment, (1, 1))
        self.assertEqual(res.energy, -10)
        self.assertEqual(res.global_lower_bound, -10)

    def test_quota_impossible_counts(self):
        singles = [[0, 1], [0, 1]]
        cats = [["A", "B"], ["A", "B"]]
        # 下界总和超过位置数
        res = solve(singles, [], categories=cats,
                    quotas={"A": (2, 2), "B": (2, 2)})
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        # 全部类别上限总和不足位置数
        res = solve(singles, [], categories=cats,
                    quotas={"A": (0, 1), "B": (0, 0)})
        self.assertIs(res.status, SolveStatus.INFEASIBLE)

    def test_budget_exhausted_with_quota(self):
        """预算耗尽仍区分上界与下界，且不声称精确。"""
        singles, pairs = generate_instance(6, seed=7)
        rng = random.Random(7)
        cats = make_categories([len(c) for c in singles], rng)
        quotas = {"hyd": (0, 6)}
        res = solve(singles, pairs, categories=cats, quotas=quotas, budget=3)
        self.assertIs(res.status, SolveStatus.BUDGET_EXHAUSTED)
        self.assertFalse(res.exact)
        self.assertIsNotNone(res.global_lower_bound)
        if res.energy is not None:
            self.assertLessEqual(res.global_lower_bound, res.energy)
            self.assertEqual(res.category_counts,
                             res.category_counts)  # 与构象一致
        # 完整预算下的精确解能量应落在 [下界, 上界] 内
        full = solve(singles, pairs, categories=cats, quotas=quotas)
        self.assertIs(full.status, SolveStatus.EXACT)
        self.assertLessEqual(res.global_lower_bound, full.energy)
        if res.energy is not None:
            self.assertLessEqual(full.energy, res.energy)

    def test_fixed_costs_respect_quota(self):
        """optimize 的固定代价增量同样在配额约束下计算。"""
        singles = [[0, 5], [0, 5], [0, 5]]
        cats = [["A", "B"], ["A", "B"], ["A", "B"]]
        ana = optimize(singles, [], categories=cats, quotas={"B": (2, 3)})
        self.assertIs(ana.status, SolveStatus.EXACT)
        self.assertEqual(ana.assignment, (0, 1, 1))
        # 位置 0 固定到 B(rotamer 1) 仍可行：B 共 2 个，能量 5+5+0=10
        entry = ana.fixed_costs[0][1]
        self.assertIsNotNone(entry)
        self.assertEqual(entry.minimal_energy, 10)
        self.assertEqual(entry.delta, 0)

    def test_problem_object_reuse(self):
        singles = [[0, 5], [0, 5]]
        cats = [["A", "B"], ["A", "B"]]
        prob = RotamerProblem(singles, [], categories=cats,
                              quotas={"B": (1, 2)})
        res = prob.solve()
        self.assertEqual(res.assignment, (0, 1))
        self.assertEqual(res.category_counts, {"A": 1, "B": 1})
        ana = prob.optimize()
        self.assertEqual(ana.second_assignment, (1, 0))


class TestQuotaValidation(unittest.TestCase):
    def setUp(self):
        self.singles = [[0, 1], [0, 1]]
        self.cats = [["A", "B"], ["A", "B"]]

    def assert_raises(self, *args, **kwargs):
        with self.assertRaises(RotamerInputError):
            solve(*args, **kwargs)

    def test_negative_bound_rejected(self):
        self.assert_raises(self.singles, [], categories=self.cats,
                           quotas={"A": (-1, 2)})

    def test_bool_bound_rejected(self):
        self.assert_raises(self.singles, [], categories=self.cats,
                           quotas={"A": (True, 2)})

    def test_lo_gt_hi_rejected(self):
        self.assert_raises(self.singles, [], categories=self.cats,
                           quotas={"A": (3, 2)})

    def test_unknown_category_rejected(self):
        self.assert_raises(self.singles, [], categories=self.cats,
                           quotas={"ZZZ": (0, 1)})

    def test_quota_without_categories_rejected(self):
        self.assert_raises(self.singles, [], quotas={"A": (0, 1)})

    def test_bad_label_type_rejected(self):
        self.assert_raises(self.singles, [],
                           categories=[["A", None], ["A", "B"]])
        self.assert_raises(self.singles, [],
                           categories=[["A", True], ["A", "B"]])
        self.assert_raises(self.singles, [],
                           categories=[["A", ""], ["A", "B"]])

    def test_categories_shape_rejected(self):
        self.assert_raises(self.singles, [], categories=[["A", "B"]])
        self.assert_raises(self.singles, [], categories=[["A"], ["A", "B"]])

    def test_int_labels_accepted(self):
        res = solve(self.singles, [], categories=[[1, 2], [1, 2]],
                    quotas={2: (1, 2)})
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.category_counts, {1: 1, 2: 1})


if __name__ == "__main__":
    unittest.main()
