'''类别标签与全局数量配额的测试。

期望值来源：tests/reference.py 的独立全枚举（不导入被测核心），
以及手工构造的边界实例（局部最低能破坏配额、类别只在冲突位置可选、
零配额、同分布局、预算耗尽与无解的区分）。
'''

import unittest

from rotamer_select import (
    optimize,
    solve,
    RotamerProblem,
    RotamerInputError,
    SolveStatus,
    FixedCostStatus,
)
from reference import (
    brute_best,
    brute_best_quota,
    brute_second_quota,
    brute_fixed_quota,
)
from datagen import generate_instance, generate_categories, generate_quotas


def _with_quota(n, seed, **kw):
    singles, pairs = generate_instance(n, seed=seed, **kw)
    cats = generate_categories([len(c) for c in singles], seed=seed)
    quotas = generate_quotas(cats, n, seed=seed)
    return singles, pairs, cats, quotas


class TestQuotaAgainstBruteForce(unittest.TestCase):
    '''2..6 位置随机实例：带配额的最优/次优/固定增量对独立全枚举。'''

    def test_best_with_quota_matches_brute(self):
        for n in range(2, 7):
            for seed in range(30):
                with self.subTest(n=n, seed=seed):
                    singles, pairs, cats, quotas = _with_quota(n, 1000 + seed)
                    res = solve(singles, pairs,
                                rotamer_categories=cats, quotas=quotas)
                    ba, be = brute_best_quota(singles, pairs, cats, quotas)
                    if be is None:
                        self.assertIs(res.status, SolveStatus.INFEASIBLE)
                        self.assertIsNone(res.assignment)
                        self.assertIsNone(res.category_counts)
                    else:
                        self.assertIs(res.status, SolveStatus.EXACT)
                        self.assertEqual(res.energy, be)
                        self.assertEqual(tuple(res.assignment), ba)
                        # 类别计数与构象一致且满足配额
                        expect = {}
                        for p, r in enumerate(ba):
                            c = cats[p][r]
                            expect[c] = expect.get(c, 0) + 1
                        self.assertEqual(res.category_counts, expect)
                        for c, (lo, hi) in quotas.items():
                            self.assertLessEqual(lo, expect.get(c, 0))
                            self.assertLessEqual(expect.get(c, 0), hi)

    def test_second_with_quota_matches_brute(self):
        for n in range(2, 7):
            for seed in range(20):
                with self.subTest(n=n, seed=seed):
                    singles, pairs, cats, quotas = _with_quota(n, 2000 + seed)
                    ana = optimize(singles, pairs,
                                   rotamer_categories=cats, quotas=quotas)
                    ba, be = brute_best_quota(singles, pairs, cats, quotas)
                    sa, se = brute_second_quota(singles, pairs, cats, quotas)
                    if be is None:
                        self.assertIs(ana.status, SolveStatus.INFEASIBLE)
                        continue
                    self.assertEqual(ana.energy, be)
                    if se is None:
                        self.assertIsNone(ana.second_assignment)
                    else:
                        self.assertEqual(ana.second_energy, se)
                        self.assertEqual(tuple(ana.second_assignment), sa)
                        self.assertNotEqual(ana.second_assignment,
                                            ana.assignment)

    def test_fixed_costs_with_quota_match_brute(self):
        checked = 0
        for n in range(2, 7):
            for seed in range(8):
                singles, pairs, cats, quotas = _with_quota(n, 3000 + seed)
                ana = optimize(singles, pairs,
                               rotamer_categories=cats, quotas=quotas)
                ba, be = brute_best_quota(singles, pairs, cats, quotas)
                if ba is None:
                    continue
                for p in range(n):
                    for r in range(len(singles[p])):
                        entry = ana.fixed_costs[p][r]
                        if r == ba[p]:
                            self.assertIsNone(entry)
                            continue
                        checked += 1
                        fa, fe = brute_fixed_quota(
                            singles, pairs, cats, quotas, p, r)
                        if fe is None:
                            self.assertIs(entry.status,
                                          FixedCostStatus.INFEASIBLE)
                        else:
                            self.assertIs(entry.status,
                                          FixedCostStatus.OPTIMAL)
                            self.assertEqual(entry.minimal_energy, fe)
                            self.assertEqual(entry.delta, fe - be)
        self.assertGreater(checked, 50)

    def test_categories_without_quotas_just_report_counts(self):
        # 只给类别、不给配额：解与无类别时完全一致，仅多报计数。
        for n in range(2, 7):
            for seed in range(10):
                singles, pairs = generate_instance(n, seed=4000 + seed)
                cats = generate_categories([len(c) for c in singles],
                                           seed=4000 + seed)
                plain = solve(singles, pairs)
                tagged = solve(singles, pairs, rotamer_categories=cats)
                self.assertEqual(plain.status, tagged.status)
                self.assertEqual(plain.assignment, tagged.assignment)
                self.assertEqual(plain.energy, tagged.energy)
                self.assertIsNone(plain.category_counts)
                if tagged.assignment is not None:
                    expect = {}
                    for p, r in enumerate(tagged.assignment):
                        c = cats[p][r]
                        expect[c] = expect.get(c, 0) + 1
                    self.assertEqual(tagged.category_counts, expect)


class TestQuotaHandbuilt(unittest.TestCase):
    def test_local_optimum_breaks_global_quota(self):
        # 局部（也是全局无配额）最低能构象 (0,0) 全是类别 X，
        # 违反 Y>=1；带配额后必须把一个位置翻成 Y，代价 +5。
        singles = [[0, 5], [0, 5]]
        cats = [['X', 'Y'], ['X', 'Y']]
        quotas = {'Y': (1, 2)}
        plain = solve(singles, [])
        self.assertEqual(plain.assignment, (0, 0))
        self.assertEqual(plain.energy, 0)
        res = solve(singles, [], rotamer_categories=cats, quotas=quotas)
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.assignment, (0, 1))  # 字典序最小的可行最优
        self.assertEqual(res.energy, 5)
        self.assertEqual(res.category_counts, {'X': 1, 'Y': 1})
        ana = optimize(singles, [], rotamer_categories=cats, quotas=quotas)
        self.assertEqual(ana.second_assignment, (1, 0))
        self.assertEqual(ana.second_energy, 5)
        self.assertEqual(ana.second_category_counts, {'X': 1, 'Y': 1})

    def test_category_only_available_at_conflicting_positions(self):
        # 类别 B 只出现在位置 1 的 r=1，而它与位置 0 的任何 rotamer
        # 都硬冲突；要求 B>=1 必须证明无解（硬冲突不能用罚分绕过）。
        singles = [[0, 0], [0, 0]]
        pairs = [(0, 1, [[0, None], [0, None]])]
        cats = [['A', 'A'], ['A', 'B']]
        res = solve(singles, pairs,
                    rotamer_categories=cats, quotas={'B': (1, 2)})
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        self.assertTrue(res.proven_infeasible)
        # 不要求 B 时本来有解
        self.assertIs(solve(singles, pairs).status, SolveStatus.EXACT)

    def test_zero_quota_excludes_category(self):
        # max=0：类别 B 完全禁用，最优只能在 A 中选。
        singles = [[0, -100], [0, -100]]
        cats = [['A', 'B'], ['A', 'B']]
        res = solve(singles, [], rotamer_categories=cats,
                    quotas={'B': (0, 0)})
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.assignment, (0, 0))
        self.assertEqual(res.energy, 0)
        self.assertEqual(res.category_counts, {'A': 2})
        # 全部类别都被零配额封死 -> 无解
        res2 = solve(singles, [], rotamer_categories=cats,
                     quotas={'A': (0, 0), 'B': (0, 0)})
        self.assertIs(res2.status, SolveStatus.INFEASIBLE)

    def test_tie_layouts_lexicographic(self):
        # 全零能量 + 配额 B 恰好 1 次：可行最优并列 (0,1) 与 (1,0)，
        # 取字典序最小 (0,1)，次优 (1,0)。
        singles = [[0, 0], [0, 0]]
        cats = [['A', 'B'], ['A', 'B']]
        quotas = {'B': (1, 1)}
        ana = optimize(singles, [], rotamer_categories=cats, quotas=quotas)
        self.assertIs(ana.status, SolveStatus.EXACT)
        self.assertEqual(ana.assignment, (0, 1))
        self.assertEqual(ana.energy, 0)
        self.assertEqual(ana.second_assignment, (1, 0))
        self.assertEqual(ana.second_energy, 0)

    def test_min_quota_above_possible_is_infeasible(self):
        singles = [[0, 0], [0, 0]]
        cats = [['A', 'B'], ['A', 'A']]
        # B 全链只有 1 个 rotamer 可取，要求 >=2 直接证明无解
        res = solve(singles, [], rotamer_categories=cats,
                    quotas={'B': (2, 2)})
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        # 配额键是未出现的类别且 min>0：同样证明无解
        res2 = solve(singles, [], rotamer_categories=cats,
                     quotas={'Z': (1, 2)})
        self.assertIs(res2.status, SolveStatus.INFEASIBLE)

    def test_quota_with_negative_pair_energies(self):
        # 负成对能吸引 (1,1)，但配额 X>=1 禁止全取 Y；
        # 不能为了剪枝丢掉负对能，结果要与全枚举一致。
        singles = [[0, 5], [0, 5]]
        pairs = [(0, 1, [[0, 0], [0, -20]])]
        cats = [['X', 'Y'], ['X', 'Y']]
        quotas = {'X': (1, 1)}  # 恰好一个 X：排除全 X 的 (0,0) 与全 Y 的 (1,1)
        res = solve(singles, pairs, rotamer_categories=cats, quotas=quotas)
        ba, be = brute_best_quota(singles, pairs, cats, quotas)
        self.assertEqual(res.assignment, ba)
        self.assertEqual(res.energy, be)
        self.assertEqual(res.energy, 5)  # (0,1) 或并列，能量 5


class TestQuotaBudget(unittest.TestCase):
    def setUp(self):
        self.singles, self.pairs = generate_instance(
            12, k_range=(4, 5), seed=5111, none_prob=0.01,
            single_none_prob=0.0)
        self.cats = generate_categories(
            [len(c) for c in self.singles], seed=5111)
        self.quotas = generate_quotas(self.cats, 12, seed=5111)

    def test_budget_exhaustion_bounds_with_quota(self):
        exact = solve(self.singles, self.pairs,
                      rotamer_categories=self.cats, quotas=self.quotas)
        self.assertIs(exact.status, SolveStatus.EXACT)
        res = solve(self.singles, self.pairs,
                    rotamer_categories=self.cats, quotas=self.quotas,
                    budget=10)
        self.assertIs(res.status, SolveStatus.BUDGET_EXHAUSTED)
        self.assertFalse(res.exact)
        self.assertFalse(res.proven_infeasible)
        self.assertIsNotNone(res.global_lower_bound)
        self.assertLessEqual(res.global_lower_bound, exact.energy)
        if res.energy is not None:
            self.assertLessEqual(exact.energy, res.energy)
            self.assertLessEqual(res.global_lower_bound, res.energy)

    def test_quota_infeasible_is_not_budget_exhausted(self):
        # 不可能满足的配额（min 超过位置数）在小预算下也必须报 INFEASIBLE，
        # 而不是 BUDGET_EXHAUSTED。
        res = solve(self.singles, self.pairs,
                    rotamer_categories=self.cats,
                    quotas={'A': (13, 20)}, budget=5)
        self.assertIs(res.status, SolveStatus.INFEASIBLE)


class TestQuotaValidation(unittest.TestCase):
    def setUp(self):
        self.singles = [[0, 1], [2, 3]]
        self.cats = [['A', 'B'], ['A', 'B']]

    def test_categories_shape_errors(self):
        with self.assertRaises(RotamerInputError):
            solve(self.singles, [], rotamer_categories=[['A', 'B']])  # 少一行
        with self.assertRaises(RotamerInputError):
            solve(self.singles, [],
                  rotamer_categories=[['A'], ['A', 'B']])  # 列数不符
        with self.assertRaises(RotamerInputError):
            solve(self.singles, [],
                  rotamer_categories=[['A', None], ['A', 'B']])  # 非法标签
        with self.assertRaises(RotamerInputError):
            solve(self.singles, [],
                  rotamer_categories=[['A', True], ['A', 'B']])  # bool 标签
        with self.assertRaises(RotamerInputError):
            solve(self.singles, [], rotamer_categories='AB')

    def test_quota_bound_errors(self):
        for bad in ({'A': (-1, 2)},   # 负下界
                    {'A': (0, -1)},   # 负上界
                    {'A': (3, 2)},    # 空区间
                    {'A': (True, 2)},  # bool 冒充整数
                    {'A': (0, 1.5)},  # 非整数
                    {'A': 1},         # 非二元组
                    {None: (0, 1)}):  # 非法类别键
            with self.subTest(bad=bad):
                with self.assertRaises(RotamerInputError):
                    solve(self.singles, [],
                          rotamer_categories=self.cats, quotas=bad)
        with self.assertRaises(RotamerInputError):
            solve(self.singles, [], quotas={'A': (0, 1)})  # 缺类别
        with self.assertRaises(RotamerInputError):
            solve(self.singles, [], rotamer_categories=self.cats,
                  quotas=[('A', (0, 1))])  # quotas 非 dict

    def test_problem_class_with_quota(self):
        prob = RotamerProblem(self.singles, [],
                              rotamer_categories=self.cats,
                              quotas={'B': (1, 2)})
        res = prob.solve()
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(sum(res.category_counts.values()), 2)
        ana = prob.optimize()
        self.assertEqual(ana.category_counts, res.category_counts)


if __name__ == '__main__':
    unittest.main(verbosity=2)
