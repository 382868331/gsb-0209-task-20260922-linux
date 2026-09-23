"""被测核心的必要测试。

期望值来源：
- tests/reference.py 的独立全枚举（n<=6，不导入 rotamer_select）；
- tests/reference.py 的独立链式 DP（n 可达 12，算法与被测不同）；
- 手工构造的边界实例。
被测结果与 energy_of 的往返一致性只作为补充断言。
"""

import math
import unittest

from rotamer_select import (
    optimize,
    solve,
    energy_of,
    RotamerProblem,
    RotamerInputError,
    SolveStatus,
    FixedCostStatus,
    MAX_POSITIONS,
    MAX_ROTAMERS,
)
from reference import brute_best, brute_second, brute_fixed, chain_dp
from datagen import generate_instance


class TestAgainstBruteForce(unittest.TestCase):
    """2..6 位置随机实例：最优/次优/固定增量全部对独立全枚举。"""

    def test_best_matches_brute_force(self):
        for n in range(2, 7):
            for seed in range(40):
                with self.subTest(n=n, seed=seed):
                    singles, pairs = generate_instance(
                        n, seed=seed, extra_pairs=(seed % 3)
                    )
                    res = solve(singles, pairs)
                    ba, be = brute_best(singles, pairs)
                    if be is None:
                        self.assertIs(res.status, SolveStatus.INFEASIBLE)
                        self.assertIsNone(res.assignment)
                        self.assertIsNone(res.energy)
                        self.assertIsNone(res.global_lower_bound)
                    else:
                        self.assertIs(res.status, SolveStatus.EXACT)
                        self.assertEqual(res.energy, be)
                        self.assertEqual(tuple(res.assignment), ba)
                        self.assertEqual(res.global_lower_bound, be)
                        # 补充：往返一致
                        self.assertEqual(energy_of(singles, pairs, res.assignment), be)

    def test_second_best_matches_brute_force(self):
        for n in range(2, 7):
            for seed in range(30):
                with self.subTest(n=n, seed=seed):
                    singles, pairs = generate_instance(n, seed=100 + seed)
                    ana = optimize(singles, pairs)
                    sa, se = brute_second(singles, pairs)
                    _, be = brute_best(singles, pairs)
                    if be is None:
                        self.assertIs(ana.status, SolveStatus.INFEASIBLE)
                        continue
                    self.assertIs(ana.status, SolveStatus.EXACT)
                    if se is None:
                        self.assertIsNone(ana.second_assignment)
                    else:
                        self.assertEqual(ana.second_energy, se)
                        self.assertEqual(tuple(ana.second_assignment), sa)
                        self.assertNotEqual(ana.second_assignment, ana.assignment)
                        self.assertGreaterEqual(ana.second_energy, ana.energy)

    def test_fixed_costs_match_brute_force(self):
        checked = 0
        for n in range(2, 7):
            for seed in range(12):
                with self.subTest(n=n, seed=seed):
                    singles, pairs = generate_instance(n, seed=200 + seed)
                    ana = optimize(singles, pairs)
                    ba, be = brute_best(singles, pairs)
                    if ba is None:
                        continue
                    for p in range(n):
                        for r in range(len(singles[p])):
                            entry = ana.fixed_costs[p][r]
                            if r == ba[p]:
                                self.assertIsNone(entry)
                                continue
                            checked += 1
                            fa, fe = brute_fixed(singles, pairs, p, r)
                            if fe is None:
                                self.assertIs(entry.status, FixedCostStatus.INFEASIBLE)
                                self.assertIsNone(entry.delta)
                            else:
                                self.assertIs(entry.status, FixedCostStatus.OPTIMAL)
                                self.assertEqual(entry.minimal_energy, fe)
                                self.assertEqual(entry.delta, fe - be)
        self.assertGreater(checked, 200)  # 确实核到了大量固定项


class TestFixedAssignmentsDirect(unittest.TestCase):
    def test_solve_with_fixed_matches_brute(self):
        for n in range(2, 7):
            for seed in range(10):
                singles, pairs = generate_instance(n, seed=300 + seed)
                ba, _ = brute_best(singles, pairs)
                if ba is None:
                    continue
                p = seed % n
                r = (ba[p] + 1) % len(singles[p])
                res = solve(singles, pairs, fixed={p: r})
                fa, fe = brute_fixed(singles, pairs, p, r)
                if fe is None:
                    self.assertIs(res.status, SolveStatus.INFEASIBLE)
                else:
                    self.assertIs(res.status, SolveStatus.EXACT)
                    self.assertEqual(res.energy, fe)
                    self.assertEqual(tuple(res.assignment), tuple(fa))
                    self.assertEqual(res.assignment[p], r)


class TestNegativePairEnergies(unittest.TestCase):
    def test_negative_pair_dominates_choice(self):
        # 单体能倾向 rotamer 0，但对能强烈为负，最优必须一起取 1。
        singles = [[0, 5], [0, 5]]
        pairs = [(0, 1, [[0, 0], [0, -20]])]
        res = solve(singles, pairs)
        self.assertEqual(res.assignment, (1, 1))
        self.assertEqual(res.energy, 5 + 5 - 20)

    def test_all_negative_pair_random(self):
        for n in range(2, 7):
            for seed in range(10):
                with self.subTest(n=n, seed=seed):
                    singles, pairs = generate_instance(
                        n, seed=400 + seed, none_prob=0.0, neg_prob=1.0,
                        single_none_prob=0.0)
                    res = solve(singles, pairs)
                    ba, be = brute_best(singles, pairs)
                    self.assertEqual(res.energy, be)
                    self.assertEqual(res.assignment, ba)
                    self.assertLess(be, 0)


class TestHardCollisions(unittest.TestCase):
    def test_pair_all_none_means_infeasible(self):
        singles = [[0, 1], [2, 3]]
        pairs = [(0, 1, [[None, None], [None, None]])]
        res = solve(singles, pairs)
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        self.assertTrue(res.proven_infeasible)
        self.assertFalse(res.exact)

    def test_single_column_all_none_means_infeasible(self):
        singles = [[0, 1], [None, None]]
        res = solve(singles, [])
        self.assertIs(res.status, SolveStatus.INFEASIBLE)

    def test_only_one_feasible_conformation(self):
        singles = [[0, 10], [10, 0]]
        pairs = [(0, 1, [
            [None, None],
            [None, 7],
        ])]
        res = solve(singles, pairs)
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.assignment, (1, 1))
        self.assertEqual(res.energy, 17)
        ana = optimize(singles, pairs)
        self.assertIsNone(ana.second_assignment)  # 唯一可行构象，没有次优

    def test_fixed_into_collision_infeasible(self):
        # 位置1取 rotamer 0 时，与位置0的任何 rotamer 都硬冲突：
        # 固定该约束必须报 INFEASIBLE（而不是预算耗尽或静默纠正）。
        singles = [[0, 0], [0, 0]]
        pairs = [(0, 1, [[None, 0], [None, 0]])]
        res = solve(singles, pairs, fixed={1: 0})
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        ana = optimize(singles, pairs)
        e0 = ana.fixed_costs[1][0]
        self.assertIs(e0.status, FixedCostStatus.INFEASIBLE)
        self.assertIsNone(e0.delta)
        self.assertIsNone(e0.minimal_energy)


class TestTieBreaking(unittest.TestCase):
    def test_all_zero_takes_lexicographic_minimum(self):
        singles = [[0, 0, 0], [0, 0]]
        res = solve(singles, [])
        self.assertEqual(res.assignment, (0, 0))
        ana = optimize(singles, [])
        self.assertEqual(ana.second_assignment, (0, 1))
        self.assertEqual(ana.second_energy, 0)

    def test_handbuilt_tie(self):
        # 构象 (0,1) 与 (1,0) 都是 1，(0,0)=9，(1,1)=9：取字典序 (0,1)。
        singles = [[0, 0], [0, 0]]
        pairs = [(0, 1, [[9, 1], [1, 9]])]
        res = solve(singles, pairs)
        self.assertEqual(res.energy, 1)
        self.assertEqual(res.assignment, (0, 1))
        ana = optimize(singles, pairs)
        self.assertEqual(ana.second_assignment, (1, 0))
        self.assertEqual(ana.second_energy, 1)

    def test_ties_random_match_brute(self):
        # 高 none 概率关闭，能量小范围，大量并列，核对字典序选择。
        for seed in range(20):
            with self.subTest(seed=seed):
                singles, pairs = generate_instance(
                    4, k_range=(2, 3), seed=500 + seed,
                    none_prob=0.0, single_none_prob=0.0)
                # 压平能量制造并列
                singles = [[v // 3 if v is not None else None for v in col]
                           for col in singles]
                pairs = [(i, j, [[(v // 4 if v is not None else None) for v in row]
                                 for row in m]) for i, j, m in pairs]
                res = solve(singles, pairs)
                ba, be = brute_best(singles, pairs)
                self.assertEqual(res.energy, be)
                self.assertEqual(res.assignment, ba)


class TestLargeChainDP(unittest.TestCase):
    """最多 12 位置：对独立链式 DP（非全枚举）核对最优。"""

    def test_12_positions_chain(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                singles, pairs = generate_instance(
                    MAX_POSITIONS, seed=600 + seed, none_prob=0.05,
                    single_none_prob=0.05)
                res = solve(singles, pairs)
                self.assertIs(res.status, SolveStatus.EXACT)
                a, e = chain_dp(singles, pairs)
                self.assertIsNotNone(a)
                self.assertEqual(res.energy, e)
                self.assertEqual(res.assignment, a)

    def test_12_positions_infeasible_is_proven(self):
        singles, pairs = generate_instance(
            MAX_POSITIONS, seed=601, none_prob=0.0)
        # 人为制造：位置 7 全部 rotamer 与位置 8 的 rotamer 0 冲突，
        # 再把位置 8 单体能除 rotamer 0 外置 None，使整链无解。
        i, j = 7, 8
        for idx, (a, b, m) in enumerate(pairs):
            if (a, b) == (i, j):
                pairs[idx] = (i, j, [[None] + [v for v in row[1:]] for row in m])
        singles[8] = [singles[8][0]] + [None] * (len(singles[8]) - 1)
        res = solve(singles, pairs)
        self.assertIs(res.status, SolveStatus.INFEASIBLE)


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.singles, self.pairs = generate_instance(
            MAX_POSITIONS, k_range=(4, 5), seed=700, none_prob=0.01,
            single_none_prob=0.0)

    def test_tiny_budget_exhaustion_is_distinct_from_infeasible(self):
        exact = solve(self.singles, self.pairs)
        self.assertIs(exact.status, SolveStatus.EXACT)
        star = exact.energy

        # budget=15：首个叶子在深度 12 之后，能拿到一个可行解但远未证明。
        # 该 12 位置 4..5 rotamer 实例精确解需展开 2485 个节点（种子确定）。
        res = solve(self.singles, self.pairs, budget=15)
        self.assertIs(res.status, SolveStatus.BUDGET_EXHAUSTED)
        self.assertFalse(res.exact)
        self.assertFalse(res.proven_infeasible)
        # 界的口径：下界 <= 真实最优 <= 已知最优
        self.assertIsNotNone(res.global_lower_bound)
        self.assertLessEqual(res.global_lower_bound, star)
        self.assertIsNotNone(res.energy)
        self.assertLessEqual(star, res.energy)
        self.assertLessEqual(res.global_lower_bound, res.energy)
        self.assertEqual(res.nodes_expanded, 15)

    def test_budget_exhausted_without_any_leaf(self):
        # 预算 1：只展开根，尚未有叶解；仍要给可信下界。
        res = solve(self.singles, self.pairs, budget=1)
        self.assertIs(res.status, SolveStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(res.energy)
        self.assertIsNone(res.assignment)
        star = solve(self.singles, self.pairs).energy
        self.assertIsNotNone(res.global_lower_bound)
        self.assertLessEqual(res.global_lower_bound, star)

    def test_second_best_budget_exhausted(self):
        # 次优搜索（排除最优构象）预算耗尽时，也要给最优已知与可信下界，
        # 而放开预算后二者应收敛到同一个已证次优能量。
        from rotamer_select.solver import compile_problem, branch_and_bound
        compiled = compile_problem(self.singles, self.pairs)
        base = branch_and_bound(compiled)
        star2_exact = branch_and_bound(compiled, forbid=base["assignment"],
                                       budget=10_000_000)
        self.assertEqual(star2_exact["status"], "exact")
        star2 = branch_and_bound(compiled, forbid=base["assignment"], budget=2485)
        self.assertEqual(star2["status"], "budget_exhausted")
        self.assertIsNotNone(star2["energy"])
        self.assertEqual(star2["energy"], star2_exact["energy"])
        self.assertLessEqual(star2["global_lower_bound"], star2_exact["energy"])

    def test_exactly_enough_budget_is_exact(self):
        res = solve(self.singles, self.pairs, budget=10_000_000)
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.global_lower_bound, res.energy)

    def test_optimize_aggregates_budget_status(self):
        # 每个子搜索 3000 个节点：base（需 2485）可证最优，
        # 但多数带固定约束的子搜索在该预算下耗尽（种子 700 下实测 21 个），
        # 且它们都已找到可行叶，因此 delta 与其下界都应给出且口径正确。
        ana = optimize(self.singles, self.pairs, budget=3000)
        self.assertIs(ana.status, SolveStatus.BUDGET_EXHAUSTED)
        self.assertEqual(ana.energy, -117)  # base 仍为已证最优
        best_known = [e for col in ana.fixed_costs for e in col
                      if e is not None and e.status is FixedCostStatus.BEST_KNOWN]
        self.assertTrue(best_known)
        for e in best_known:
            self.assertIsNotNone(e.lower_bound_delta)
            self.assertIsNotNone(e.delta)
            self.assertLessEqual(e.lower_bound_delta, e.delta)


class TestInputValidation(unittest.TestCase):
    def _base(self):
        return [[0, 1], [2, 3, 4]], [(0, 1, [[0, 0, 0], [1, 1, 1]])]

    def test_wrong_row_count(self):
        singles, pairs = self._base()
        bad_matrix = [[0, 0, 0]]  # 位置0有2个rotamer，只给1行
        with self.assertRaises(RotamerInputError) as cm:
            solve(singles, [(0, 1, bad_matrix)])
        self.assertIn("pair_matrices[0]", str(cm.exception))
        self.assertIn("期望 2 行", str(cm.exception))

    def test_wrong_column_count(self):
        singles, pairs = self._base()
        bad_matrix = [[0, 0], [1, 1]]  # 位置1有3个rotamer，只给2列
        with self.assertRaises(RotamerInputError) as cm:
            solve(singles, [(0, 1, bad_matrix)])
        self.assertIn("期望 3 列", str(cm.exception))
        self.assertIn("[0]", str(cm.exception))

    def test_ragged_and_non_sequence(self):
        with self.assertRaises(RotamerInputError):
            solve([[0, 1], [2]], [])  # 每位置 rotamer 数必须 >=2
        with self.assertRaises(RotamerInputError):
            solve([[0, 1], [2, 3, 4, 5, 6, 7]], [])  # >5
        big = [[0, 1]] * (MAX_POSITIONS + 1)  # 位置数 >12
        with self.assertRaises(RotamerInputError):
            solve(big, [])
        with self.assertRaises(RotamerInputError):
            solve("not-a-sequence", [])
        with self.assertRaises(RotamerInputError):
            solve([[0, 1], [2, 3]], [(0, 1, 5)])

    def test_pair_index_errors(self):
        singles, _ = self._base()
        with self.assertRaises(RotamerInputError):
            solve(singles, [(0, 0, [[0, 0, 0], [1, 1, 1]])])  # i==j
        with self.assertRaises(RotamerInputError):
            solve(singles, [(1, 0, [[0, 0], [0, 0], [0, 0]])])  # i>j
        with self.assertRaises(RotamerInputError):
            solve(singles, [(0, 2, [[0, 0], [0, 0]])])  # 越界
        m = [[0, 0, 0], [1, 1, 1]]
        with self.assertRaises(RotamerInputError):
            solve(singles, [(0, 1, m), (0, 1, m)])  # 重复位置对

    def test_reject_nan_infinity_float(self):
        nan = float("nan")
        inf = float("inf")
        with self.assertRaises(RotamerInputError) as cm:
            solve([[0, nan], [1, 2]], [])
        self.assertIn("NaN", str(cm.exception))
        with self.assertRaises(RotamerInputError) as cm:
            solve([[0, inf], [1, 2]], [])
        self.assertIn("Infinity", str(cm.exception))
        with self.assertRaises(RotamerInputError):
            solve([[0, -inf], [1, 2]], [])
        with self.assertRaises(RotamerInputError) as cm:
            solve([[0, 1], [2, 3]], [(0, 1, [[0, nan], [0, 0]])])
        self.assertIn("[0][1]", str(cm.exception))
        self.assertTrue(math.isnan(nan))  # 占位使用，避免静态工具提示

    def test_reject_bool_as_integer(self):
        with self.assertRaises(RotamerInputError):
            solve([[True, 1], [2, 3]], [])
        with self.assertRaises(RotamerInputError):
            solve([[0, 1], [2, 3]], [(0, 1, [[0, False], [0, 0]])])
        with self.assertRaises(RotamerInputError):
            solve([[0, 1], [2, 3]], [], budget=True)
        with self.assertRaises(RotamerInputError):
            solve([[0, 1], [2, 3]], [], fixed={True: 0})
        with self.assertRaises(RotamerInputError):
            solve([[0, 1], [2, 3]], [], fixed={0: False})
        with self.assertRaises(RotamerInputError):
            RotamerProblem([[0, 1], [2, 3]], [(True, 1, [[0, 0], [0, 0]])])

    def test_reject_float_energy(self):
        with self.assertRaises(RotamerInputError):
            solve([[0, 1.5], [2, 3]], [])

    def test_budget_validation(self):
        singles, pairs = self._base()
        for bad in (0, -5, 1.0, "100", None):
            with self.assertRaises(RotamerInputError):
                solve(singles, pairs, budget=bad)

    def test_fixed_validation(self):
        singles, pairs = self._base()
        with self.assertRaises(RotamerInputError):
            solve(singles, pairs, fixed={2: 0})  # 位置越界
        with self.assertRaises(RotamerInputError):
            solve(singles, pairs, fixed={0: 5})  # rotamer 越界

    def test_energy_of_rejects_collision(self):
        singles = [[0, 1], [2, 3]]
        pairs = [(0, 1, [[None, 0], [0, 0]])]
        with self.assertRaises(RotamerInputError):
            energy_of(singles, pairs, (0, 0))
        with self.assertRaises(RotamerInputError):
            energy_of(singles, pairs, (0,))
        self.assertEqual(energy_of(singles, pairs, (1, 0)), 1 + 2 + 0)


class TestNonAdjacentPairs(unittest.TestCase):
    def test_skip_edge_pair_only(self):
        # 只给非相邻位置对 (0, 2)，中间位置 1 只有单体能。
        singles = [[0, 1], [10, 0], [0, 1]]
        pairs = [(0, 2, [[5, -8], [2, 3]])]
        res = solve(singles, pairs)
        ba, be = brute_best(singles, pairs)
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(res.energy, be)
        self.assertEqual(res.assignment, ba)

    def test_arbitrary_pairs_random(self):
        # n<=6 + 多条非相邻对，仍对独立全枚举。
        for seed in range(20):
            with self.subTest(seed=seed):
                singles, pairs = generate_instance(
                    6, seed=800 + seed, extra_pairs=4)
                res = solve(singles, pairs)
                ba, be = brute_best(singles, pairs)
                self.assertEqual(res.energy, be)
                self.assertEqual(res.assignment, ba)


class TestLowerBoundFuzz(unittest.TestCase):
    def test_lower_bound_always_valid_under_random_budgets(self):
        # 任意预算下（含耗尽、含拿不到叶），global_lower_bound 必须 <= 已证最优。
        for seed in range(12):
            singles, pairs = generate_instance(
                MAX_POSITIONS, k_range=(3, 5), seed=900 + seed,
                none_prob=0.02, single_none_prob=0.02)
            exact = solve(singles, pairs)
            self.assertIs(exact.status, SolveStatus.EXACT)
            for budget in (1, 5, 20, 100, 1000):
                with self.subTest(seed=seed, budget=budget):
                    res = solve(singles, pairs, budget=budget)
                    if res.status is SolveStatus.BUDGET_EXHAUSTED:
                        self.assertIsNotNone(res.global_lower_bound)
                        self.assertLessEqual(
                            res.global_lower_bound, exact.energy)
                        if res.energy is not None:
                            self.assertLessEqual(
                                exact.energy, res.energy)
                            self.assertLessEqual(
                                res.global_lower_bound, res.energy)

    def test_infeasible_all_rotamers_blocked_at_a_position(self):
        # 每个 rotamer 都在某条边上存在硬冲突的链式实例：全枚举确认无解。
        singles = [[0, 0], [0, 0, 0], [0, 0]]
        m01 = [[None, 0, 0], [None, 0, 0]]  # 位置1 的 r=0 与位置0 恒冲突
        m12 = [[0, 0], [None, None], [None, None]]  # r=1, r=2 与位置2 恒冲突
        pairs = [(0, 1, m01), (1, 2, m12)]
        ba, be = brute_best(singles, pairs)
        self.assertIsNone(ba)
        res = solve(singles, pairs)
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        self.assertIsNone(res.global_lower_bound)


class TestProblemClassAPI(unittest.TestCase):
    def test_reusable_compiled_problem(self):
        singles = [[0, 5], [0, 5]]
        pairs = [(0, 1, [[0, 0], [0, -20]])]
        prob = RotamerProblem(singles, pairs)
        self.assertEqual(prob.n, 2)
        self.assertEqual(prob.counts, (2, 2))
        r1 = prob.solve()
        r2 = prob.solve(fixed={0: 0})
        self.assertEqual(r1.assignment, (1, 1))
        self.assertEqual(r2.assignment[0], 0)
        ana = prob.optimize()
        self.assertEqual(ana.energy, r1.energy)
        self.assertEqual(prob.energy_of((1, 1)), ana.energy)

    def test_single_position_allowed(self):
        # 契约下限是“最多12位置”，1 个位置（仅单体能）合法。
        res = solve([[3, -1, 2]], [])
        self.assertEqual(res.assignment, (1,))
        self.assertEqual(res.energy, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
