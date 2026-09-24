"""多骨架共用 rotamer 两阶段选择的测试。

期望值来源：
- tests/reference.py 的独立全枚举 brute_backbones（不导入 rotamer_select）；
- 手算的极小实例（minimax 与平均冲突、权重破并列、对能参与等）；
- 手工构造的边界与非法输入。
与单骨架 solve 的结果一致性只作为补充断言，不用被测实现生成期望值。
"""

import unittest

from rotamer_select import (
    solve,
    select_for_backbones,
    MultiBackboneProblem,
    RotamerInputError,
    BackboneSolveStatus,
    MULTI_MAX_POSITIONS,
    MULTI_MAX_ROTAMERS,
    MAX_BACKBONES,
)
from reference import brute_backbones
from datagen import generate_backbones


class TestAgainstBruteForce(unittest.TestCase):
    """1..6 位点、1..3 骨架的随机实例：选择与每骨架得分对独立全枚举。"""

    def test_select_matches_brute(self):
        for n in range(1, 7):
            for b in range(1, 4):
                for seed in range(6):
                    with self.subTest(n=n, b=b, seed=seed):
                        backbones = generate_backbones(
                            n, b, k_range=(2, 3), seed=1000 + seed * 7 + n)
                        res = select_for_backbones(backbones)
                        ba, bs, bw, bwsum = brute_backbones(backbones)
                        if ba is None:
                            self.assertIs(res.status,
                                          BackboneSolveStatus.INFEASIBLE)
                            self.assertIsNone(res.assignment)
                            self.assertIsNone(res.scores)
                            self.assertIsNone(res.worst_energy)
                            self.assertIsNone(res.weighted_sum)
                            continue
                        self.assertIs(res.status, BackboneSolveStatus.EXACT)
                        self.assertEqual(tuple(res.assignment), tuple(ba))
                        self.assertEqual(tuple(res.scores), tuple(bs))
                        self.assertEqual(res.worst_energy, bw)
                        self.assertEqual(res.weighted_sum, bwsum)
                        self.assertEqual(res.worst_energy, max(res.scores))
                        self.assertEqual(res.weighted_sum, sum(res.scores))
                        self.assertTrue(res.exact)

    def test_weighted_select_matches_brute(self):
        weight_sets_2 = [(1, 1), (2, 3), (1, 5)]
        weight_sets_3 = [(3, 1, 4), (2, 2, 2)]
        for n in range(1, 6):
            for seed in range(8):
                b = 3 if seed % 2 else 2
                backbones = generate_backbones(
                    n, b, k_range=(2, 3), seed=2000 + seed * 13 + n)
                pool = weight_sets_3 if b == 3 else weight_sets_2
                weights = pool[seed % len(pool)]
                with self.subTest(n=n, b=b, seed=seed, weights=weights):
                    res = select_for_backbones(backbones, weights=weights)
                    ba, bs, bw, bwsum = brute_backbones(backbones, weights)
                    if ba is None:
                        self.assertIs(res.status,
                                      BackboneSolveStatus.INFEASIBLE)
                        continue
                    self.assertEqual(tuple(res.assignment), tuple(ba))
                    self.assertEqual(res.scores, tuple(bs))
                    self.assertEqual(res.worst_energy, bw)
                    self.assertEqual(res.weighted_sum, bwsum)
                    self.assertEqual(tuple(res.weights), tuple(weights))

    def test_max_size_enumeration_counts(self):
        # 上限：6 位点 × 每点 3 rotamer × 3 骨架 = 729 个构象全枚举。
        backbones = generate_backbones(
            MULTI_MAX_POSITIONS, MAX_BACKBONES,
            k_range=(MULTI_MAX_ROTAMERS, MULTI_MAX_ROTAMERS), seed=33,
            none_prob=0.0, single_none_prob=0.0)
        prob = MultiBackboneProblem(backbones)
        self.assertEqual(prob.n, 6)
        self.assertEqual(prob.counts, (3, 3, 3, 3, 3, 3))
        res = prob.select()
        self.assertIs(res.status, BackboneSolveStatus.EXACT)
        self.assertEqual(res.assignments_evaluated, 3 ** 6)
        self.assertEqual(res.feasible_assignments, 3 ** 6)
        ba, bs, bw, bwsum = brute_backbones(backbones)
        self.assertEqual(res.assignment, ba)
        self.assertEqual(res.scores, bs)


class TestHandbuiltExamples(unittest.TestCase):
    def test_minimax_beats_average_minimizer(self):
        # 单位点 3 个 rotamer，两骨架能量对着干：
        #   r0: (0, 100)  平均 50   最坏 100
        #   r1: (100, 0)  平均 50   最坏 100
        #   r2: (60, 50)  平均 55   最坏 60  <- 平均最差但最坏最好
        backbones = [
            ([[0, 100, 60]], []),
            ([[100, 0, 50]], []),
        ]
        res = select_for_backbones(backbones)
        self.assertEqual(res.assignment, (2,))
        self.assertEqual(res.scores, (60, 50))
        self.assertEqual(res.worst_energy, 60)
        self.assertEqual(res.weighted_sum, 110)
        self.assertEqual(res.assignments_evaluated, 3)
        self.assertEqual(res.feasible_assignments, 3)

    def test_weights_break_worst_case_tie(self):
        # r0 与 r1 的最坏能量都是 10；等权时加权和也并列 -> 字典序 r0；
        # 给骨架 1 更大权重后，r1 的加权和更低 -> r1。
        backbones = [
            ([[0, 10]], []),
            ([[10, 0]], []),
        ]
        equal = select_for_backbones(backbones)
        self.assertEqual(equal.assignment, (0,))
        self.assertEqual(equal.worst_energy, 10)
        self.assertEqual(equal.weighted_sum, 10)

        weighted = select_for_backbones(backbones, weights=(1, 2))
        self.assertEqual(weighted.assignment, (1,))
        self.assertEqual(weighted.scores, (10, 0))
        self.assertEqual(weighted.worst_energy, 10)
        self.assertEqual(weighted.weighted_sum, 10)  # 1*10 + 2*0
        # 平均口径（仅供展示，比较全程用整数和）
        self.assertAlmostEqual(weighted.weighted_average, 10 / 3)

    def test_pair_energy_handbuilt_optimum(self):
        # 两骨架的对能奖励不同搭配，但只有 (0,0) 在两骨架都吃到 -5：
        #   b0 对能 [[-5,-2],[-2,10]]；b1 对能 [[-5,3],[3,4]]
        #   (0,0): (-5,-5) 最坏 -5  <- 唯一最优
        #   (0,1)/(1,0): 最坏 3；(1,1): 最坏 10
        zeros = [[0, 0], [0, 0]]
        backbones = [
            (zeros, [(0, 1, [[-5, -2], [-2, 10]])]),
            (zeros, [(0, 1, [[-5, 3], [3, 4]])]),
        ]
        res = select_for_backbones(backbones)
        self.assertEqual(res.assignment, (0, 0))
        self.assertEqual(res.scores, (-5, -5))
        self.assertEqual(res.worst_energy, -5)
        self.assertEqual(res.weighted_sum, -10)
        # 其余三种搭配的最坏能量都严格更大
        for other in ((0, 1), (1, 0), (1, 1)):
            prob = MultiBackboneProblem(backbones)
            scores = prob.scores_of(other)
            self.assertGreater(max(scores), -5)

    def test_lexicographic_on_full_tie(self):
        backbones = [
            ([[0, 0, 0], [0, 0]], []),
            ([[0, 0, 0], [0, 0]], []),
        ]
        res = select_for_backbones(backbones)
        self.assertEqual(res.assignment, (0, 0))
        self.assertEqual(res.scores, (0, 0))

    def test_negative_energies_allowed(self):
        backbones = [
            ([[-3, -8]], []),
        ]
        res = select_for_backbones(backbones)
        self.assertEqual(res.assignment, (1,))
        self.assertEqual(res.worst_energy, -8)

    def test_single_backbone_matches_core_solve(self):
        # 补充交叉核对（期望值仍以独立全枚举为准）：
        # 单骨架时 worst_energy 必须等于单骨架求解器的最优能量。
        backbones = generate_backbones(4, 1, k_range=(2, 3), seed=4242)
        singles, pairs = backbones[0]
        res = select_for_backbones(backbones)
        core = solve(singles, pairs)
        self.assertEqual(res.worst_energy, core.energy)
        self.assertEqual(res.assignment, core.assignment)
        ba, _, bw, _ = brute_backbones(backbones)
        self.assertEqual(res.assignment, ba)
        self.assertEqual(res.worst_energy, bw)


class TestBoundaries(unittest.TestCase):
    def test_one_site_one_rotamer_one_backbone(self):
        res = select_for_backbones([([[7]], [])])
        self.assertIs(res.status, BackboneSolveStatus.EXACT)
        self.assertEqual(res.assignment, (0,))
        self.assertEqual(res.scores, (7,))
        self.assertEqual(res.worst_energy, 7)
        self.assertEqual(res.weighted_sum, 7)
        self.assertEqual(res.assignments_evaluated, 1)
        self.assertEqual(res.feasible_assignments, 1)

    def test_single_rotamer_sites(self):
        # 每位点只有 1 个 rotamer（多骨架契约下限为 1）。
        backbones = [
            ([[2], [3]], [(0, 1, [[4]])]),
            ([[-1], [1]], [(0, 1, [[2]])]),
        ]
        res = select_for_backbones(backbones)
        self.assertEqual(res.assignment, (0, 0))
        self.assertEqual(res.scores, (9, 2))
        self.assertEqual(res.worst_energy, 9)
        self.assertEqual(res.assignments_evaluated, 1)

    def test_collision_makes_infeasible(self):
        # r0 在骨架 1 冲突，r1 在骨架 0 冲突：没有共用可行构象。
        backbones = [
            ([[0, None]], []),
            ([[None, 0]], []),
        ]
        res = select_for_backbones(backbones)
        self.assertIs(res.status, BackboneSolveStatus.INFEASIBLE)
        self.assertFalse(res.exact)
        self.assertIsNone(res.assignment)
        self.assertIsNone(res.scores)
        self.assertIsNone(res.worst_energy)
        self.assertIsNone(res.weighted_sum)
        self.assertIsNone(res.weighted_average)
        self.assertEqual(res.assignments_evaluated, 2)
        self.assertEqual(res.feasible_assignments, 0)

    def test_pair_collision_filters_only(self):
        # 三个构象中只有 (0,1) 在两骨架都不冲突。
        b0 = ([[0, 0], [0, 0]], [(0, 1, [[None, 0], [None, None]])])
        b1 = ([[0, 0], [0, 0]], [(0, 1, [[0, 0], [0, None]])])
        res = select_for_backbones([b0, b1])
        self.assertEqual(res.assignment, (0, 1))
        self.assertEqual(res.feasible_assignments, 1)
        self.assertEqual(res.assignments_evaluated, 4)

    def test_reusable_problem_with_different_weights(self):
        backbones = [
            ([[0, 10]], []),
            ([[10, 0]], []),
        ]
        prob = MultiBackboneProblem(backbones)
        self.assertEqual(prob.n, 1)
        self.assertEqual(prob.counts, (2,))
        self.assertEqual(prob.backbone_count, 2)
        self.assertEqual(prob.select().assignment, (0,))
        self.assertEqual(prob.select(weights=(1, 2)).assignment, (1,))
        self.assertEqual(prob.scores_of((1,)), (10, 0))


class TestScoresOf(unittest.TestCase):
    def setUp(self):
        b0 = ([[1, 2], [3, 4]], [(0, 1, [[5, 6], [7, 8]])])
        b1 = ([[-1, 0], [1, 2]], [(0, 1, [[0, -2], [-3, 0]])])
        self.prob = MultiBackboneProblem([b0, b1])

    def test_scores_handcomputed(self):
        # (0,1): b0 = 1+4+6 = 11；b1 = -1+2-2 = -1
        self.assertEqual(self.prob.scores_of((0, 1)), (11, -1))
        # (1,0): b0 = 2+3+7 = 12；b1 = 0+1-3 = -2
        self.assertEqual(self.prob.scores_of([1, 0]), (12, -2))

    def test_scores_match_selection(self):
        res = self.prob.select()
        self.assertEqual(self.prob.scores_of(res.assignment), res.scores)

    def test_scores_of_rejects_bad_assignment(self):
        with self.assertRaises(RotamerInputError):
            self.prob.scores_of((0,))          # 长度错
        with self.assertRaises(RotamerInputError):
            self.prob.scores_of((0, 5))        # 越界
        with self.assertRaises(RotamerInputError):
            self.prob.scores_of((0, True))     # bool
        with self.assertRaises(RotamerInputError):
            self.prob.scores_of("01")          # 类型错

    def test_scores_of_reports_collision_with_backbone_index(self):
        backbones = [
            ([[0, 0], [0, 0]], [(0, 1, [[0, 0], [0, 0]])]),
            ([[0, 0], [0, 0]], [(0, 1, [[0, None], [0, 0]])]),
        ]
        prob = MultiBackboneProblem(backbones)
        with self.assertRaises(RotamerInputError) as cm:
            prob.scores_of((0, 1))
        self.assertIn("骨架 1", str(cm.exception))
        self.assertIn("(0, 1)", str(cm.exception))


class TestInputValidation(unittest.TestCase):
    def _good(self):
        return [
            ([[0, 1], [2, 3]], [(0, 1, [[0, 0], [0, 0]])]),
            ([[1, 0], [3, 2]], [(0, 1, [[1, 1], [1, 1]])]),
        ]

    def test_backbones_container_errors(self):
        with self.assertRaises(RotamerInputError):
            select_for_backbones("not-a-sequence")
        with self.assertRaises(RotamerInputError):
            select_for_backbones([])                      # 0 个骨架
        with self.assertRaises(RotamerInputError):
            select_for_backbones([( [[0, 1]], []) ] * (MAX_BACKBONES + 1))
        with self.assertRaises(RotamerInputError):
            select_for_backbones([([[0, 1]], [], [])])   # 三元组而非二元组
        with self.assertRaises(RotamerInputError):
            select_for_backbones([([[0, 1]],)])          # 只有 1 项
        with self.assertRaises(RotamerInputError):
            select_for_backbones([([[0, 1]], None)])     # pair_matrices 非序列

    def test_backbones_must_share_rotamer_set(self):
        bad = [
            ([[0, 1]], []),                 # 1 位点
            ([[0, 1], [2, 3]], []),         # 2 位点
        ]
        with self.assertRaises(RotamerInputError) as cm:
            select_for_backbones(bad)
        self.assertIn("共用同一组 rotamer", str(cm.exception))

        bad2 = [
            ([[0, 1], [2, 3]], []),         # 位点 0 有 2 rotamer
            ([[0, 1, 2], [2, 3]], []),      # 位点 0 有 3 rotamer
        ]
        with self.assertRaises(RotamerInputError):
            select_for_backbones(bad2)

    def test_size_limits(self):
        # 位点数超过 6
        too_many = [([[0, 1]] * (MULTI_MAX_POSITIONS + 1), [])]
        with self.assertRaises(RotamerInputError):
            select_for_backbones(too_many)
        # 每点 rotamer 超过 3
        too_wide = [([[0, 1, 2, 3]], [])]
        with self.assertRaises(RotamerInputError):
            select_for_backbones(too_wide)
        # 每点 0 个 rotamer
        empty = [([[]], [])]
        with self.assertRaises(RotamerInputError):
            select_for_backbones(empty)

    def test_energy_type_rules_inherited(self):
        good_b1 = ([[1, 0]], [])
        with self.assertRaises(RotamerInputError):
            select_for_backbones([([[0, True]], []), good_b1])     # bool
        with self.assertRaises(RotamerInputError):
            select_for_backbones([([[0, 1.5]], []), good_b1])      # float
        with self.assertRaises(RotamerInputError):
            select_for_backbones([([[0, float("nan")]], []),
                                  good_b1])                        # NaN
        with self.assertRaises(RotamerInputError):
            select_for_backbones(
                [([[0, 0], [0, 0]], [(0, 1, [[0, 0], [0, 1]])]),
                 ([[0, 0], [0, 0]], [(0, 1, [[0, 0], [1, True]])])])
        # 矩阵维度错误仍由原校验负责
        with self.assertRaises(RotamerInputError) as cm:
            select_for_backbones([
                ([[0, 1], [2, 3]], [(0, 1, [[0, 0]])]),
                ([[0, 1], [2, 3]], [(0, 1, [[0, 0], [0, 0]])]),
            ])
        self.assertIn("期望 2 行", str(cm.exception))

    def test_weights_validation(self):
        backbones = self._good()
        for bad in (
            (1, 1, 1),          # 数量不符（2 个骨架给 3 个权重）
            (0, 1),             # 零权重
            (-1, 2),            # 负权重
            (1, True),          # bool
            (1, 2.0),           # float
            "11",               # 非序列
            (1, None),          # None 元素
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(RotamerInputError):
                    select_for_backbones(backbones, weights=bad)

    def test_default_weights_equal(self):
        res = select_for_backbones(self._good())
        self.assertEqual(res.weights, (1, 1))

    def test_no_partial_state_on_failure(self):
        # 非法构造不产生对象；同一份合法数据之后仍可正常使用。
        with self.assertRaises(RotamerInputError):
            MultiBackboneProblem([([[0, 1]], []), ([[0, 1, 2]], [])])
        prob = MultiBackboneProblem(self._good())
        self.assertEqual(prob.backbone_count, 2)

    def test_inputs_not_mutated(self):
        backbones = self._good()
        import copy
        snapshot = copy.deepcopy(backbones)
        select_for_backbones(backbones, weights=(2, 3))
        self.assertEqual(backbones, snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
