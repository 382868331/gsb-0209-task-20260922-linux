"""多骨架共享 rotamer 全枚举选择的测试（本迭代）。

期望值来源：
- tests/reference.py 的独立多骨架全枚举 ``brute_backbones``
  （朴素求和，不导入 rotamer_select）；
- 手工构造的极小极大/加权平均/字典序并列例子；
- 解析关系（加权和、最坏骨架下标、枚举构象数）。
与 ``MultiBackboneProblem.scores_of`` 的往返一致只作为补充断言。
"""

import unittest

from rotamer_select import (
    solve_backbones,
    MultiBackboneProblem,
    BackboneResult,
    RotamerInputError,
    SolveStatus,
    MULTI_MAX_POSITIONS,
    MULTI_MAX_ROTAMERS,
    MIN_BACKBONES,
    MAX_BACKBONES,
)
from reference import brute_backbones
from datagen import generate_backbones


class TestAgainstBruteBackbones(unittest.TestCase):
    """1..6 位点、2..3 骨架的随机实例：对独立全枚举，等权/加权/固定全覆盖。"""

    def test_equal_weights_matches_brute(self):
        cases = 0
        for n in range(1, MULTI_MAX_POSITIONS + 1):
            for n_b in (MIN_BACKBONES, MAX_BACKBONES):
                for seed in range(8):
                    with self.subTest(n=n, n_b=n_b, seed=seed):
                        bbs, counts = generate_backbones(n, n_b, seed=seed)
                        res = solve_backbones(bbs)
                        ba, bs, bw, bwsum = brute_backbones(bbs)
                        if ba is None:
                            self.assertIs(res.status, SolveStatus.INFEASIBLE)
                            self.assertIsNone(res.assignment)
                            self.assertIsNone(res.scores)
                            self.assertIsNone(res.worst_energy)
                            self.assertEqual(res.conformations_feasible, 0)
                        else:
                            cases += 1
                            self.assertIs(res.status, SolveStatus.EXACT)
                            self.assertEqual(tuple(res.assignment), tuple(ba))
                            self.assertEqual(tuple(res.scores), tuple(bs))
                            self.assertEqual(res.worst_energy, bw)
                            self.assertEqual(res.weighted_sum, bwsum)
                            # 解析关系：最坏能量 = max(scores)
                            self.assertEqual(res.worst_energy, max(res.scores))
                            self.assertEqual(res.weighted_sum, sum(res.scores))
                            self.assertEqual(res.weight_total, n_b)
        self.assertGreater(cases, 50)  # 确实核到大量可行实例

    def test_random_weights_match_brute(self):
        for n in range(1, 7):
            for n_b in (2, 3):
                for seed in range(6):
                    with self.subTest(n=n, n_b=n_b, seed=seed):
                        bbs, _ = generate_backbones(n, n_b, seed=300 + seed)
                        # 确定性地取一组非平凡正整数权重
                        weights = tuple(1 + ((seed * 3 + n + b * 2) % 4)
                                        for b in range(n_b))
                        res = solve_backbones(bbs, weights=weights)
                        ba, bs, bw, bwsum = brute_backbones(bbs, weights=weights)
                        if ba is None:
                            self.assertIs(res.status, SolveStatus.INFEASIBLE)
                            continue
                        self.assertEqual(tuple(res.assignment), tuple(ba))
                        self.assertEqual(res.weighted_sum, bwsum)
                        self.assertEqual(res.worst_energy, bw)
                        self.assertEqual(res.weights, weights)
                        self.assertEqual(res.weight_total, sum(weights))

    def test_fixed_matches_brute(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                bbs, counts = generate_backbones(4, 2, seed=600 + seed)
                p = seed % 4
                r = seed % counts[p]
                res = solve_backbones(bbs, fixed={p: r})
                ba, bs, bw, bwsum = brute_backbones(bbs, fixed={p: r})
                if ba is None:
                    self.assertIs(res.status, SolveStatus.INFEASIBLE)
                else:
                    self.assertIs(res.status, SolveStatus.EXACT)
                    self.assertEqual(res.assignment[p], r)
                    self.assertEqual(tuple(res.assignment), tuple(ba))
                    self.assertEqual(res.worst_energy, bw)
                    self.assertEqual(res.weighted_sum, bwsum)


class TestEnumerationCounts(unittest.TestCase):
    def test_scanned_is_full_cartesian_product(self):
        bbs, counts = generate_backbones(6, 3, k_range=(3, 3), seed=11,
                                         none_prob=0.0, single_none_prob=0.0)
        res = solve_backbones(bbs)
        self.assertIs(res.status, SolveStatus.EXACT)
        self.assertEqual(len(res.assignment), 6)
        # 上限规模：3^6 = 729
        self.assertEqual(res.conformations_scanned, 729)
        self.assertEqual(res.conformations_feasible, 729)
        self.assertLessEqual(res.conformations_feasible,
                             res.conformations_scanned)

    def test_scanned_with_fixed(self):
        bbs, _ = generate_backbones(3, 2, k_range=(2, 3), seed=12,
                                    none_prob=0.0, single_none_prob=0.0)
        res = solve_backbones(bbs, fixed={0: 1})
        # 位点 0 固定，其余 2 位点各 2..3 选项
        counts = [len(c) for c in bbs[0]["singles"]]
        expected = counts[1] * counts[2]
        self.assertEqual(res.conformations_scanned, expected)
        self.assertEqual(res.conformations_feasible, expected)

    def test_single_position_two_backbones(self):
        # 边界 n=1：只有单体能，2 个骨架。
        b0 = {"singles": [[3, -1, 2]], "pairs": []}
        b1 = {"singles": [[-2, 4, 0]], "pairs": []}
        res = solve_backbones([b0, b1])
        # r0: scores(3,-2) worst 3；r1:(-1,4) worst 4；r2:(2,0) worst 2
        # 极小极大最优是 r2 (worst=2)
        self.assertEqual(res.assignment, (2,))
        self.assertEqual(res.scores, (2, 0))
        self.assertEqual(res.worst_energy, 2)
        self.assertEqual(res.weighted_sum, 2)


class TestObjectiveHandbuilt(unittest.TestCase):
    def test_minimax_beats_better_average(self):
        # 单位点：r0 得分 (0,100) 平均 50 但最坏 100；
        # r1 得分 (60,60) 平均 60 但最坏 60。先最小化最坏 → 必须选 r1，
        # 即使它的等权平均更差——证明两级目标的顺序。
        b0 = {"singles": [[0, 60]], "pairs": []}
        b1 = {"singles": [[100, 60]], "pairs": []}
        res = solve_backbones([b0, b1])
        self.assertEqual(res.assignment, (1,))
        self.assertEqual(res.scores, (60, 60))
        self.assertEqual(res.worst_energy, 60)
        self.assertEqual(res.weighted_sum, 120)
        self.assertEqual(res.worst_backbones, (0, 1))

    def test_secondary_weighted_average_breaks_worst_tie(self):
        # (0,0) 得分 (0,5)；(1,1) 得分 (5,0)，最坏都是 5。
        b0 = {"singles": [[0, 10], [0, 10]],
              "pairs": [(0, 1, [[0, 0], [0, -15]])]}
        b1 = {"singles": [[10, 0], [10, 0]],
              "pairs": [(0, 1, [[-15, 0], [0, 0]])]}
        # 等权：加权和都是 5 → 落到字典序 → (0,0)
        r_eq = solve_backbones([b0, b1])
        self.assertEqual(r_eq.worst_energy, 5)
        self.assertEqual(r_eq.assignment, (0, 0))
        self.assertEqual(r_eq.weighted_sum, 5)
        # 权重 (1,2)：(0,0) 加权和 10，(1,1) 加权和 5 → 改选 (1,1)
        r_w = solve_backbones([b0, b1], weights=(1, 2))
        self.assertEqual(r_w.assignment, (1, 1))
        self.assertEqual(r_w.scores, (5, 0))
        self.assertEqual(r_w.weighted_sum, 5)
        self.assertEqual(r_w.weight_total, 3)
        # 解析关系：加权平均 = weighted_sum / weight_total
        self.assertAlmostEqual(r_w.weighted_average, 5 / 3)

    def test_tertiary_lexicographic_on_full_tie(self):
        # 所有能量为 0：每个构象得分向量完全相同，必须取字典序最小 (0,0,0)。
        bbs = [
            {"singles": [[0, 0], [0, 0, 0], [0, 0]], "pairs": []}
            for _ in range(3)
        ]
        res = solve_backbones(bbs)
        self.assertEqual(res.assignment, (0, 0, 0))
        self.assertEqual(res.scores, (0, 0, 0))
        self.assertEqual(res.worst_backbones, (0, 1, 2))
        self.assertEqual(res.conformations_feasible, 2 * 3 * 2)

    def test_pair_energies_participate_in_minimax(self):
        # 两骨架单位点对 (0,1)：单体倾向一致为 0，但负对能只在不同骨架出现，
        # 共享选择必须兼顾两边。
        b0 = {"singles": [[0, 0], [0, 0]],
              "pairs": [(0, 1, [[0, 0], [0, -8]])]}
        b1 = {"singles": [[0, 0], [0, 0]],
              "pairs": [(0, 1, [[-8, 0], [0, 0]])]}
        res = solve_backbones([b0, b1])
        # (0,0):(0,-8) worst 0; (0,1):(0,0) worst 0; (1,0):(0,0) worst 0;
        # (1,1):(-8,0) worst 0。全部最坏 0；等权和：-8,0,0,-8
        # → 加权和 -8 并列 (0,0)/(1,1) → 字典序 (0,0)
        self.assertEqual(res.worst_energy, 0)
        self.assertEqual(res.assignment, (0, 0))
        self.assertEqual(res.scores, (0, -8))

    def test_worst_backbones_indices(self):
        b0 = {"singles": [[5, 1]], "pairs": []}
        b1 = {"singles": [[2, 9]], "pairs": []}
        b2 = {"singles": [[5, 0]], "pairs": []}
        res = solve_backbones([b0, b1, b2])
        # r0: (5,2,5) worst 5; r1: (1,9,0) worst 9 → r0，最坏骨架 0 与 2
        self.assertEqual(res.assignment, (0,))
        self.assertEqual(res.worst_energy, 5)
        self.assertEqual(res.worst_backbones, (0, 2))


class TestInfeasible(unittest.TestCase):
    def test_no_jointly_feasible_conformation(self):
        # 骨架 1 唯一无冲突构象是 (1,1)，骨架 0 恰好对 (1,1) 硬冲突。
        b0 = {"singles": [[0, 0], [0, 0]],
              "pairs": [(0, 1, [[0, 0], [0, None]])]}
        b1 = {"singles": [[10, 0], [10, 0]],
              "pairs": [(0, 1, [[None, None], [None, 0]])]}
        res = solve_backbones([b0, b1])
        self.assertIs(res.status, SolveStatus.INFEASIBLE)
        self.assertIsNone(res.assignment)
        self.assertIsNone(res.scores)
        self.assertIsNone(res.worst_energy)
        self.assertIsNone(res.weighted_sum)
        self.assertEqual(res.worst_backbones, ())
        self.assertEqual(res.conformations_scanned, 4)
        self.assertEqual(res.conformations_feasible, 0)
        self.assertFalse(res.exact)
        # INFEASIBLE 下取加权平均必须明确报错，而不是给出误导数值
        with self.assertRaises(ValueError):
            res.weighted_average

    def test_infeasible_under_fixed(self):
        # 全体可行，但固定位点 0=1 后没有任何构象同时满足两骨架。
        b0 = {"singles": [[0, 0], [0, 0]],
              "pairs": [(0, 1, [[0, 0], [0, None]])]}
        b1 = {"singles": [[0, 0], [0, 0]],
              "pairs": [(0, 1, [[None, 0], [None, None]])]}
        free = solve_backbones([b0, b1])
        self.assertIs(free.status, SolveStatus.EXACT)
        fixed = solve_backbones([b0, b1], fixed={0: 1})
        self.assertIs(fixed.status, SolveStatus.INFEASIBLE)


class TestProblemClassAPI(unittest.TestCase):
    def setUp(self):
        self.b0 = {"singles": [[0, 10], [0, 10]],
                   "pairs": [(0, 1, [[0, 0], [0, -15]])]}
        self.b1 = {"singles": [[10, 0], [10, 0]],
                   "pairs": [(0, 1, [[-15, 0], [0, 0]])]}

    def test_compile_once_repeat_solve(self):
        prob = MultiBackboneProblem([self.b0, self.b1])
        self.assertEqual(prob.n, 2)
        self.assertEqual(prob.counts, (2, 2))
        self.assertEqual(prob.n_backbones, 2)
        r1 = prob.solve()
        r2 = prob.solve(weights=(1, 2))
        self.assertEqual(r1.assignment, (0, 0))
        self.assertEqual(r2.assignment, (1, 1))
        # 同一对象重复求解互不影响
        r3 = prob.solve()
        self.assertEqual(r3.assignment, (0, 0))

    def test_scores_of_roundtrip(self):
        prob = MultiBackboneProblem([self.b0, self.b1])
        res = prob.solve()
        # 补充断言：最优构象的逐骨架得分与枚举结果一致
        self.assertEqual(prob.scores_of(res.assignment), res.scores)
        self.assertEqual(prob.scores_of((1, 1)), (5, 0))

    def test_scores_of_rejects_collision_and_bad_index(self):
        b0 = {"singles": [[0, 0], [0, 0]],
              "pairs": [(0, 1, [[0, 0], [0, None]])]}
        b1 = {"singles": [[0, 0], [0, 0]], "pairs": []}
        prob = MultiBackboneProblem([b0, b1])
        with self.assertRaises(RotamerInputError):
            prob.scores_of((1, 1))      # 骨架 0 硬冲突
        with self.assertRaises(RotamerInputError):
            prob.scores_of((0,))        # 长度不符
        with self.assertRaises(RotamerInputError):
            prob.scores_of((0, 5))      # rotamer 越界
        with self.assertRaises(RotamerInputError):
            prob.scores_of((0, True))   # bool 冒充

    def test_compatible_single_backbone_key_names(self):
        b0 = {"single_energies": [[1, 2]], "pair_matrices": []}
        b1 = {"single_energies": [[3, 0]], "pair_matrices": []}
        res = solve_backbones([b0, b1])
        self.assertEqual(res.assignment, (1,))  # (2,0) worst 2 优于 (1,3) worst 3

    def test_solve_does_not_mutate_inputs(self):
        bbs, _ = generate_backbones(3, 2, seed=42)
        import copy
        snapshot = copy.deepcopy(bbs)
        solve_backbones(bbs, weights=(2, 1))
        self.assertEqual(bbs, snapshot)


class TestMultiValidation(unittest.TestCase):
    def _ok(self):
        return {"singles": [[0, 0], [0, 0]], "pairs": []}

    def test_backbone_count_out_of_range(self):
        ok = self._ok()
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok])
        self.assertIn("骨架数 1", str(cm.exception))
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok, ok, ok, ok])
        self.assertIn("骨架数 4", str(cm.exception))

    def test_not_sequence_or_not_dict(self):
        ok = self._ok()
        with self.assertRaises(RotamerInputError):
            solve_backbones("xx")
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok, "not-a-dict"])
        self.assertIn("backbones[1]", str(cm.exception))

    def test_missing_singles_key(self):
        ok = self._ok()
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok, {"pairs": []}])
        self.assertIn("backbones[1]", str(cm.exception))
        self.assertIn("singles", str(cm.exception))

    def test_shape_must_match(self):
        ok = self._ok()
        other_n = {"singles": [[0, 0], [0, 0], [0, 0]], "pairs": []}
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok, other_n])
        self.assertIn("共用同一组 rotamer", str(cm.exception))
        other_k = {"singles": [[0, 0, 0], [0, 0]], "pairs": []}
        with self.assertRaises(RotamerInputError):
            solve_backbones([ok, other_k])

    def test_multi_limits_are_tighter_than_single(self):
        # 单骨架接口允许的 n=7..12 在多骨架接口被拒（题面限定 n<=6）
        big_n = {"singles": [[0, 0]] * (MULTI_MAX_POSITIONS + 1), "pairs": []}
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([big_n, big_n])
        self.assertIn("多骨架位点数 7", str(cm.exception))
        # 每点 4 rotamer 超过多骨架上限 3
        big_k = {"singles": [[0, 0, 0, 0], [0, 0]], "pairs": []}
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([big_k, big_k])
        self.assertIn("rotamer 数 4", str(cm.exception))

    def test_inner_energy_error_is_located_to_backbone(self):
        bad_nan = {"singles": [[0, float("nan")], [0, 0]], "pairs": []}
        ok = self._ok()
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok, bad_nan])
        self.assertIn("backbones[1]", str(cm.exception))
        self.assertIn("NaN", str(cm.exception))
        bad_bool = {"singles": [[0, True], [0, 0]], "pairs": []}
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok, bad_bool])
        self.assertIn("backbones[1]", str(cm.exception))
        bad_matrix = {"singles": [[0, 0], [0, 0]],
                      "pairs": [(0, 1, [[0, 0]])]}  # 维度错误
        with self.assertRaises(RotamerInputError) as cm:
            solve_backbones([ok, bad_matrix])
        self.assertIn("backbones[1]", str(cm.exception))

    def test_weights_validation(self):
        bbs = [self._ok(), self._ok(), self._ok()]
        for bad in ((1, 1), (1, 1, 0), (-1, 1, 1), (1.0, 1, 1),
                    (True, 1, 1), (1, "1", 1), None):
            if bad is None:
                continue  # None 合法（等权）
            with self.assertRaises(RotamerInputError):
                solve_backbones(bbs, weights=bad)
        # None 等权合法
        res = solve_backbones(bbs, weights=None)
        self.assertEqual(res.weights, (1, 1, 1))

    def test_fixed_validation(self):
        ok = self._ok()
        with self.assertRaises(RotamerInputError):
            solve_backbones([ok, ok], fixed={2: 0})
        with self.assertRaises(RotamerInputError):
            solve_backbones([ok, ok], fixed={0: 2})

    def test_result_is_frozen(self):
        res = solve_backbones([self._ok(), self._ok()])
        with self.assertRaises(Exception):
            res.assignment = (9, 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
