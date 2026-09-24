"""多个给定骨架共用一组 rotamer 时的离散能量选择。

场景
----
同一条肽链（同样的位点、每点同一组候选 rotamer）要同时放到若干**给定骨架**
上评估：每个骨架各自有一套单体能与成对位置能（整数或 ``None`` 硬冲突，
口径与 :mod:`rotamer_select.solver` 相同）。需要选出**一个共用构象**，
按字典序优先级优化两个目标：

1. **最小化最坏骨架能量**（minimax）：``min_a max_b E_b(a)``；
2. 最坏能量并列时，**最小化给定权重的加权平均能量**
   ``sum_b w_b * E_b(a) / sum_b w_b``（分母为常数，实现中按整数
   加权和比较，不引入浮点）；
3. 两者都并列时，取 rotamer 下标序列**字典序最小**的构象。

规模边界（题面限定的可枚举范围）：位点数 1..6、每位点 1..3 个 rotamer、
骨架数 1..3。总构象数至多 3^6 = 729，直接全枚举即可给出精确结果与
每骨架得分——本模块不做分支限界、预算或恢复之类的扩展。

输入
----
``backbones`` 为每骨架一个 ``(single_energies, pair_matrices)`` 二元组的
序列，两套能量的格式与单骨架接口完全一致；所有骨架必须有相同的位点数
与每位点 rotamer 数（“共用一组 rotamer”）。任一骨架上为硬冲突的构象
即整体不可行。
"""

from dataclasses import dataclass
from enum import Enum
from itertools import product

from .errors import RotamerInputError
from .validation import (
    validate_problem,
    _is_real_int,
    MULTI_MAX_POSITIONS,
    MULTI_MIN_ROTAMERS,
    MULTI_MAX_ROTAMERS,
    MAX_BACKBONES,
)

__all__ = [
    "MULTI_MAX_POSITIONS",
    "MULTI_MIN_ROTAMERS",
    "MULTI_MAX_ROTAMERS",
    "MAX_BACKBONES",
    "BackboneSolveStatus",
    "BackboneSelection",
    "MultiBackboneProblem",
    "select_for_backbones",
]


class BackboneSolveStatus(str, Enum):
    """多骨架枚举的结束状态。"""

    EXACT = "exact"
    """全枚举完整结束：最坏能量最优性（含两级并列规则）已证明。"""

    INFEASIBLE = "infeasible"
    """枚举了全部构象：不存在对所有骨架都无硬冲突的共用构象。"""


@dataclass(frozen=True)
class BackboneSelection:
    """多骨架共用构象选择结果。"""

    status: BackboneSolveStatus
    assignment: tuple | None
    """最优共用构象（两级并列后取字典序最小）；无解时为 None。"""
    scores: tuple | None
    """每个骨架在 ``assignment`` 下的总能量，按输入顺序；无解时为 None。"""
    worst_energy: int | None
    """``max(scores)``：被最小化的最坏骨架能量；无解时为 None。"""
    weighted_sum: int | None
    """``sum(weights[b] * scores[b])``；加权平均 = 它 / ``sum(weights)``，
    分母为常数故最小化平均等价于最小化此整数和；无解时为 None。"""
    weights: tuple
    """使用的权重（未给权重时为全 1）。"""
    assignments_evaluated: int
    """枚举的构象总数（含在某个骨架上硬冲突的不可行构象）。"""
    feasible_assignments: int
    """对所有骨架都无硬冲突的构象数。"""

    @property
    def exact(self) -> bool:
        return self.status is BackboneSolveStatus.EXACT

    @property
    def weighted_average(self):
        """加权平均（分数）；仅在可行时有值。保留属性以便展示，比较用整数和。"""
        if self.weighted_sum is None:
            return None
        return self.weighted_sum / sum(self.weights)


class MultiBackboneProblem:
    """一个可重复评估的多骨架问题（校验/归一化一次后缓存）。

    参数
    ----
    backbones: 1..3 个 ``(single_energies, pair_matrices)``，每个骨架的
        能量格式与 :class:`rotamer_select.RotamerProblem` 相同；所有骨架
        必须有相同位点数与每位点 rotamer 数。
    """

    def __init__(self, backbones):
        # _backbones = (n, counts, tuple[(singles, pair_map), ...])
        self._backbones = _compile_backbones(backbones)
        self.n = self._backbones[0]
        self.counts = self._backbones[1]
        self.backbone_count = len(self._backbones[2])

    def select(self, weights=None) -> BackboneSelection:
        """全枚举求最优共用构象，见 :func:`select_for_backbones`。"""
        return _enumerate_select(self._backbones, weights)

    def scores_of(self, assignment) -> tuple:
        """计算给定构象在每骨架下的总能量。

        任一骨架上遇硬冲突抛 :class:`RotamerInputError`（指明骨架下标与
        位置对）；下标越界/类型错误同样拒绝。
        """
        return _exact_scores(self._backbones, assignment)


def select_for_backbones(backbones, weights=None) -> BackboneSelection:
    """一次性便捷接口：校验多骨架输入并全枚举求最优共用构象。"""
    compiled = _compile_backbones(backbones)
    return _enumerate_select(compiled, weights)


# --------------------------------------------------------------------------- #
# 校验与编译
# --------------------------------------------------------------------------- #


def _compile_backbones(backbones):
    """校验所有骨架并归一化为 ``(n, counts, tuple[(singles, pair_map), ...])``。

    任一输入不合法都在任何计算之前抛 :class:`RotamerInputError`；
    本函数不持有可变状态，调用方也不会得到半成品。
    """
    if not isinstance(backbones, (list, tuple)):
        raise RotamerInputError(
            f"backbones: 必须是 list/tuple 的骨架序列, 得到 "
            f"{type(backbones).__name__}"
        )
    b = len(backbones)
    if not (1 <= b <= MAX_BACKBONES):
        raise RotamerInputError(
            f"backbones: 骨架数必须在 1..{MAX_BACKBONES}, 实际 {b}"
        )

    common_n = None
    common_counts = None
    compiled = []
    for idx, item in enumerate(backbones):
        where = f"backbones[{idx}]"
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RotamerInputError(
                f"{where}: 每个骨架必须是 (single_energies, pair_matrices) 二元组"
            )
        singles_raw, pairs_raw = item
        n, counts, singles, pairs = validate_problem(
            singles_raw, pairs_raw,
            min_rotamers=MULTI_MIN_ROTAMERS,
            max_positions=MULTI_MAX_POSITIONS,
            max_rotamers=MULTI_MAX_ROTAMERS,
        )
        if common_n is None:
            common_n, common_counts = n, counts
        elif n != common_n or counts != common_counts:
            raise RotamerInputError(
                f"{where}: 所有骨架必须共用同一组 rotamer：骨架 0 为 "
                f"{common_n} 位点、每点 rotamer 数 {tuple(common_counts)}, "
                f"骨架 {idx} 为 {n} 位点、{tuple(counts)}"
            )
        pair_map = {(a, bb): rows for a, bb, rows in pairs}
        compiled.append((singles, pair_map))

    return common_n, common_counts, tuple(compiled)


def _validate_weights(b, weights):
    """校验权重：None 表示等权；否则要求 b 个真正的正整数（拒绝 bool）。"""
    if weights is None:
        return (1,) * b
    if not isinstance(weights, (list, tuple)):
        raise RotamerInputError(
            f"weights: 必须是 list/tuple 或 None, 得到 {type(weights).__name__}"
        )
    if len(weights) != b:
        raise RotamerInputError(
            f"weights: 权重数 {len(weights)} 与骨架数 {b} 不一致"
        )
    out = []
    for i, w in enumerate(weights):
        if isinstance(w, bool) or not isinstance(w, int):
            raise RotamerInputError(
                f"weights[{i}]: 权重必须是正整数（bool 不接受）, "
                f"得到 {type(w).__name__}"
            )
        if w <= 0:
            raise RotamerInputError(
                f"weights[{i}]: 权重必须为正整数, 得到 {w}"
            )
        out.append(w)
    return tuple(out)


def _validate_assignment(n, counts, assignment):
    if not isinstance(assignment, (list, tuple)):
        raise RotamerInputError(
            f"assignment: 必须是 list/tuple, 得到 {type(assignment).__name__}"
        )
    if len(assignment) != n:
        raise RotamerInputError(
            f"assignment: 长度必须等于位点数 {n}, 实际 {len(assignment)}"
        )
    for p, r in enumerate(assignment):
        if isinstance(r, bool) or not _is_real_int(r):
            raise RotamerInputError(
                f"assignment[{p}]: rotamer 必须是 int（bool 不接受）, "
                f"得到 {type(r).__name__}"
            )
        if not (0 <= r < counts[p]):
            raise RotamerInputError(
                f"assignment[{p}]: rotamer {r} 越界, 合法范围 0..{counts[p] - 1}"
            )
    return tuple(assignment)


# --------------------------------------------------------------------------- #
# 能量与枚举
# --------------------------------------------------------------------------- #


def _score_one(singles, pair_map, assign):
    """一个骨架上的构象能量；单体或对能为 None（硬冲突）时返回 None。"""
    total = 0
    for p, r in enumerate(assign):
        v = singles[p][r]
        if v is None:
            return None
        total += v
    for (a, b), rows in pair_map.items():
        v = rows[assign[a]][assign[b]]
        if v is None:
            return None
        total += v
    return total


def _exact_scores(compiled, assignment):
    """指定构象在每骨架上的能量；硬冲突或下标非法时抛 RotamerInputError。"""
    n, counts, backbones = compiled
    assign = _validate_assignment(n, counts, assignment)
    scores = []
    for b, (singles, pair_map) in enumerate(backbones):
        total = 0
        for p, r in enumerate(assign):
            v = singles[p][r]
            if v is None:
                raise RotamerInputError(
                    f"assignment[{p}]={r}: 骨架 {b} 上该 rotamer "
                    f"单体能为硬冲突 None"
                )
            total += v
        for (a, c), rows in pair_map.items():
            v = rows[assign[a]][assign[c]]
            if v is None:
                raise RotamerInputError(
                    f"assignment: 骨架 {b} 上位置对 ({a}, {c}) 取 "
                    f"({assign[a]}, {assign[c]}) 为硬冲突 None"
                )
            total += v
        scores.append(total)
    return tuple(scores)


def _enumerate_select(compiled, weights) -> BackboneSelection:
    n, counts, backbones = compiled
    weights = _validate_weights(len(backbones), weights)

    total = 1
    for k in counts:
        total *= k

    # itertools.product 按位点 0..n-1 产出，枚举顺序本身即构象字典序；
    # 因此只在目标严格更优时更新，保留下来的并列最优自动是字典序最小者。
    best_assign = None
    best_scores = None
    best_worst = None
    best_wsum = None
    feasible = 0

    for assign in product(*(range(k) for k in counts)):
        scores = [_score_one(singles, pair_map, assign)
                  for singles, pair_map in backbones]
        if any(s is None for s in scores):
            continue
        feasible += 1
        worst = max(scores)
        wsum = sum(w * s for w, s in zip(weights, scores))
        if (best_worst is None
                or worst < best_worst
                or (worst == best_worst and wsum < best_wsum)):
            best_assign = assign
            best_scores = tuple(scores)
            best_worst = worst
            best_wsum = wsum

    if best_assign is None:
        return BackboneSelection(
            status=BackboneSolveStatus.INFEASIBLE,
            assignment=None, scores=None, worst_energy=None,
            weighted_sum=None, weights=weights,
            assignments_evaluated=total, feasible_assignments=0,
        )
    return BackboneSelection(
        status=BackboneSolveStatus.EXACT,
        assignment=tuple(best_assign),
        scores=best_scores,
        worst_energy=best_worst,
        weighted_sum=best_wsum,
        weights=weights,
        assignments_evaluated=total,
        feasible_assignments=feasible,
    )
