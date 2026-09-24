"""多骨架共享 rotamer 的全枚举选择核心（纯整数运算）。

问题设定
--------
多个（2..3 个）给定骨架**共用同一组 rotamer**：位点数 n ≤ 6、每位点
2..3 个 rotamer。每个骨架 b 有自己的单体能与成对整数能量（``None`` 仍表示
硬冲突）。要为所有骨架选择**同一个构象**（每个位点选哪个 rotamer），
按字典序的两级目标最小化：

1. **最坏骨架能量** ``max_b E_b(a)``（极小极大）；
2. 并列时最小化给定正整数权重下的**加权平均能量**
   ``Σ_b w_b E_b(a) / Σ_b w_b``——分母为常数，整数地比较加权和即可，
   全程不产生浮点。

构象必须在**每个**骨架中都无硬冲突；任一骨架冲突的构象不可行。
规模上限 3^6 = 729 个构象，因此直接全枚举即可给出**已证明**的最优
（不存在“预算耗尽”状态），枚举按 rotamer 下标升序进行，
``itertools.product`` 的产出顺序即构象序列字典序，故两级目标并列时
保留下来的第一个构象就是字典序最小最优构象。
"""

from dataclasses import dataclass
from itertools import product

from .errors import RotamerInputError
from .validation import (
    validate_backbones,
    validate_weights,
    validate_fixed,
)
from .model import SolveStatus

__all__ = [
    "MultiBackboneProblem",
    "BackboneResult",
    "solve_backbones",
]


@dataclass(frozen=True)
class BackboneResult:
    """多骨架全枚举选择的结果。"""

    status: SolveStatus
    """EXACT（已全枚举证明）或 INFEASIBLE（不存在对所有骨架都可行的构象）。"""

    assignment: tuple | None
    """最优共享构象（字典序最小的极小极大最优者）；无解时 None。"""

    scores: tuple | None
    """``scores[b]`` 为所选构象在骨架 b 上的精确总能量（int）；无解时 None。"""

    worst_energy: int | None
    """``max(scores)``——被最小化的第一级目标；无解时 None。"""

    weighted_sum: int | None
    """``Σ w_b·score_b``——加权平均的分子（分母为 ``weight_total``）；
    被最小化的第二级目标；无解时 None。"""

    weights: tuple
    """归一化后的各骨架正整数权重（未指定时为全 1 等权）。"""

    weight_total: int
    """权重之和；加权平均 = weighted_sum / weight_total（精确有理数）。"""

    worst_backbones: tuple
    """达到最坏能量的骨架下标元组（可能并列多个）；无解时为空元组。"""

    conformations_scanned: int
    """枚举的构象总数（笛卡尔积大小，受 fixed 限制后）。"""

    conformations_feasible: int
    """在所有骨架上都无硬冲突的构象数。"""

    @property
    def exact(self) -> bool:
        return self.status is SolveStatus.EXACT

    @property
    def weighted_average(self) -> float:
        """加权平均的浮点表示，仅用于展示；比较与优化全部用整数加权和。"""
        if self.weighted_sum is None:
            raise ValueError("INFEASIBLE：不存在可行构象，没有加权平均")
        return self.weighted_sum / self.weight_total


def _energy_of_spec(spec, assign):
    """构象在一个骨架（validate_problem 归一化结果）上的精确总能量。

    无硬冲突返回 int；任一单体/对能为 None 返回 None。
    """
    _n, _counts, singles, pairs = spec
    total = 0
    for p, r in enumerate(assign):
        v = singles[p][r]
        if v is None:
            return None
        total += v
    for i, j, rows in pairs:
        v = rows[assign[i]][assign[j]]
        if v is None:
            return None
        total += v
    return total


class _CompiledBackbones:
    """编译后的多骨架问题（只读，便于重复求解）。"""

    __slots__ = ("n", "counts", "specs", "n_backbones")

    def __init__(self, specs):
        self.specs = specs
        self.n = specs[0][0]
        self.counts = specs[0][1]
        self.n_backbones = len(specs)


def compile_backbones(backbones):
    """校验并编译多骨架输入。"""
    specs = validate_backbones(backbones)
    return _CompiledBackbones(specs)


def enumerate_backbones(compiled, weights=None, fixed=None):
    """全枚举核心。返回 :class:`BackboneResult`（不经高层封装）。"""
    n = compiled.n
    counts = compiled.counts
    n_backbones = compiled.n_backbones
    specs = compiled.specs

    weights = validate_weights(n_backbones, weights)
    fixed = validate_fixed(n, counts, fixed)
    weight_total = sum(weights)

    domains = [
        (fixed[p],) if p in fixed else tuple(range(counts[p]))
        for p in range(n)
    ]

    scanned = 0
    feasible = 0
    best_key = None       # (worst, weighted_sum)
    best_assign = None
    best_scores = None

    for assign in product(*domains):
        scanned += 1
        scores = []
        ok = True
        for spec in specs:
            e = _energy_of_spec(spec, assign)
            if e is None:
                ok = False
                break
            scores.append(e)
        if not ok:
            continue
        feasible += 1
        scores = tuple(scores)
        worst = max(scores)
        wsum = 0
        for w, e in zip(weights, scores):
            wsum += w * e
        key = (worst, wsum)
        # 枚举顺序即字典序：严格更优才替换，并列时保留先出现（字典序更小）者。
        if best_key is None or key < best_key:
            best_key = key
            best_assign = assign
            best_scores = scores

    if best_assign is None:
        return BackboneResult(
            status=SolveStatus.INFEASIBLE,
            assignment=None,
            scores=None,
            worst_energy=None,
            weighted_sum=None,
            weights=weights,
            weight_total=weight_total,
            worst_backbones=(),
            conformations_scanned=scanned,
            conformations_feasible=0,
        )

    worst_idx = tuple(b for b in range(n_backbones)
                      if best_scores[b] == best_key[0])
    return BackboneResult(
        status=SolveStatus.EXACT,
        assignment=tuple(best_assign),
        scores=best_scores,
        worst_energy=best_key[0],
        weighted_sum=best_key[1],
        weights=weights,
        weight_total=weight_total,
        worst_backbones=worst_idx,
        conformations_scanned=scanned,
        conformations_feasible=feasible,
    )


class MultiBackboneProblem:
    """可重复求解的多骨架共享 rotamer 问题（编译一次，缓存复用）。

    参数
    ----
    backbones: 2..3 个骨架组成的序列，每项为
        ``{"singles": ..., "pairs": ...}``（键名兼容单骨架接口的
        ``"single_energies"`` / ``"pair_matrices"``）。各骨架位点数与
        逐位点 rotamer 数必须完全一致（共用同一组 rotamer），
        且 n ≤ 6、每位点 2..3 个 rotamer。
    """

    def __init__(self, backbones):
        self._compiled = compile_backbones(backbones)
        self.n = self._compiled.n
        self.counts = self._compiled.counts
        self.n_backbones = self._compiled.n_backbones

    def solve(self, weights=None, *, fixed=None) -> BackboneResult:
        """全枚举求极小极大（并列时加权平均最小、再并列取字典序最小）。"""
        return enumerate_backbones(self._compiled, weights=weights, fixed=fixed)

    def scores_of(self, assignment) -> tuple:
        """计算给定完整构象在每个骨架上的精确能量。

        返回长度为骨架数的 int 元组；任一骨架遇硬冲突/下标非法抛
        :class:`RotamerInputError`（不返回半组分数）。
        """
        n = self._compiled.n
        counts = self._compiled.counts
        if not isinstance(assignment, (list, tuple)):
            raise RotamerInputError(
                f"assignment: 必须是 list/tuple, 得到 {type(assignment).__name__}"
            )
        if len(assignment) != n:
            raise RotamerInputError(
                f"assignment: 长度必须等于位点数 {n}, 实际 {len(assignment)}"
            )
        for p, r in enumerate(assignment):
            if isinstance(r, bool) or not isinstance(r, int):
                raise RotamerInputError(
                    f"assignment[{p}]: rotamer 必须是 int（bool 不接受）, "
                    f"得到 {type(r).__name__}"
                )
            if not (0 <= r < counts[p]):
                raise RotamerInputError(
                    f"assignment[{p}]: rotamer {r} 越界, "
                    f"合法范围 0..{counts[p] - 1}"
                )
        scores = []
        for b, spec in enumerate(self._compiled.specs):
            e = _energy_of_spec(spec, assignment)
            if e is None:
                raise RotamerInputError(
                    f"assignment: 构象 {tuple(assignment)} 在骨架 {b} 上存在硬冲突"
                )
            scores.append(e)
        return tuple(scores)


def solve_backbones(backbones, weights=None, *, fixed=None) -> BackboneResult:
    """一次性便捷接口：校验多骨架输入并全枚举求共享构象。

    ``weights`` 为 ``None`` 时各骨架等权；否则必须是与骨架等长的正整数
    序列（拒绝 bool/float/负数）。``fixed`` 可给 ``{position: rotamer}``
    附加共享固定约束。
    """
    compiled = compile_backbones(backbones)
    return enumerate_backbones(compiled, weights=weights, fixed=fixed)
