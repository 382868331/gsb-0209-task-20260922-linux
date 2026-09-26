"""公开数据结构与高层接口。

数值口径
--------
所有能量、下界、代价增量均为整数，求解过程不使用浮点，因此不存在浮点误差；
“精确”指分支限界完整跑完（未触发预算）后由剪枝正确性保证的整数最优。

预算语义
--------
- ``SolveStatus.EXACT``：已证明最优（或在 ``INFEASIBLE`` 情况下证明无解）。
- ``SolveStatus.INFEASIBLE``：证明不存在无硬冲突构象——与预算耗尽严格分开。
- ``SolveStatus.BUDGET_EXHAUSTED``：预算用完仍未证明；此时给出当前找到的最优
  可行构象（可能为 ``None``）与一个可信的全局下界
  ``global_lower_bound <= 真实最优 <= best.energy``，绝不声称精确。
"""

from dataclasses import dataclass
from enum import Enum

from .errors import RotamerInputError
from .solver import (
    compile_problem,
    branch_and_bound,
    exact_energy,
    DEFAULT_NODE_BUDGET,
)
from .validation import validate_budget, validate_fixed

__all__ = [
    "DEFAULT_NODE_BUDGET",
    "SolveStatus",
    "SolveResult",
    "FixedCostStatus",
    "FixedCostEntry",
    "AnalysisResult",
    "RotamerProblem",
    "solve",
    "optimize",
    "energy_of",
    "validate_allowed",
]


class SolveStatus(str, Enum):
    """单次搜索的结束状态。"""

    EXACT = "exact"
    """已找到并证明最优构象（下界与最优重合 / 栈被剪空）。"""

    INFEASIBLE = "infeasible"
    """已证明不存在任何无硬冲突构象。"""

    BUDGET_EXHAUSTED = "budget_exhausted"
    """节点预算耗尽，未证明最优；结果只给当前最优与可信全局下界。"""


class FixedCostStatus(str, Enum):
    """“固定某位置取某个不同 rotamer”子问题的状态。"""

    OPTIMAL = "optimal"
    """该约束下的最优代价增量已证明。"""

    BEST_KNOWN = "best_known"
    """预算耗尽：给出的是已找到的最好值（上界）与单独的增量下界。"""

    INFEASIBLE = "infeasible"
    """已证明该固定约束下不存在无冲突构象（代价增量无定义）。"""


@dataclass(frozen=True)
class SolveResult:
    """单次分支限界搜索的结果。"""

    status: SolveStatus
    assignment: tuple | None
    """字典序最小的最优构象（EXACT）；预算耗尽时为当前最优，可能为 None。"""
    energy: int | None
    global_lower_bound: int | None
    """真实最优总能量的可信下界；INFEASIBLE 时为 None。"""
    nodes_expanded: int
    budget: int

    @property
    def exact(self) -> bool:
        """True 当且仅当最优性已被证明。"""
        return self.status is SolveStatus.EXACT

    @property
    def proven_infeasible(self) -> bool:
        return self.status is SolveStatus.INFEASIBLE


@dataclass(frozen=True)
class FixedCostEntry:
    """固定位置 ``position`` 取 ``rotamer``（不同于最优 rotamer）时的结果。"""

    position: int
    rotamer: int
    status: FixedCostStatus
    minimal_energy: int | None
    """该固定约束下的最小总能量；INFEASIBLE 时为 None。"""
    delta: int | None
    """``minimal_energy - 最优总能量``（代价增量上界）；INFEASIBLE 时为 None。"""
    lower_bound_delta: int | None
    """仅 BEST_KNOWN 时给出：真实代价增量的整数下界。"""
    nodes_expanded: int


@dataclass(frozen=True)
class AnalysisResult:
    """完整分析：最优、次优、逐位置固定代价增量。"""

    status: SolveStatus
    """聚合状态：任一子搜索预算耗尽则为 BUDGET_EXHAUSTED。"""
    assignment: tuple | None
    energy: int | None
    global_lower_bound: int | None
    nodes_expanded: int
    """所有子搜索展开节点数之和。"""
    budget_per_run: int
    second_status: SolveStatus
    second_assignment: tuple | None
    """第二个不同构象（与最优构象不同的能量最小者，并列取字典序最小）。"""
    second_energy: int | None
    fixed_costs: tuple
    """``fixed_costs[p]`` 是长度 k_p 的元组；最优 rotamer 位置为 None，
    其余为 :class:`FixedCostEntry`。"""

    @property
    def exact(self) -> bool:
        return self.status is SolveStatus.EXACT


_STATUS_MAP = {
    "exact": SolveStatus.EXACT,
    "infeasible": SolveStatus.INFEASIBLE,
    "budget_exhausted": SolveStatus.BUDGET_EXHAUSTED,
}


def _to_result(raw: dict) -> SolveResult:
    return SolveResult(
        status=_STATUS_MAP[raw["status"]],
        assignment=raw["assignment"],
        energy=raw["energy"],
        global_lower_bound=raw["global_lower_bound"],
        nodes_expanded=raw["nodes_expanded"],
        budget=raw["budget"],
    )


class RotamerProblem:
    """一个可重复求解的 rotamer 能量问题（编译后缓存）。

    参数
    ----
    single_energies: 每位置一个长度 2..5 的序列，元素 int 或 None。
    pair_matrices: 若干 ``(i, j, matrix)``，必须满足 ``i < j``，
        matrix 为 k_i × k_j 的 list/tuple 矩形（行对应 i、列对应 j），
        元素 int 或 None；同一位置对至多给一个矩阵。
    """

    def __init__(self, single_energies, pair_matrices=(), allowed=None):
        self._full = compile_problem(single_energies, pair_matrices)
        self.n = self._full.n
        self.counts = self._full.counts
        if allowed is None:
            self._compiled = self._full
            self._maps = None
            self._revs = None
        else:
            norm = validate_allowed(self.n, self.counts, allowed)
            self._compiled, self._maps, self._revs = _reduce_compiled(
                self._full, norm
            )

    def _translate_fixed(self, fixed):
        """把原始 ID 的 fixed 约束翻译成缩减下标。

        固定到已被淘汰的 rotamer 时返回 None（该约束下必然无解）。"""
        if self._maps is None or not fixed:
            return fixed
        out = {}
        for p, r in fixed.items():
            if r not in self._revs[p]:
                return None
            out[p] = self._revs[p][r]
        return out

    def _restore(self, assignment):
        """把缩减下标构象还原为原始 rotamer ID。"""
        if assignment is None or self._maps is None:
            return assignment
        return tuple(self._maps[p][r] for p, r in enumerate(assignment))

    def solve(self, fixed=None, budget=DEFAULT_NODE_BUDGET) -> SolveResult:
        """求最小能量构象。``fixed`` 可给 ``{position: rotamer}`` 附加固定约束。"""
        budget = validate_budget(budget)
        fixed = validate_fixed(self.n, self.counts, fixed)
        translated = self._translate_fixed(fixed)
        if translated is None:
            return _infeasible_result(budget)
        raw = branch_and_bound(self._compiled, fixed=translated, budget=budget)
        res = _to_result(raw)
        if self._maps is None or res.assignment is None:
            return res
        return SolveResult(
            status=res.status, assignment=self._restore(res.assignment),
            energy=res.energy, global_lower_bound=res.global_lower_bound,
            nodes_expanded=res.nodes_expanded, budget=res.budget,
        )

    def optimize(self, budget=DEFAULT_NODE_BUDGET) -> AnalysisResult:
        """完整分析，见 :func:`optimize`。使用 allowed 时，fixed_costs 只覆盖
        幸存候选，entry.rotamer 与返回构象均为原始 ID。"""
        ana = _analyze(self._compiled, budget)
        if self._maps is None:
            return ana
        columns = []
        for p in range(self.n):
            col = []
            for entry in ana.fixed_costs[p]:
                if entry is None:
                    col.append(None)
                    continue
                col.append(FixedCostEntry(
                    position=entry.position,
                    rotamer=self._maps[p][entry.rotamer],
                    status=entry.status,
                    minimal_energy=entry.minimal_energy,
                    delta=entry.delta,
                    lower_bound_delta=entry.lower_bound_delta,
                    nodes_expanded=entry.nodes_expanded,
                ))
            columns.append(tuple(col))
        return AnalysisResult(
            status=ana.status,
            assignment=self._restore(ana.assignment),
            energy=ana.energy,
            global_lower_bound=ana.global_lower_bound,
            nodes_expanded=ana.nodes_expanded,
            budget_per_run=ana.budget_per_run,
            second_status=ana.second_status,
            second_assignment=self._restore(ana.second_assignment),
            second_energy=ana.second_energy,
            fixed_costs=tuple(columns),
        )

    def energy_of(self, assignment) -> int:
        """计算给定完整构象的精确总能量（原始 ID；遇 None 硬冲突抛错）。"""
        return exact_energy(self._full, assignment)


def solve(single_energies, pair_matrices=(), *, fixed=None, allowed=None,
          budget=DEFAULT_NODE_BUDGET) -> SolveResult:
    """一次性便捷接口：校验输入并求最小能量构象。

    allowed 可给 {position: [原始 rotamer ID, ...]} 剩余候选映射
    （如 eliminate() 的 remaining），返回构象仍用原始 ID。"""
    return RotamerProblem(single_energies, pair_matrices, allowed=allowed).solve(
        fixed=fixed, budget=budget
    )


def optimize(single_energies, pair_matrices=(), *, allowed=None,
             budget=DEFAULT_NODE_BUDGET) -> AnalysisResult:
    """一次性便捷接口：求最优、次优与全部固定代价增量。

    ``budget`` 是**每个**子搜索（共 2 + 非最优 rotamer 数次，至多 50 次）
    各自的节点展开预算；``nodes_expanded`` 为所有子搜索的实际展开数之和。
    """
    return RotamerProblem(single_energies, pair_matrices, allowed=allowed).optimize(
        budget=budget
    )


def energy_of(single_energies, pair_matrices, assignment) -> int:
    """一次性便捷接口：计算构象精确总能量。"""
    compiled = compile_problem(single_energies, pair_matrices)
    return exact_energy(compiled, assignment)


def _analyze(compiled, budget) -> AnalysisResult:
    from .validation import validate_budget
    budget = validate_budget(budget)

    n = compiled.n
    counts = compiled.counts
    total_nodes = 0

    base = _to_result(branch_and_bound(compiled, budget=budget))
    total_nodes += base.nodes_expanded

    # ---- 第二个不同构象：排除最优（或当前最优）构象后再搜 ----
    best_assign = base.assignment
    if best_assign is not None:
        raw2 = branch_and_bound(compiled, forbid=best_assign, budget=budget)
        second = _to_result(raw2)
    else:
        # 没有任何已知可行构象可排除：
        # - base 为 INFEASIBLE：确无第二个构象；
        # - base 为 BUDGET_EXHAUSTED：分析未完成，不能谎称“证明无次优”，
        #   如实标记 BUDGET_EXHAUSTED（不发起无意义的子搜索）。
        placeholder_status = ("infeasible" if base.status is SolveStatus.INFEASIBLE
                              else "budget_exhausted")
        second = _to_result({
            "status": placeholder_status, "assignment": None, "energy": None,
            "global_lower_bound": None, "nodes_expanded": 0, "budget": budget,
        })
    total_nodes += second.nodes_expanded

    # ---- 逐位置固定到每个非最优 rotamer 的代价增量 ----
    columns = []
    for p in range(n):
        col = [None] * counts[p]
        if best_assign is None:
            columns.append(tuple(col))
            continue
        for r in range(counts[p]):
            if r == best_assign[p]:
                continue
            raw = branch_and_bound(compiled, fixed={p: r}, budget=budget)
            total_nodes += raw["nodes_expanded"]
            st = raw["status"]
            if st == "infeasible":
                entry = FixedCostEntry(
                    position=p, rotamer=r, status=FixedCostStatus.INFEASIBLE,
                    minimal_energy=None, delta=None,
                    lower_bound_delta=None,
                    nodes_expanded=raw["nodes_expanded"],
                )
            else:
                delta = raw["energy"] - base.energy
                if st == "exact":
                    entry = FixedCostEntry(
                        position=p, rotamer=r, status=FixedCostStatus.OPTIMAL,
                        minimal_energy=raw["energy"], delta=delta,
                        lower_bound_delta=None,
                        nodes_expanded=raw["nodes_expanded"],
                    )
                else:
                    # 预算耗尽。base 必为 EXACT（否则固定分析整列跳过），
                    # 故可用已证最优值计算增量下界；子搜索可能连一个可行叶
                    # 都没找到，此时 minimal_energy/delta 为 None，只给下界。
                    lb_delta = (raw["global_lower_bound"] - base.energy
                                if raw["global_lower_bound"] is not None else None)
                    entry = FixedCostEntry(
                        position=p, rotamer=r, status=FixedCostStatus.BEST_KNOWN,
                        minimal_energy=raw["energy"],
                        delta=(None if raw["energy"] is None else delta),
                        lower_bound_delta=lb_delta,
                        nodes_expanded=raw["nodes_expanded"],
                    )
            col[r] = entry
        columns.append(tuple(col))

    # ---- 聚合状态 ----
    exhausted = base.status is SolveStatus.BUDGET_EXHAUSTED or \
        second.status is SolveStatus.BUDGET_EXHAUSTED or \
        any(e is not None and e.status is FixedCostStatus.BEST_KNOWN
            for col in columns for e in col)
    if exhausted:
        agg_status = SolveStatus.BUDGET_EXHAUSTED
    elif base.status is SolveStatus.INFEASIBLE:
        agg_status = SolveStatus.INFEASIBLE
    else:
        agg_status = SolveStatus.EXACT

    return AnalysisResult(
        status=agg_status,
        assignment=base.assignment,
        energy=base.energy,
        global_lower_bound=base.global_lower_bound,
        nodes_expanded=total_nodes,
        budget_per_run=budget,
        second_status=second.status,
        second_assignment=second.assignment,
        second_energy=second.energy,
        fixed_costs=tuple(columns),
    )


# ---------------------------------------------------------------------------
# 剩余候选映射（allowed）：在淘汰后的子集上求解，并把构象还原为原始 ID。
# ---------------------------------------------------------------------------

def validate_allowed(n, counts, allowed):
    """校验剩余候选映射 {position: [原始 rotamer ID, ...]}。

    缺省位置表示保留全部候选；每个给出的列表必须非空、元素为界内真 int、
    无重复。返回 tuple[tuple[int, ...], ...]（每位置升序原始 ID）。
    """
    if not isinstance(allowed, dict):
        raise RotamerInputError(
            f"allowed: 必须是 dict {{position: [rotamer, ...]}}, 得到 {type(allowed).__name__}"
        )
    result = [tuple(range(k)) for k in counts]
    for p, kept in allowed.items():
        if isinstance(p, bool) or not isinstance(p, int):
            raise RotamerInputError(
                f"allowed 的位置键 {p!r}: 必须是 int（bool 不接受）"
            )
        if not (0 <= p < n):
            raise RotamerInputError(f"allowed: 位置 {p} 越界, 合法范围 0..{n - 1}")
        if not isinstance(kept, (list, tuple)):
            raise RotamerInputError(
                f"allowed[{p}]: 必须是 list/tuple, 得到 {type(kept).__name__}"
            )
        if len(kept) == 0:
            raise RotamerInputError(
                f"allowed[{p}]: 至少保留 1 个候选；若某位置无可行候选，"
                f"请用 eliminate() 返回的无解证据而不是继续求解"
            )
        seen = set()
        norm = []
        for r in kept:
            if isinstance(r, bool) or not isinstance(r, int):
                raise RotamerInputError(
                    f"allowed[{p}]: rotamer 必须是 int（bool 不接受）, 得到 {type(r).__name__}"
                )
            if not (0 <= r < counts[p]):
                raise RotamerInputError(
                    f"allowed[{p}]: rotamer {r} 越界, 合法范围 0..{counts[p] - 1}"
                )
            if r in seen:
                raise RotamerInputError(f"allowed[{p}]: rotamer {r} 重复")
            seen.add(r)
            norm.append(r)
        result[p] = tuple(sorted(norm))
    return tuple(result)


def _reduce_compiled(compiled, allowed):
    """按 allowed（每位置原始 ID 元组）构建缩减问题与 ID 映射。

    返回 (reduced_compiled, maps, revs)：
    maps[p][缩减下标] = 原始 ID；revs[p][原始 ID] = 缩减下标。
    """
    from .solver import _Compiled

    n = compiled.n
    maps = allowed
    revs = tuple({orig: idx for idx, orig in enumerate(maps[p])} for p in range(n))
    singles = tuple(
        tuple(compiled.singles[p][orig] for orig in maps[p]) for p in range(n)
    )
    pairs = []
    for (a, b), rows in compiled.pair_map.items():
        sub = tuple(
            tuple(rows[oa][ob] for ob in maps[b]) for oa in maps[a]
        )
        pairs.append((a, b, sub))
    pairs.sort(key=lambda t: (t[0], t[1]))
    reduced = _Compiled(
        n, tuple(len(m) for m in maps), singles, tuple(pairs)
    )
    return reduced, maps, revs


def _infeasible_result(budget):
    return SolveResult(
        status=SolveStatus.INFEASIBLE, assignment=None, energy=None,
        global_lower_bound=None, nodes_expanded=0, budget=budget,
    )
