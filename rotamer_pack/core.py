"""核心算法：rotamer 离散能量选择的分支限界搜索。

能量模型
--------
- ``singles[i][r]``：位置 i 取 rotamer r 的单体能，``None`` 表示该 rotamer
  自身不可用（硬冲突）。
- ``pairs`` 中每条 ``((i, j), matrix)``（i < j）给出成对项
  ``matrix[ri][rj]``，``None`` 表示该组合硬碰撞；未列出的位置对相互作用为 0。
- 全部能量为整数（允许负数），任意可行构象的总能量为精确整数，核心不使用
  任何浮点运算。

搜索
----
深度优先分支限界，按位置顺序、rotamer 索引顺序展开，因此第一个完整可行叶
即 rotamer 索引序列字典序最小者；搜索完整跟踪两个不同构象以得到“第二个不
同构象能量”（同能量并列时保留字典序最小的第二个）。每个节点的可采纳下界
包含：

1. 已分配项的精确贡献（在路径代价 g 中）；
2. 每个未定位置的最小单体能；
3. 一端已定、一端未定的成对项：沿已定端所选行的最小非 None 值；
4. 两端均未定的成对项：整张矩阵的最小非 None 值。

任一最小值不存在（全 None）则该子问题不可行，直接剪枝。另辅以轻量前向
检查：为每个未定位置保留至少一个与已分配邻居相容的 rotamer。

预算（节点展开数）耗尽时抛出内部异常并展开为 ``budget_exhausted`` 报告，
与“证明无解”严格分开。预算耗尽时的全局下界取自**尚未展开的搜索前沿**
中最小的节点下界（剪枝掉的分支由 incumbent 能量背书），因此是对真实最优
能量的合法证书。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

MAX_POSITIONS = 12
MIN_ROTAMERS = 2
MAX_ROTAMERS = 5

STATUS_OPTIMAL = "optimal"
STATUS_INFEASIBLE = "infeasible"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"


class RotamerError(Exception):
    """本库所有公开异常的基类。"""


class ProblemValidationError(RotamerError, ValueError):
    """输入数据或参数非法；消息中带可定位的下标路径。"""


class _BudgetExhausted(Exception):
    """内部信号：节点预算耗尽，不对外暴露。"""


def _is_sequence(value) -> bool:
    return isinstance(value, (list, tuple))


def _check_energy(value, where: str):
    """校验单个能量项：仅接受 int（拒绝 bool）或 None；浮点显式拒绝。"""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProblemValidationError(
            f"{where}: 能量必须是 int 或 None，不能是 bool（布尔值不算整数参数）"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ProblemValidationError(f"{where}: 能量不能是 NaN")
        if math.isinf(value):
            raise ProblemValidationError(f"{where}: 能量不能是 Infinity")
        raise ProblemValidationError(
            f"{where}: 能量必须是整数 int 或 None，不接受浮点值 {value!r}"
        )
    raise ProblemValidationError(
        f"{where}: 能量必须是 int 或 None，得到 {type(value).__name__}"
    )


def _check_index(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProblemValidationError(
            f"{where}: 下标必须是 int（不能是 bool 或 {type(value).__name__}）"
        )
    return value


def _opt_min(values):
    """非 None 整数序列的最小值；全 None/空 返回 None（代表 +∞）。"""
    m = None
    for x in values:
        if x is None:
            continue
        if m is None or x < m:
            m = x
    return m


class _Budget:
    __slots__ = ("limit", "used")

    def __init__(self, limit: Optional[int]):
        self.limit = limit
        self.used = 0

    def tick(self) -> None:
        """登记一次节点展开；达到上限则拒绝展开该节点（其下界仍在前沿）。"""
        if self.limit is not None and self.used >= self.limit:
            raise _BudgetExhausted
        self.used += 1


class _TwoBest:
    """搜索阶段一的两个 incumbent（可变，供异常路径读取）。"""

    __slots__ = ("b1", "b2")

    def __init__(self):
        self.b1: Optional[tuple[int, tuple[int, ...]]] = None
        self.b2: Optional[tuple[int, tuple[int, ...]]] = None

    def add(self, energy: int, seq: tuple[int, ...]) -> None:
        leaf = (energy, seq)
        if self.b1 is None:
            self.b1 = leaf
        elif energy < self.b1[0]:
            self.b2 = self.b1
            self.b1 = leaf
        elif seq != self.b1[1] and (self.b2 is None or energy < self.b2[0]):
            # 相等能量不替换：DFS 顺序保证先到者字典序更小。
            self.b2 = leaf


class _Fringe:
    """开放搜索前沿的下界多重集（懒删除堆），预算中断时给出合法全局 LB。"""

    __slots__ = ("_heap", "_count")

    def __init__(self):
        self._heap: list[int] = []
        self._count: dict[int, int] = {}

    def add(self, bound: int) -> None:
        self._count[bound] = self._count.get(bound, 0) + 1
        heapq.heappush(self._heap, bound)

    def remove(self, bound: int) -> None:
        self._count[bound] -= 1

    def minimum(self) -> Optional[int]:
        heap = self._heap
        while heap and self._count.get(heap[0], 0) == 0:
            heapq.heappop(heap)
        return heap[0] if heap else None


@dataclass(frozen=True)
class SearchReport:
    """一次搜索的完整结果。

    属性
    ----
    status:
        ``"optimal"``（已证明最优，含次优与代价增量）、
        ``"infeasible"``（已证明无可行构象）、
        ``"budget_exhausted"``（预算耗尽，结果不声称精确）。
    energy / assignment:
        已知最优能量与其 rotamer 索引序列；无解或预算内未找到可行叶时为 None。
    second_energy / second_assignment:
        与 assignment 不同的次优构象；能量可与最优相等（并列），并列时取
        字典序最小的第二个序列。仅当 energy_proven 为真时有意义；可行构象
        唯一时二者为 None。
    costs:
        ``costs[i][r]`` 为把位置 i 固定为 rotamer r 时的最优总能量相对
        ``energy`` 的增量（非负整数）；固定后无解为 None。最优构象自身槽位
        为 0。预算在该阶段耗尽则整体为 None（不交付半成品）。
    lower_bound:
        终止时对真实最优能量的全局可采纳下界；已证明最优时等于 energy；
        已证明无解时为 None；预算耗尽且无可行叶时可能为整数。
    energy_proven:
        energy/assignment 是否已被证明为全局最优（与预算是否在代价增量
        阶段耗尽无关）。
    """

    status: str
    energy: Optional[int]
    assignment: Optional[tuple[int, ...]]
    second_energy: Optional[int]
    second_assignment: Optional[tuple[int, ...]]
    costs: Optional[tuple[tuple[Optional[int], ...], ...]]
    lower_bound: Optional[int]
    energy_proven: bool
    costs_complete: bool
    nodes_expanded: int
    budget: Optional[int]

    @property
    def is_optimal(self) -> bool:
        return self.status == STATUS_OPTIMAL

    @property
    def is_infeasible(self) -> bool:
        return self.status == STATUS_INFEASIBLE

    @property
    def budget_exhausted(self) -> bool:
        return self.status == STATUS_BUDGET_EXHAUSTED

    @property
    def feasible(self) -> Optional[bool]:
        """True=找到可行构象；False=已证无解；None=预算内可行性未知。"""
        if self.status == STATUS_INFEASIBLE:
            return False
        if self.assignment is not None:
            return True
        return None

    def __str__(self) -> str:
        lines = [f"SearchReport(status={self.status!r}, nodes={self.nodes_expanded}"
                 f"/budget={self.budget})"]
        if self.assignment is not None:
            tag = "proven optimal" if self.energy_proven else "NOT proven"
            lines.append(f"  best : seq={list(self.assignment)} energy={self.energy}"
                         f"  [{tag}]")
        if self.second_assignment is not None:
            lines.append(f"  2nd  : seq={list(self.second_assignment)} "
                         f"energy={self.second_energy}")
        if self.costs is not None:
            lines.append("  per-position fixed-cost increments (None = infeasible):")
            for i, row in enumerate(self.costs):
                lines.append(f"    pos {i}: {list(row)}")
        if self.status == STATUS_BUDGET_EXHAUSTED:
            lines.append(f"  global lower bound on true optimum: {self.lower_bound}")
        if self.status == STATUS_INFEASIBLE:
            lines.append("  no feasible conformation exists (proven).")
        return "\n".join(lines)


class RotamerProblem:
    """不可变的 rotamer 能量最小化问题实例。

    参数
    ----
    singles:
        长度 1..12 的序列；每个元素是长度 2..5 的序列，元素为 int 或 None。
    pairs:
        可迭代项 ``(key, matrix)``；``key=(i, j)`` 满足 0 <= i < j < n，
        不可重复；matrix 为 n_i 行、每行 n_j 个 int/None。
    """

    def __init__(self, singles, pairs=()):
        self._singles, self._nrot = self._validate_singles(singles)
        n = len(self._singles)
        self._n = n
        self._pairs, self._pair_map, self._before = self._validate_pairs(
            pairs, n, self._nrot
        )

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_singles(singles):
        if not _is_sequence(singles):
            raise ProblemValidationError(
                f"singles: 必须是 list/tuple 序列，得到 {type(singles).__name__}"
            )
        n = len(singles)
        if not (1 <= n <= MAX_POSITIONS):
            raise ProblemValidationError(
                f"singles: 位置数必须在 1..{MAX_POSITIONS} 之间，得到 {n}"
            )
        out = []
        nrot = []
        for i, col in enumerate(singles):
            where = f"singles[{i}]"
            if not _is_sequence(col):
                raise ProblemValidationError(
                    f"{where}: 每个位置的能量必须是 list/tuple，得到 "
                    f"{type(col).__name__}"
                )
            k = len(col)
            if not (MIN_ROTAMERS <= k <= MAX_ROTAMERS):
                raise ProblemValidationError(
                    f"{where}: 每位置 rotamer 数必须在 "
                    f"{MIN_ROTAMERS}..{MAX_ROTAMERS} 之间，得到 {k}"
                )
            row = tuple(
                _check_energy(v, f"singles[{i}][{r}]") for r, v in enumerate(col)
            )
            out.append(row)
            nrot.append(k)
        return tuple(out), tuple(nrot)

    @staticmethod
    def _validate_pairs(pairs, n: int, nrot: tuple[int, ...]):
        if isinstance(pairs, (str, bytes)) or not hasattr(pairs, "__iter__"):
            raise ProblemValidationError(
                f"pairs: 必须是 (key, matrix) 项的序列，得到 {type(pairs).__name__}"
            )
        try:
            enumerated = list(enumerate(pairs))
        except TypeError as exc:
            raise ProblemValidationError(f"pairs: 不可迭代: {exc}") from exc

        ordered = []
        pmap = {}
        before: list[set[int]] = [set() for _ in range(n)]
        for idx, item in enumerated:
            where = f"pairs[{idx}]"
            if not _is_sequence(item) or len(item) != 2:
                raise ProblemValidationError(
                    f"{where}: 每项必须是 (key, matrix) 二元组"
                )
            key, matrix = item
            if not _is_sequence(key) or len(key) != 2:
                raise ProblemValidationError(
                    f"{where}: key 必须是 (i, j) 二元组"
                )
            i = _check_index(key[0], f"{where} key[0]")
            j = _check_index(key[1], f"{where} key[1]")
            if not (0 <= i < n) or not (0 <= j < n):
                raise ProblemValidationError(
                    f"{where}: key=({i}, {j}) 越界，合法位置为 0..{n - 1}"
                )
            if i >= j:
                raise ProblemValidationError(
                    f"{where}: key=({i}, {j}) 非法，必须满足 i < j"
                )
            if (i, j) in pmap:
                raise ProblemValidationError(
                    f"{where}: 位置对 ({i}, {j}) 重复给出"
                )
            if not _is_sequence(matrix):
                raise ProblemValidationError(
                    f"{where} matrix: 必须是 list/tuple，得到 "
                    f"{type(matrix).__name__}"
                )
            if len(matrix) != nrot[i]:
                raise ProblemValidationError(
                    f"{where} matrix: 行数 {len(matrix)} 与位置 {i} 的 rotamer 数 "
                    f"{nrot[i]} 不一致"
                )
            rows = []
            for ri, row in enumerate(matrix):
                if not _is_sequence(row):
                    raise ProblemValidationError(
                        f"{where} matrix[{ri}]: 必须是 list/tuple"
                    )
                if len(row) != nrot[j]:
                    raise ProblemValidationError(
                        f"{where} matrix[{ri}]: 列数 {len(row)} 与位置 {j} 的 "
                        f"rotamer 数 {nrot[j]} 不一致"
                    )
                rows.append(tuple(
                    _check_energy(v, f"{where} matrix[{ri}][{rj}]")
                    for rj, v in enumerate(row)
                ))
            m = tuple(rows)
            ordered.append(((i, j), m))
            pmap[(i, j)] = m
            before[j].add(i)
        return tuple(ordered), pmap, tuple(frozenset(s) for s in before)

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def n_positions(self) -> int:
        return self._n

    @property
    def n_rotamers(self) -> tuple[int, ...]:
        return self._nrot

    @property
    def pairs(self):
        return tuple(((i, j), m) for (i, j), m in self._pairs)

    def evaluate(self, assignment) -> int:
        """计算一个完整构象的精确能量；含 None 冲突或下标非法时抛
        ProblemValidationError。"""
        if not _is_sequence(assignment) or len(assignment) != self._n:
            raise ProblemValidationError(
                "evaluate: assignment 必须是长度等于位置数的 int 序列"
            )
        total = 0
        chosen = []
        for i, r in enumerate(assignment):
            r = _check_index(r, f"assignment[{i}]")
            if not (0 <= r < self._nrot[i]):
                raise ProblemValidationError(
                    f"assignment[{i}]: rotamer 下标 {r} 越界 "
                    f"(位置 {i} 有 {self._nrot[i]} 个 rotamer)"
                )
            s = self._singles[i][r]
            if s is None:
                raise ProblemValidationError(
                    f"assignment[{i}]={r}: 该 rotamer 单体项为硬冲突 (None)"
                )
            total += s
            chosen.append(r)
        for (i, j), m in self._pairs:
            v = m[chosen[i]][chosen[j]]
            if v is None:
                raise ProblemValidationError(
                    f"pair ({i},{j}): rotamer ({chosen[i]},{chosen[j]}) 硬碰撞 (None)"
                )
            total += v
        return total

    # ------------------------------------------------------------------ #
    # 下界与前向检查
    # ------------------------------------------------------------------ #
    def _lower_bound(self, d: int, assigned) -> Optional[int]:
        """状态（0..d-1 已分配）的可采纳下界；不可行时返回 None(+∞)。"""
        total = 0
        for i in range(d, self._n):
            m = _opt_min(self._singles[i])
            if m is None:
                return None
            total += m
        for (i, j), matrix in self._pairs:
            if j < d:
                continue  # 已计入路径代价
            if i < d:
                m = _opt_min(matrix[assigned[i]])
            else:
                m = _opt_min(x for row in matrix for x in row)
            if m is None:
                return None
            total += m
        return total

    def _forward_ok(self, d: int, assigned) -> bool:
        """每个未定位置至少存在一个与**已分配**邻居相容的 rotamer。"""
        for f in range(d, self._n):
            for rf in range(self._nrot[f]):
                if self._singles[f][rf] is None:
                    continue
                ok = True
                for a in self._before[f]:
                    if a >= d:
                        continue  # 该邻居尚未分配，不能用 assigned[-1] 误判
                    if self._pair_map[(a, f)][assigned[a]][rf] is None:
                        ok = False
                        break
                if ok:
                    break
            else:
                return False
        return True

    def _children(self, depth: int, assigned, g: int):
        """产生 (rotamer, 新路径代价)；跳过单体硬冲突与成对硬碰撞。"""
        for r in range(self._nrot[depth]):
            s = self._singles[depth][r]
            if s is None:
                continue
            add = s
            for a in self._before[depth]:
                v = self._pair_map[(a, depth)][assigned[a]][r]
                if v is None:
                    break
                add += v
            else:
                yield r, g + add

    # ------------------------------------------------------------------ #
    # 搜索阶段一：最优 + 次优（两个不同构象）
    # ------------------------------------------------------------------ #
    def _search_two_best(self, fringe: _Fringe, path: list, bg: _Budget,
                         best: _TwoBest) -> None:
        """完整 DFS，通过 best 跟踪两个不同构象；预算耗尽抛 _BudgetExhausted。

        开放前沿由两部分组成：已生成但未展开的子节点（fringe 堆）与活动
        调用路径上各节点自身的下界（path 栈，覆盖尚未惰性生成的兄弟）。
        path 由调用方持有；预算触发时不弹栈，从而保留中断瞬间的路径。
        """
        n = self._n
        assigned = [-1] * n

        def enter(depth: int, g: int, bound: int) -> None:
            bg.tick()                       # 预算触发：bound 仍在堆中
            fringe.remove(bound)
            path.append(bound)              # 预算触发时不弹栈，路径得以保留

            if depth == n:
                best.add(g, tuple(assigned))
                path.pop()
                return

            for r, g2 in self._children(depth, assigned, g):
                assigned[depth] = r
                lb = self._lower_bound(depth + 1, assigned)
                if lb is None:
                    assigned[depth] = -1
                    continue
                child_bound = g2 + lb
                # 两个 incumbent 齐备前不能按能量剪枝（次优可能严格更差）。
                if best.b2 is not None and child_bound >= best.b2[0]:
                    assigned[depth] = -1
                    continue
                if not self._forward_ok(depth + 1, assigned):
                    assigned[depth] = -1
                    continue
                fringe.add(child_bound)
                enter(depth + 1, g2, child_bound)
                assigned[depth] = -1
            path.pop()

        root_bound = self._lower_bound(0, assigned)
        fringe.add(root_bound)
        enter(0, 0, root_bound)

    # ------------------------------------------------------------------ #
    # 搜索阶段二：单点固定的最优代价
    # ------------------------------------------------------------------ #
    def _search_fixed(self, fixed: dict, bg: _Budget) -> Optional[int]:
        """带固定约束的最小能量搜索；无可行完整构象返回 None。"""
        n = self._n
        assigned = [-1] * n
        incumbent = None

        def enter(depth: int, g: int) -> None:
            nonlocal incumbent
            bg.tick()
            if depth == n:
                if incumbent is None or g < incumbent:
                    incumbent = g
                return
            wanted = fixed.get(depth)
            for r, g2 in self._children(depth, assigned, g):
                if wanted is not None and r != wanted:
                    continue
                assigned[depth] = r
                feasible = True
                lb = self._lower_bound(depth + 1, assigned)
                if lb is None:
                    feasible = False
                elif incumbent is not None and g2 + lb >= incumbent:
                    feasible = False
                elif not self._forward_ok(depth + 1, assigned):
                    feasible = False
                if feasible:
                    enter(depth + 1, g2)
                assigned[depth] = -1

        enter(0, 0)
        return incumbent

    @staticmethod
    def _validate_budget(budget) -> Optional[int]:
        if budget is None:
            return None
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ProblemValidationError(
                f"budget: 必须是 int 或 None，得到 {type(budget).__name__}"
            )
        if budget < 0:
            raise ProblemValidationError(f"budget: 不能为负，得到 {budget}")
        return budget

    # ------------------------------------------------------------------ #
    # 公开入口
    # ------------------------------------------------------------------ #
    def solve(self, budget: Optional[int] = None) -> SearchReport:
        """执行分支限界搜索。

        budget 为最多展开的搜索节点数（根节点计 1）；None 表示不限。
        实例不可变，可重复调用，多次调用之间无共享状态。
        """
        budget = self._validate_budget(budget)
        bg = _Budget(budget)

        if self._lower_bound(0, [-1] * self._n) is None:
            return SearchReport(
                status=STATUS_INFEASIBLE,
                energy=None, assignment=None,
                second_energy=None, second_assignment=None,
                costs=None, lower_bound=None,
                energy_proven=False, costs_complete=False,
                nodes_expanded=0, budget=budget,
            )

        fringe = _Fringe()
        path: list[int] = []
        best = _TwoBest()
        try:
            self._search_two_best(fringe, path, bg, best)
            completed = True
        except _BudgetExhausted:
            completed = False

        if not completed:
            # 开放前沿 = 堆中已生成未展开节点 ∪ 活动路径上的节点（覆盖尚未
            # 惰性生成的兄弟；正在展开节点自身仍在堆中，因 tick 先于 remove）。
            # 被剪枝分支其 bound >= b2 >= b1，由 incumbent 背书。
            cands = [b for b in (fringe.minimum(),
                                 min(path) if path else None)
                     if b is not None]
            if best.b1 is not None:
                cands.append(best.b1[0])
            lower_bound = min(cands) if cands else None
            return SearchReport(
                status=STATUS_BUDGET_EXHAUSTED,
                energy=best.b1[0] if best.b1 else None,
                assignment=best.b1[1] if best.b1 else None,
                second_energy=None, second_assignment=None,
                costs=None, lower_bound=lower_bound,
                energy_proven=False, costs_complete=False,
                nodes_expanded=bg.used, budget=budget,
            )

        if best.b1 is None:
            return SearchReport(
                status=STATUS_INFEASIBLE,
                energy=None, assignment=None,
                second_energy=None, second_assignment=None,
                costs=None, lower_bound=None,
                energy_proven=False, costs_complete=False,
                nodes_expanded=bg.used, budget=budget,
            )

        best_e, best_seq = best.b1

        costs = []
        for i in range(self._n):
            row: list[Optional[int]] = []
            exhausted = False
            for r in range(self._nrot[i]):
                if r == best_seq[i]:
                    row.append(0)
                    continue
                try:
                    fixed_best = self._search_fixed({i: r}, bg)
                except _BudgetExhausted:
                    exhausted = True
                    break
                row.append(None if fixed_best is None else fixed_best - best_e)
            if exhausted:
                # 原子语义：代价增量不交付半成品。
                return SearchReport(
                    status=STATUS_BUDGET_EXHAUSTED,
                    energy=best_e, assignment=best_seq,
                    second_energy=best.b2[0] if best.b2 else None,
                    second_assignment=best.b2[1] if best.b2 else None,
                    costs=None, lower_bound=best_e,
                    energy_proven=True, costs_complete=False,
                    nodes_expanded=bg.used, budget=budget,
                )
            costs.append(tuple(row))

        return SearchReport(
            status=STATUS_OPTIMAL,
            energy=best_e, assignment=best_seq,
            second_energy=best.b2[0] if best.b2 else None,
            second_assignment=best.b2[1] if best.b2 else None,
            costs=tuple(costs), lower_bound=best_e,
            energy_proven=True, costs_complete=True,
            nodes_expanded=bg.used, budget=budget,
        )


def minimize_energy(singles, pairs=(), budget: Optional[int] = None) -> SearchReport:
    """便捷函数：构造 :class:`RotamerProblem` 并执行一次 :meth:`solve`。"""
    return RotamerProblem(singles, pairs).solve(budget)
