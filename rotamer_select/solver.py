"""分支限界求解核心（纯整数运算，不使用浮点）。

搜索按位置自然顺序 0..n-1 做深度优先分支限界：

- 每个分支把某位置固定到一个 rotamer；
- 节点的可采纳下界 = 已固定部分的精确能量
  + 每个未定位置的最小单体能
  + 每条“尚未完全固定”位置对的最小对能（若一端已固定则只在该行内取最小）；
- ``None``（硬冲突）不参与取最小；某位置/某行没有任何有限取值则该分支不可行；
- 完全叶节点给出真实总能量；下界 >= 当前最优的节点不再展开；
- 子节点按 rotamer 下标升序访问（栈上逆序压入），DFS 叶序即构象序列的
  字典序，因此第一个取得最优能量的叶节点就是字典序最小的最优构象；
- 预算按“展开的节点数”计。预算耗尽时，全局下界取
  ``min(当前最优, 前沿栈中仍可能更优节点的下界)``，与“证明无解”严格分开。
"""

from .errors import RotamerInputError
from .validation import (
    validate_problem,
    validate_budget,
    validate_fixed,
    validate_categories,
    validate_quotas,
)

DEFAULT_NODE_BUDGET = 200_000

# 状态标记在 model.py 的 SolveStatus 中定义，这里用字符串避免循环导入。
_STATUS_EXACT = "exact"
_STATUS_INFEASIBLE = "infeasible"
_STATUS_BUDGET = "budget_exhausted"


class _Compiled:
    """编译后的问题：预计算每行/全矩阵最小有限值，加速增量下界。

    可选地携带 rotamer 类别标签与各类数量的全局闭区间配额；
    配额只增加可行性剪枝，不改变能量下界的可采纳性。
    """

    __slots__ = ("n", "counts", "singles", "single_min",
                 "incoming", "outgoing", "pair_map",
                 "categories", "quotas", "cat_labels", "cat_ids",
                 "quota_idx", "suffix_avail")

    def __init__(self, n, counts, singles, pairs, categories=None, quotas=None):
        self.n = n
        self.counts = counts
        self.singles = singles

        single_min = []
        for p in range(n):
            finite = [v for v in singles[p] if v is not None]
            single_min.append(min(finite) if finite else None)
        self.single_min = tuple(single_min)

        # incoming[d]: 位置对 (a, d), a < d
        #   每项 (a, rows, row_min)，rows[a 行][d 列]
        # outgoing[d]: 位置对 (d, b), d < b
        #   每项 (full_min, row_min)，固定 d 后下界用 d 行的最小
        incoming = tuple([] for _ in range(n))
        outgoing = tuple([] for _ in range(n))
        pair_map = {}

        for (a, b, rows) in pairs:
            row_min = []
            for row in rows:
                finite = [v for v in row if v is not None]
                row_min.append(min(finite) if finite else None)
            all_finite = [v for row in rows for v in row if v is not None]
            full_min = min(all_finite) if all_finite else None
            incoming[b].append((a, rows, tuple(row_min)))
            outgoing[a].append((full_min, tuple(row_min)))
            pair_map[(a, b)] = rows

        self.incoming = incoming
        self.outgoing = outgoing
        self.pair_map = pair_map

        # ---- 类别与配额（可选）----
        self.categories = categories
        self.quotas = quotas or {}
        if categories is None:
            self.cat_labels = ()
            self.cat_ids = None
            self.quota_idx = {}
            self.suffix_avail = None
        else:
            labels = []
            index = {}
            cat_ids = []
            for p in range(n):
                row = []
                for label in categories[p]:
                    if label not in index:
                        index[label] = len(labels)
                        labels.append(label)
                    row.append(index[label])
                cat_ids.append(tuple(row))
            self.cat_labels = tuple(labels)
            self.cat_ids = tuple(cat_ids)
            self.quota_idx = {index[label]: bounds
                              for label, bounds in self.quotas.items()
                              if label in index}
            # 未出现类别的配额（min>0 必无解）单独保留在 self.quotas 中，
            # 由 branch_and_bound 的根检查处理。
            num_cat = len(labels)
            # suffix_avail[d][c]: 位置 d..n-1 中，类别 c 至少有一个
            # 单体能有限（即可被选择）的 rotamer 的位置数。
            suffix = [[0] * num_cat for _ in range(n + 1)]
            for p in range(n - 1, -1, -1):
                row = suffix[p + 1][:]
                offered = set()
                for r in range(counts[p]):
                    if singles[p][r] is not None:
                        offered.add(cat_ids[p][r])
                for c in offered:
                    row[c] += 1
                suffix[p] = row
            self.suffix_avail = tuple(tuple(r) for r in suffix)


def compile_problem(single_energies, pair_matrices,
                    rotamer_categories=None, quotas=None):
    n, counts, singles, pairs = validate_problem(single_energies, pair_matrices)
    categories = validate_categories(n, counts, rotamer_categories)
    quota_map = validate_quotas(quotas, categories)
    return _Compiled(n, counts, singles, pairs, categories, quota_map)


def _validate_forbid(n, counts, forbid):
    if forbid is None:
        return None
    if not isinstance(forbid, (list, tuple)):
        raise RotamerInputError(
            f"forbid: 必须是 list/tuple, 得到 {type(forbid).__name__}"
        )
    if len(forbid) != n:
        raise RotamerInputError(
            f"forbid: 长度必须等于位置数 {n}, 实际 {len(forbid)}"
        )
    out = []
    for p, r in enumerate(forbid):
        if isinstance(r, bool) or not isinstance(r, int):
            raise RotamerInputError(
                f"forbid[{p}]: rotamer 必须是 int（bool 不接受）, 得到 {type(r).__name__}"
            )
        if not (0 <= r < counts[p]):
            raise RotamerInputError(
                f"forbid[{p}]: rotamer {r} 越界, 合法范围 0..{counts[p] - 1}"
            )
        out.append(r)
    return tuple(out)


def branch_and_bound(compiled, fixed=None, forbid=None, budget=DEFAULT_NODE_BUDGET):
    """执行一次分支限界搜索。

    返回 dict：
      status: "exact" | "infeasible" | "budget_exhausted"
      assignment, energy: 当前最优（可能为 None）
      global_lower_bound: 全局最优能量的可信下界（可能为 None：无解且未探索时）
      nodes_expanded: int
      budget: int
    """
    n = compiled.n
    counts = compiled.counts
    singles = compiled.singles
    single_min = compiled.single_min
    incoming = compiled.incoming
    outgoing = compiled.outgoing
    cat_ids = compiled.cat_ids
    quota_idx = compiled.quota_idx
    suffix_avail = compiled.suffix_avail
    num_cat = len(compiled.cat_labels)

    budget = validate_budget(budget)
    fixed = validate_fixed(n, compiled.counts, fixed)
    forbid = _validate_forbid(n, compiled.counts, forbid)

    # 配额静态可行性：未在标签中出现的类别若要求 min>0，直接证明无解；
    # 已出现类别若可提供的位置总数不足 min，同样直接证明无解。
    if compiled.categories is not None and compiled.quotas:
        for label, (lo, _hi) in compiled.quotas.items():
            if lo <= 0:
                continue
            cidx = (compiled.cat_labels.index(label)
                    if label in compiled.cat_labels else None)
            if cidx is None or suffix_avail[0][cidx] < lo:
                return _result(_STATUS_INFEASIBLE, None, None, None, 0, budget)

    # 根下界：全部单体最小 + 每条位置对的全矩阵最小。
    root_lb = 0
    for m in single_min:
        if m is None:
            return _result(_STATUS_INFEASIBLE, None, None, None, 0, budget)
        root_lb += m
    for d in range(n):
        for full_min, _ in outgoing[d]:
            if full_min is None:
                return _result(_STATUS_INFEASIBLE, None, None, None, 0, budget)
            root_lb += full_min

    # 栈元素: (depth, assignment_tuple, exact_energy, lower_bound, cat_counts)
    # cat_counts[c] = 已固定位置中类别 c 的选取次数（无类别时恒为空元组）。
    stack = [(0, (), 0, root_lb, (0,) * num_cat)]
    best_energy = None
    best_assign = None
    expanded = 0
    status = _STATUS_EXACT

    while stack:
        if expanded >= budget:
            status = _STATUS_BUDGET
            break

        d, assign, exact, lb, cat_counts = stack.pop()
        if best_energy is not None and lb >= best_energy:
            # 已不可能更优；等界分支的叶构象字典序只会更大，无需展开。
            continue

        expanded += 1

        if d == n:
            # forbid 只针对完整构象，绝不在中间深度剪前缀，
            # 否则会误杀仅前缀相同的其他构象。
            if forbid is not None and assign == forbid:
                continue
            best_energy = exact
            best_assign = assign
            continue

        # 逆序生成，使 r 最小的子节点位于栈顶，保证叶节点按字典序访问。
        children = []
        for r in range(counts[d] - 1, -1, -1):
            if d in fixed and r != fixed[d]:
                continue

            s = singles[d][r]
            if s is None:
                continue  # 单体硬冲突

            # exact_add: 本步固定后变为确定值的能量（单体 + 入边对能）。
            # lb_add/sub: 下界中被替换掉的估计项与其收紧后的新值。
            exact_add = s
            lb_add = s
            sub = single_min[d]

            feasible = True
            for a, rows, rmins in incoming[d]:
                v = rows[assign[a]][r]
                if v is None:
                    feasible = False
                    break
                exact_add += v
                lb_add += v
                sub += rmins[assign[a]]  # 一端固定时的行最小
            if not feasible:
                continue

            # 出边 (d, b) 两端尚未都固定，对能仍只是“估计项”：
            # 下界从全矩阵最小收紧为 d 所在行的最小；绝不进入 exact_add，
            # 否则该位置对会在 b 固定时被计算两次。
            for full_min, rmins in outgoing[d]:
                m = rmins[r]
                if m is None:
                    feasible = False
                    break
                sub += full_min
                lb_add += m
            if not feasible:
                continue

            new_assign = assign + (r,)

            # ---- 类别配额剪枝（可行性剪枝，与能量下界正交）----
            # 1) 已超上限：该类别计数超过 max，直接剪掉；
            # 2) 下界无望：剩余位置中仍可能取该类别 c 的位置数
            #    （suffix_avail，忽略对能冲突的乐观估计）加上当前计数
            #    仍达不到 min，则该子树不可能满足配额，剪掉。
            # 两者都只剪掉“必然不可行”的分支，不影响最优性。
            new_counts = cat_counts
            if quota_idx:
                cid = cat_ids[d][r]
                tmp = list(cat_counts)
                tmp[cid] += 1
                q = quota_idx.get(cid)
                if q is not None and tmp[cid] > q[1]:
                    continue
                suf = suffix_avail[d + 1]
                hopeless = False
                for c2, (lo, _hi) in quota_idx.items():
                    if lo and tmp[c2] + suf[c2] < lo:
                        hopeless = True
                        break
                if hopeless:
                    continue
                new_counts = tuple(tmp)

            child_lb = lb - sub + lb_add
            if best_energy is not None and child_lb >= best_energy:
                continue
            children.append((d + 1, new_assign, exact + exact_add, child_lb,
                             new_counts))

        stack.extend(children)

    if status == _STATUS_BUDGET:
        alive_lbs = [node_lb for (_, _, _, node_lb, _) in stack
                     if best_energy is None or node_lb < best_energy]
        if not alive_lbs:
            # 剩余前沿已全部被最优界剪光——实际上已证最优。
            status = _STATUS_EXACT
            global_lb = best_energy
        else:
            cand = alive_lbs + ([best_energy] if best_energy is not None else [])
            global_lb = min(cand)
    else:
        # 栈自然清空：完整搜索完毕。
        if best_energy is None:
            status = _STATUS_INFEASIBLE
            global_lb = None
        else:
            global_lb = best_energy

    return _result(status, best_assign, best_energy, global_lb, expanded,
                   budget, _counts_for(compiled, best_assign))


def _counts_for(compiled, assignment):
    '''统计一个完整构象的类别计数；无类别或无构象时返回 None。'''
    if compiled.categories is None or assignment is None:
        return None
    counts = {}
    for p, r in enumerate(assignment):
        label = compiled.categories[p][r]
        counts[label] = counts.get(label, 0) + 1
    return counts


def _result(status, assignment, energy, global_lower_bound, nodes_expanded,
            budget, category_counts=None):
    return {
        "status": status,
        "assignment": assignment,
        "energy": energy,
        "global_lower_bound": global_lower_bound,
        "nodes_expanded": nodes_expanded,
        "budget": budget,
        "category_counts": category_counts,
    }


def exact_energy(compiled, assignment):
    """计算完整构象的精确总能量；遇硬冲突/非法下标抛 RotamerInputError。"""
    n = compiled.n
    counts = compiled.counts
    if not isinstance(assignment, (list, tuple)):
        raise RotamerInputError(
            f"assignment: 必须是 list/tuple, 得到 {type(assignment).__name__}"
        )
    if len(assignment) != n:
        raise RotamerInputError(
            f"assignment: 长度必须等于位置数 {n}, 实际 {len(assignment)}"
        )
    total = 0
    for p, r in enumerate(assignment):
        if isinstance(r, bool) or not isinstance(r, int):
            raise RotamerInputError(
                f"assignment[{p}]: rotamer 必须是 int（bool 不接受）, 得到 {type(r).__name__}"
            )
        if not (0 <= r < counts[p]):
            raise RotamerInputError(
                f"assignment[{p}]: rotamer {r} 越界, 合法范围 0..{counts[p] - 1}"
            )
        v = compiled.singles[p][r]
        if v is None:
            raise RotamerInputError(
                f"assignment[{p}]={r}: 该 rotamer 单体能为硬冲突 None"
            )
        total += v
    for (a, b), rows in compiled.pair_map.items():
        v = rows[assignment[a]][assignment[b]]
        if v is None:
            raise RotamerInputError(
                f"assignment: 位置对 ({a}, {b}) 取 ({assignment[a]}, {assignment[b]}) "
                f"为硬冲突 None"
            )
        total += v
    return total
