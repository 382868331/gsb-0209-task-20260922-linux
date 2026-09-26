"""构象（rotamer）淘汰：先证明某些候选不可能最优，再把它们去掉。

判定规则（严格支配，DEE 式）
----------------------------
对位置 p 的两个候选 r（被检）与 s（支配者），定义任意其余位置选择 x 下的
能量差::

    D(x) = E(p=r, x) - E(p=s, x)
         = [single(p,r) - single(p,s)]
           + sum_{pair (p,q)} [pair(r, x_q) - pair(s, x_q)]

若能证明 D(x) > 0 对所有其余选择 x 成立，则 r 在任何可行构象中都严格
不如 s，r 不可能出现在任何最优构象中，可安全淘汰。证明方式是逐项取
最不利（最小）差界::

    LB = [single(p,r) - single(p,s)]
         + sum_{pair} min_{x_q 剩余} [pair(r, x_q) - pair(s, x_q)]

LB > 0（严格）即淘汰；LB == 0（相等能量）不淘汰，避免丢掉并列最优。

硬冲突（None）按不可行组合处理，全程整数，绝不把无穷大相减成 NaN：

- 两端都 None：该组合对 r 与 s 同样不可行，差贡献记 0（该 x 对两者都被排除）；
- r 侧 None、s 侧有限：该组合下 r 不可行而 s 可行，差为 +inf（记 INF）；
- r 侧有限、s 侧 None：差为 -inf，该项界直接失败，s 不能据此支配 r；
- 单体能 single(p,r) 为 None：r 自身在任何构象中都不可行，
  以 self_conflict 证书直接淘汰（无需支配者）。

迭代与无解证据
--------------
每轮淘汰后剩余候选变少，各项 min 界只会更紧，因此重复扫描直到不动点。
同一轮内多张证书各自相对本轮开始时的剩余集独立成立（淘汰只会让界更紧，
不会失效）。若某位置候选被清空，说明原问题不存在无硬冲突构象，
返回 status="infeasible" 与导致清空的全部证书作为无解证据。

证书自包含：每张证书记录取证时刻各位置的剩余候选（context）、
逐项差界（terms）与总下界（lower_bound_diff），第三方可不看
本库代码独立复核。
"""

from dataclasses import dataclass

from .validation import validate_problem

__all__ = [
    "INF",
    "BoundTerm",
    "EliminationCertificate",
    "EliminationResult",
    "eliminate",
]

INF = "inf"
"""证书中表示 +inf 的哨兵字符串（硬冲突导致的无界正差），不是浮点。"""

_TERM_SINGLE = "single"
_TERM_PAIR = "pair"

_REASON_SELF = "self_conflict"
_REASON_DOMINATED = "dominated"


@dataclass(frozen=True)
class BoundTerm:
    """证书中一项差界。

    min_diff 为 int 或 INF；argmin 是取得该最小差时对方
    位置的 rotamer（原始 ID），单体项为 None。
    """

    kind: str
    pair: tuple | None
    min_diff: object
    argmin: int | None


@dataclass(frozen=True)
class EliminationCertificate:
    """一条淘汰证书：证明 eliminated 不可能出现在任何最优构象中。"""

    iteration: int
    position: int
    eliminated: int
    dominator: int | None
    """支配它的候选（原始 ID）；self_conflict 淘汰时为 None。"""
    reason: str
    terms: tuple
    lower_bound_diff: object
    """D(x) 的可信下界：int 且严格 > 0，或为 INF。"""
    context: tuple
    """取证时刻各位置剩余候选（原始 ID 元组）的快照，供独立复核。"""


@dataclass(frozen=True)
class EliminationResult:
    """迭代淘汰的最终结果。"""

    status: str
    """"ok" 或 "infeasible"（某位置候选被清空，证明原问题无解）。"""
    remaining: tuple | None
    """remaining[p] 为位置 p 幸存候选的原始 ID 元组；infeasible 时为 None。"""
    certificates: tuple
    iterations: int
    infeasible_position: int | None
    infeasible_evidence: tuple
    """导致 infeasible_position 清空的全部证书（无解证据）。"""

    @property
    def eliminated_count(self) -> int:
        return len(self.certificates)


def _cell_diff(v_r, v_s):
    """两个能量格（int 或 None）的差：返回 int、INF 或 None（表示 -inf）。"""
    if v_r is None and v_s is None:
        return 0
    if v_r is None:
        return INF
    if v_s is None:
        return None  # -inf：界失败
    return v_r - v_s


def _pair_term(rows, p_is_first, r, s, other_remaining):
    """一条位置对项的 min 差界。

    返回 (min_diff, argmin)；min_diff 为 int / INF / None(-inf)。
    """
    best = None
    best_x = None
    for x in other_remaining:
        if p_is_first:
            d = _cell_diff(rows[r][x], rows[s][x])
        else:
            d = _cell_diff(rows[x][r], rows[x][s])
        if d is None:
            return None, x  # -inf，直接失败
        if d == INF:
            if best is None:
                best, best_x = INF, x
        elif best is None or best == INF or d < best:
            best, best_x = d, x
    return best, best_x


def _try_dominate(p, r, s, remaining, singles, pair_map, pair_end):
    """尝试用 s 支配 r；成功返回 (terms, lower_bound)，失败返回 None。"""
    d0 = _cell_diff(singles[p][r], singles[p][s])
    if d0 is None:
        return None  # single(s) 硬冲突，s 不能作支配者
    terms = [BoundTerm(kind=_TERM_SINGLE, pair=None, min_diff=d0, argmin=None)]
    total = INF if d0 == INF else d0
    for (a, b) in pair_end[p]:
        rows = pair_map[(a, b)]
        p_is_first = (a == p)
        other = b if p_is_first else a
        md, argmin = _pair_term(rows, p_is_first, r, s, remaining[other])
        if md is None:
            return None  # -inf：界失败
        terms.append(BoundTerm(kind=_TERM_PAIR, pair=(a, b),
                               min_diff=md, argmin=argmin))
        if total != INF:
            total = INF if md == INF else total + md
    if total == INF or total > 0:
        return tuple(terms), total
    return None


def eliminate(single_energies, pair_matrices=()):
    """迭代淘汰不可能最优的候选，直到不动点。

    返回 EliminationResult。输入契约与求解器相同
    （非法输入抛 RotamerInputError）。
    """
    n, counts, singles, pairs = validate_problem(single_energies, pair_matrices)

    pair_map = {}
    pair_end = [[] for _ in range(n)]
    for a, b, rows in pairs:
        pair_map[(a, b)] = rows
        pair_end[a].append((a, b))
        pair_end[b].append((a, b))

    remaining = [tuple(range(k)) for k in counts]
    certificates = []
    iteration = 0

    while True:
        iteration += 1
        context = tuple(remaining)
        round_certs = []
        for p in range(n):
            for r in remaining[p]:
                if singles[p][r] is None:
                    round_certs.append(EliminationCertificate(
                        iteration=iteration, position=p, eliminated=r,
                        dominator=None, reason=_REASON_SELF,
                        terms=(BoundTerm(kind=_TERM_SINGLE, pair=None,
                                         min_diff=INF, argmin=None),),
                        lower_bound_diff=INF, context=context,
                    ))
                    continue
                for s in remaining[p]:
                    if s == r or singles[p][s] is None:
                        continue
                    hit = _try_dominate(p, r, s, remaining, singles,
                                        pair_map, pair_end)
                    if hit is not None:
                        terms, lb = hit
                        round_certs.append(EliminationCertificate(
                            iteration=iteration, position=p, eliminated=r,
                            dominator=s, reason=_REASON_DOMINATED,
                            terms=terms, lower_bound_diff=lb, context=context,
                        ))
                        break
        if not round_certs:
            break
        certificates.extend(round_certs)
        eliminated_by_pos = {}
        for cert in round_certs:
            eliminated_by_pos.setdefault(cert.position, set()).add(cert.eliminated)
        for p, dead in eliminated_by_pos.items():
            remaining[p] = tuple(r for r in remaining[p] if r not in dead)
            if not remaining[p]:
                evidence = tuple(c for c in certificates if c.position == p)
                return EliminationResult(
                    status="infeasible", remaining=None,
                    certificates=tuple(certificates),
                    iterations=iteration,
                    infeasible_position=p,
                    infeasible_evidence=evidence,
                )

    return EliminationResult(
        status="ok", remaining=tuple(remaining),
        certificates=tuple(certificates),
        iterations=iteration - 1,
        infeasible_position=None,
        infeasible_evidence=(),
    )
