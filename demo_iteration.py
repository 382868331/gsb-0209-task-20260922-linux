"""构象淘汰演示：先证明某些侧链构象不可能最优，再把它们去掉。

现场真实计算（无 sleep、无预录结果，全程 < 1 秒，远低于 90 秒上限）：

1. 用仓库内固定种子确定性生成一个 8 位置实例；
2. eliminate() 迭代淘汰并打印每轮候选数变化与证书明细；
3. 在剩余候选子集上用原求解器求解（allowed 映射，构象还原为原始 ID），
   与完整问题最优解对照，确认最优构象与能量完全一致；
4. 独立复核每张证书（朴素枚举，不依赖被测实现）；
5. 触发一次“某残基没有可行候选”的无解证据路径。

运行：python demo_iteration.py
"""

import sys
import time
from itertools import product

from rotamer_select import eliminate, solve, RotamerProblem, INF
from tests.datagen import generate_instance


def independent_cert_check(singles, pairs, cert):
    """独立复核一张证书：对 context 中所有其余选择枚举真实能量差。

    返回 (是否成立, 实际最小差)；硬冲突组合按规则折算，绝不产生 NaN。
    """
    n = len(singles)
    indexed = {(i, j): m for i, j, m in pairs}
    p, r, s = cert.position, cert.eliminated, cert.dominator
    if cert.reason == "self_conflict":
        return singles[p][r] is None, None
    others = [q for q in range(n) if q != p]
    worst = None
    for xs in product(*(cert.context[q] for q in others)):
        assign = dict(zip(others, xs))
        diff = 0
        ok = True
        # 单体差
        vr, vs = singles[p][r], singles[p][s]
        if vr is None and vs is None:
            pass
        elif vr is None:
            diff = INF
        elif vs is None:
            ok = False  # 支配者自身硬冲突，证书无效
        else:
            diff = vr - vs
        if ok and diff != INF:
            for (a, b), m in indexed.items():
                if a != p and b != p:
                    continue
                if a == p:
                    x = assign[b]
                    cr, cs = m[r][x], m[s][x]
                else:
                    x = assign[a]
                    cr, cs = m[x][r], m[x][s]
                if cr is None and cs is None:
                    continue
                if cr is None:
                    diff = INF
                    break
                if cs is None:
                    ok = False  # 该组合下支配者不可行，不在比较域内
                    break
                diff += cr - cs
        if ok:
            if diff == INF:
                continue  # r 不可行而 s 可行：差为正无穷，天然成立
            worst = diff if worst is None else min(worst, diff)
    if worst is None:
        # 所有组合下要么 r 不可行（+inf），要么 s 不可行（不在比较域）
        return True, None
    return worst > 0, worst


def fmt_bound(v):
    return "+inf" if v == INF else str(v)


def main():
    t0 = time.perf_counter()

    print("=" * 68)
    print("1) 生成确定性实例（8 位置，固定种子，现场计算）")
    singles, pairs = generate_instance(8, seed=17, extra_pairs=3)
    counts = [len(c) for c in singles]
    total_before = sum(counts)
    print(f"   每位置 rotamer 数: {counts}  合计 {total_before} 个候选")

    print("=" * 68)
    print("2) 迭代淘汰（严格支配，相等不淘汰，硬冲突不作无穷大相减）")
    res = eliminate(singles, pairs)
    assert res.status == "ok"
    total_after = sum(len(c) for c in res.remaining)
    print(f"   迭代轮数: {res.iterations}   淘汰 {res.eliminated_count} 个候选")
    print(f"   候选总数: {total_before} -> {total_after} "
          f"(减少 {100 * (total_before - total_after) / total_before:.0f}%)")
    print(f"   剩余候选: {[list(c) for c in res.remaining]}")
    print("   证书明细（位置: 淘汰 <- 支配者, 下界, 逐项界）:")
    for c in res.certificates:
        terms = ", ".join(
            f"{t.kind}{t.pair or ''}={fmt_bound(t.min_diff)}" for t in c.terms
        )
        print(f"     [轮{c.iteration}] 位置{c.position}: "
              f"r{c.eliminated} <- " + (f"r{c.dominator}" if c.dominator is not None else "(自身硬冲突)") + "  "
              f"LB={fmt_bound(c.lower_bound_diff)}  ({terms})")

    print("=" * 68)
    print("3) 剩余候选子集上求解（allowed 映射，构象还原为原始 ID）")
    allowed = {p: list(c) for p, c in enumerate(res.remaining)}
    full = solve(singles, pairs)
    reduced = solve(singles, pairs, allowed=allowed)
    print(f"   完整问题最优: 构象 {full.assignment}  能量 {full.energy}  "
          f"({full.status.value}, 展开 {full.nodes_expanded} 节点)")
    print(f"   淘汰后子集最优: 构象 {reduced.assignment}  能量 {reduced.energy}  "
          f"({reduced.status.value}, 展开 {reduced.nodes_expanded} 节点)")
    same = full.assignment == reduced.assignment and full.energy == reduced.energy
    print(f"   最优构象与能量一致: {same}")
    assert same

    print("=" * 68)
    print("4) 独立复核每张淘汰证书（朴素枚举，不用被测实现）")
    checked = 0
    for c in res.certificates:
        ok, worst = independent_cert_check(singles, pairs, c)
        assert ok, f"证书复核失败: {c}"
        checked += 1
    print(f"   {checked} 张证书全部独立复核通过")

    print("=" * 68)
    print("5) 异常路径：某残基没有任何可行候选 -> 无解证据")
    bad_singles = [[None, None], [0, 1]]  # 位置 0 单体全硬冲突
    bad_pairs = []
    bad = eliminate(bad_singles, bad_pairs)
    print(f"   status={bad.status}  无解位置={bad.infeasible_position}  "
          f"证据证书数={len(bad.infeasible_evidence)}")
    assert bad.status == "infeasible"
    prob = RotamerProblem(bad_singles, bad_pairs)
    print(f"   求解器复核: {prob.solve().status.value}（证明无解，与预算耗尽分开）")

    dt = time.perf_counter() - t0
    print("=" * 68)
    print(f"全部演示现场计算完成，用时 {dt:.3f} 秒（上限 90 秒）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
