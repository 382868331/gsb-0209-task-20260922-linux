"""独立全枚举参考实现（仅用于测试，不导入被测核心 rotamer_select）。

与被测实现完全独立地直接枚举至多 6 个位置的全部构象，
用最朴素的双重循环求和。位置对契约与被测库相同：``i < j``、
matrix 行对应 i、列对应 j，``None`` 表示硬冲突。

本模块刻意不做完备输入校验（测试只喂合法输入），
也绝不 import rotamer_select，以保证期望值不是被测代码自己给的。
"""

from itertools import product


def _enumerate(single_energies, pair_matrices):
    """生成 (assignment, energy|None)；硬冲突构象能量为 None。"""
    n = len(single_energies)
    for assign in product(*(range(len(c)) for c in single_energies)):
        total = 0
        ok = True
        for p, r in enumerate(assign):
            v = single_energies[p][r]
            if v is None:
                ok = False
                break
            total += v
        if ok:
            for i, j, m in pair_matrices:
                v = m[assign[i]][assign[j]]
                if v is None:
                    ok = False
                    break
                total += v
        yield assign, (total if ok else None)


def brute_best(single_energies, pair_matrices):
    """返回 (字典序最小最优构象, 能量)；无可行构象返回 (None, None)。"""
    best_a, best_e = None, None
    for assign, e in _enumerate(single_energies, pair_matrices):
        if e is None:
            continue
        if best_e is None or e < best_e:
            best_a, best_e = assign, e
    return best_a, best_e


def brute_second(single_energies, pair_matrices):
    """返回与最优构象不同的次优 (assignment, energy)；不存在则 (None, None)。

    并列时取字典序最小者。枚举顺序本身即字典序（itertools.product）。
    """
    best_a, best_e = brute_best(single_energies, pair_matrices)
    if best_a is None:
        return None, None
    sec_a, sec_e = None, None
    for assign, e in _enumerate(single_energies, pair_matrices):
        if assign == best_a or e is None:
            continue
        if sec_e is None or e < sec_e:
            sec_a, sec_e = assign, e
    return sec_a, sec_e


def brute_fixed(single_energies, pair_matrices, position, rotamer):
    """固定 position=rotamer 下的 (最优构象, 能量)；无解返回 (None, None)。"""
    best_a, best_e = None, None
    for assign, e in _enumerate(single_energies, pair_matrices):
        if assign[position] != rotamer or e is None:
            continue
        if best_e is None or e < best_e:
            best_a, best_e = assign, e
    return best_a, best_e


def chain_dp(single_energies, pair_matrices):
    """独立的相邻对 min-sum 动态规划参考（Viterbi 式）。

    仅适用于位置对只在相邻位置 (i, i+1) 上给出的实例（本仓库生成数据即如此）。
    与全枚举参考一样不 import 被测核心；用于独立核对超过 6 位置（最多 12）
    的实例。并列最优时返回字典序最小构象：逐状态保留取得最优值的字典序
    最小前缀（前缀比较优先于较后的位置，与全序列字典序一致）。

    返回 (assignment, energy) 或 (None, None)。
    """
    n = len(single_energies)
    indexed = {(i, j): m for i, j, m in pair_matrices}
    # states[p][r] = (最小能量, 取得它的字典序最小前缀 tuple)
    states = []
    states0 = []
    for r, v in enumerate(single_energies[0]):
        states0.append(None if v is None else (v, (r,)))
    states.append(states0)

    for p in range(1, n):
        m = indexed.get((p - 1, p))
        cur = [None] * len(single_energies[p])
        for r, sv in enumerate(single_energies[p]):
            if sv is None:
                continue
            best = None
            for s, prev in enumerate(states[p - 1]):
                if prev is None:
                    continue
                pv = m[prev[1][-1]][r] if m is not None else 0
                if pv is None:
                    continue
                cand_e = prev[0] + pv + sv
                cand_a = prev[1] + (r,)
                if best is None or cand_e < best[0] or \
                        (cand_e == best[0] and cand_a < best[1]):
                    best = (cand_e, cand_a)
            cur[r] = best
        states.append(cur)

    final = [x for x in states[n - 1] if x is not None]
    if not final:
        return None, None
    e, a = min(final, key=lambda x: (x[0], x[1]))
    return a, e



# ---------------------------------------------------------------------------
# 构象淘汰的独立参考（同样不 import rotamer_select）
# ---------------------------------------------------------------------------

def all_optimal(single_energies, pair_matrices):
    """返回全部取得最优能量的构象（list[tuple]）；无可行构象返回 []。"""
    _, best_e = brute_best(single_energies, pair_matrices)
    if best_e is None:
        return []
    return [a for a, e in _enumerate(single_energies, pair_matrices)
            if e == best_e]


def verify_domination(single_energies, pair_matrices, position, eliminated,
                      dominator, context):
    """独立复核一条支配证书。

    对 context（各位置剩余候选）中所有其余位置选择，朴素计算
    E(eliminated, x) - E(dominator, x)：

    - 两侧同为硬冲突的组合对两者都不可行，不计入比较域；
    - eliminated 侧硬冲突而 dominator 侧可行：差为 +inf，天然成立；
    - dominator 侧硬冲突而 eliminated 侧可行：该组合不在比较域内，跳过
      （证书只需对“支配者可行”的选择成立——支配者不可行时该构象本就
      不含支配者，但此时 eliminated 构象是否最优需由其他证书/轮次保证；
      本参考按“所有 dominator 可行的组合差 > 0”口径复核）。

    返回 (是否成立, 有限差的最小值或 None)。全程整数，无 NaN。
    """
    n = len(single_energies)
    indexed = {(i, j): m for i, j, m in pair_matrices}
    others = [q for q in range(n) if q != position]
    worst = None
    for xs in product(*(context[q] for q in others)):
        env = dict(zip(others, xs))
        diff = 0
        state = "finite"  # finite / pos_inf / skip
        # 单体项
        vr = single_energies[position][eliminated]
        vs = single_energies[position][dominator]
        if vr is None and vs is None:
            continue  # 该位置两侧都不可行：此组合对两者都被排除
        if vr is None:
            continue  # +inf，天然成立
        if vs is None:
            continue  # dominator 不可行：不在比较域
        diff = vr - vs
        for (a, b), m in indexed.items():
            if a != position and b != position:
                continue
            if a == position:
                x = env[b]
                cr, cs = m[eliminated][x], m[dominator][x]
            else:
                x = env[a]
                cr, cs = m[x][eliminated], m[x][dominator]
            if cr is None and cs is None:
                continue
            if cr is None:
                state = "pos_inf"
                break
            if cs is None:
                state = "skip"
                break
            diff += cr - cs
        if state == "finite":
            worst = diff if worst is None else min(worst, diff)
    if worst is None:
        return True, None
    return worst > 0, worst
