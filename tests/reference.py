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

