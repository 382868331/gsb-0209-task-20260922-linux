"""独立全枚举参考实现（仅测试使用）。

刻意不导入 rotamer_pack：期望值由这份独立编写的直接枚举代码产生，
避免“用被测核心验证被测核心”。能量契约与题目相同：int / None。
"""

from itertools import product


def _matrix_lookup(pairs_dict, i, j, ri, rj):
    m = pairs_dict.get((i, j))
    if m is None:
        return 0
    return m[ri][rj]


def all_feasible(singles, pairs):
    """返回 [(seq_tuple, energy_int), ...]，仅含可行构象，按 (能量, 序列) 排序。"""
    n = len(singles)
    pdict = dict(((i, j), m) for (i, j), m in pairs)
    out = []
    for seq in product(*(range(len(c)) for c in singles)):
        total = 0
        ok = True
        for i, r in enumerate(seq):
            s = singles[i][r]
            if s is None:
                ok = False
                break
            total += s
        if not ok:
            continue
        for (i, j), m in pdict.items():
            v = m[seq[i]][seq[j]]
            if v is None:
                ok = False
                break
            total += v
        if ok:
            out.append((seq, total))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def best_two(singles, pairs):
    """返回 (best, second)；每个是 (seq, energy) 或 None。"""
    feats = all_feasible(singles, pairs)
    if not feats:
        return None, None
    best = feats[0]
    second = feats[1] if len(feats) >= 2 else None
    return best, second


def fixed_min(singles, pairs, pos, rot):
    """位置 pos 固定为 rot 时的最小可行能量；不可行返回 None。"""
    best = None
    for seq, e in all_feasible(singles, pairs):
        if seq[pos] == rot and (best is None or e < best):
            best = e
    return best


def true_optimum(singles, pairs):
    feats = all_feasible(singles, pairs)
    return feats[0][1] if feats else None
