"""测试用确定性数据生成（固定种子，离线，无第三方依赖）。"""

import random


def generate_instance(n, k_range=(2, 5), *, seed=0, none_prob=0.15,
                      neg_prob=0.35, extra_pairs=0, single_none_prob=0.1):
    """生成一个随机但确定性的实例。

    位置对默认包含全部相邻对 (i, i+1)；``extra_pairs`` 再随机加入非相邻对。
    能量范围约 [-12, 12]；对能以 ``neg_prob`` 概率为负；
    ``None`` 以给定概率出现（硬冲突）。
    """
    rng = random.Random(n * 1_000_003 + seed)
    counts = [rng.randint(*k_range) for _ in range(n)]

    singles = []
    for k in counts:
        col = []
        for _ in range(k):
            if rng.random() < single_none_prob:
                col.append(None)
            else:
                col.append(rng.randint(-8, 8))
        if all(v is None for v in col):
            col[rng.randrange(k)] = rng.randint(0, 4)  # 避免整列全 None
        singles.append(col)

    pairs = []
    edges = [(i, i + 1) for i in range(n - 1)]
    candidates = [(i, j) for i in range(n) for j in range(i + 2, n)]
    rng.shuffle(candidates)
    edges.extend(candidates[:extra_pairs])
    for i, j in edges:
        m = []
        for _ in range(counts[i]):
            row = []
            for _ in range(counts[j]):
                if rng.random() < none_prob:
                    row.append(None)
                elif rng.random() < neg_prob:
                    row.append(rng.randint(-12, -1))
                else:
                    row.append(rng.randint(0, 12))
            m.append(row)
        pairs.append((i, j, m))
    return singles, pairs


def generate_categories(counts, *, seed=0, labels=('A', 'B', 'C')):
    '''为每个 rotamer 确定性分配一个类别标签（单一归属）。'''
    rng = random.Random(9_000_017 + seed)
    return [[labels[rng.randrange(len(labels))] for _ in range(k)]
            for k in counts]


def generate_quotas(categories, n, *, seed=0):
    '''根据已生成类别确定性给出每类 (min, max) 闭区间配额。

    min 取 0..实际出现次数 之间，max 取 min..n 之间，保证可能有解
    （是否真有解仍取决于能量/硬冲突，由参考枚举判定）。
    '''
    rng = random.Random(7_000_033 + seed)
    totals = {}
    for col in categories:
        for c in col:
            totals[c] = totals.get(c, 0) + 1
    quotas = {}
    for c, total in totals.items():
        lo = rng.randint(0, min(total, n))
        hi = rng.randint(lo, n)
        quotas[c] = (lo, hi)
    return quotas
