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


def generate_backbones(n, n_backbones, k_range=(2, 3), *, seed=0,
                       none_prob=0.12, neg_prob=0.4, single_none_prob=0.05,
                       extra_pairs=0):
    """生成多个共用同一组 rotamer 的确定性骨架实例（本迭代）。

    所有骨架共用相同的位点数 n（≤6）与逐位点 rotamer 数（2..3），
    但各自有独立的单体能/对能与独立的硬冲突分布。返回
    ``(backbones, counts)``：``backbones`` 为
    ``[{"singles", "pairs"}, ...]`` 的 list，可直接喂给
    :func:`rotamer_select.solve_backbones` 与参考 ``brute_backbones``。
    """
    rng = random.Random(91_733 * n + 7_919 * n_backbones + seed)
    counts = [rng.randint(*k_range) for _ in range(n)]
    backbones = []
    for b in range(n_backbones):
        singles = []
        for k in counts:
            col = []
            for _ in range(k):
                if rng.random() < single_none_prob:
                    col.append(None)
                else:
                    col.append(rng.randint(-9, 9))
            if all(v is None for v in col):
                col[rng.randrange(k)] = rng.randint(0, 4)
            singles.append(col)

        edges = [(i, i + 1) for i in range(n - 1)]
        candidates = [(i, j) for i in range(n) for j in range(i + 2, n)]
        rng.shuffle(candidates)
        edges.extend(candidates[:extra_pairs])
        pairs = []
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
        backbones.append({"singles": singles, "pairs": pairs})
    return backbones, counts
