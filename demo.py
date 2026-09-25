"""固定输入演示：离散 rotamer 能量选择（分支限界）。

运行：python demo.py（仅标准库，秒级完成，无网络/无第三方依赖）。

演示内容全部为现场真实计算，不使用 sleep，也不预录结果：
  1. 局部贪心失败的手工实例：贪心逐位取当前最优，漏掉需要两个位置
     同时翻转才能吃到的成对负能；分支限界找到全局最优。
  2. 一个确定性生成的 6 位置实例的完整分析（最优、次优、逐位置最小代价增量）。
  3. 12 位置实例在小节点预算下耗尽：只报“当前最优 + 全局下界”，不声称精确。
  4. 实际触发的输入拒绝边界（矩阵维度错误、bool 冒充整数）。
"""

import random
import time

from rotamer_select import (
    optimize,
    solve,
    RotamerInputError,
    SolveStatus,
)


def section(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def sequential_greedy(singles, pairs):
    """朴素局部贪心：按位置顺序固定，每步只看单体能和与已固定邻居的对能。

    并列取较小 rotamer 下标。独立于被测库实现，仅用于对照。
    """
    n = len(singles)
    edge = {}
    for i, j, m in pairs:
        edge[(i, j)] = m
        edge[(j, i)] = [[m[r][c] for r in range(len(m))]
                        for c in range(len(m[0]))]
    assignment = []
    total = 0
    for p in range(n):
        best_r, best_v = None, None
        for r, sv in enumerate(singles[p]):
            if sv is None:
                continue
            v = sv
            for q, qr in enumerate(assignment):
                if (p, q) not in edge:
                    continue  # 该位置对没有给能量矩阵
                cell = edge[(p, q)][r][qr]
                if cell is None:
                    v = None
                    break
                v += cell
            if v is not None and (best_v is None or v < best_v):
                best_v, best_r = v, r
        if best_r is None:
            return None, None  # 贪心走进硬冲突，无构象
        assignment.append(best_r)
        total += best_v
    return tuple(assignment), total


def generated_instance(n, seed):
    """仓库内确定性生成（固定种子，不读外部数据）。"""
    rng = random.Random(20260923 + n * 1009 + seed)
    counts = [rng.randint(2, 5) for _ in range(n)]
    singles = [[rng.randint(-6, 6) for _ in range(k)] for k in counts]
    pairs = []
    for i in range(n - 1):
        m = [[None if rng.random() < 0.12
              else rng.randint(-10, 6) for _ in range(counts[i + 1])]
             for _ in range(counts[i])]
        pairs.append((i, i + 1, m))
    return singles, pairs


def main():
    t0 = time.perf_counter()

    # ---- 1. 局部贪心失败 ----
    section("1. 局部贪心失败的手工实例（3 位置，每位置 2 rotamer）")
    singles = [[0, 1], [0, 1], [0, 1]]
    pairs = [
        (0, 1, [[0, 0], [0, -10]]),
        (1, 2, [[0, 0], [0, -10]]),
    ]
    print("单体能（每位置 [r=0, r=1]）:", singles)
    print("对能 (0,1) 与 (1,2) 均为: [[0, 0], [0, -10]]")
    print("  -> r0/r1 搭配没有奖励，只有 (1,1) 得 -10")

    g_assign, g_energy = sequential_greedy(singles, pairs)
    ana = optimize(singles, pairs)
    print(f"\n局部贪心构象 : {g_assign}  总能 = {g_energy}")
    print("  （位置0先取单体最低的 r=0；之后位置1/2 各自翻转都要先付 +1，")
    print("    却吃不到需要两端同时为 1 的 -10，于是一路锁死在 r=0）")
    print(f"分支限界最优 : {ana.assignment}  总能 = {ana.energy}  "
          f"[{ana.status.value}]")
    print(f"第二个不同构象: {ana.second_assignment}  总能 = "
          f"{ana.second_energy}（次优并列时取字典序最小者）")
    print(f"固定位置 0 取 r=0 的最优代价增量: "
          f"{ana.fixed_costs[0][0].delta:+d}（即被迫留在贪心起点的代价）")
    print(f"\n贪心比全局最优差 {g_energy - ana.energy} 个能量单位："
          f"只看局部会错过跨位置协同的负成对能。")

    # ---- 2. 6 位置确定性实例完整分析 ----
    section("2. 确定性生成的 6 位置实例（固定种子，含负对能与硬冲突）")
    singles6, pairs6 = generated_instance(6, seed=1)
    print("每位置 rotamer 数:", [len(c) for c in singles6])
    a6 = optimize(singles6, pairs6)
    print(f"求解状态     : {a6.status.value}（展开节点 {a6.nodes_expanded}）")
    print(f"最优构象     : {a6.assignment}")
    print(f"最优总能量   : {a6.energy}")
    print(f"次优构象     : {a6.second_assignment}  能量 {a6.second_energy}"
          f"（差距 {a6.second_energy - a6.energy:+d}）")
    print("逐位置固定到“最便宜的另一个 rotamer”的代价增量:")
    for p, col in enumerate(a6.fixed_costs):
        alt = min((e for e in col if e is not None),
                  key=lambda e: e.delta, default=None)
        if alt is None:
            print(f"  位置 {p}: 只有一个 rotamer 可行")
        else:
            tag = "硬冲突，无解" if alt.delta is None else f"{alt.delta:+d}"
            print(f"  位置 {p}: 强制 r={alt.rotamer} -> {tag}")

    # ---- 3. 预算耗尽：不给“精确”假象 ----
    section("3. 12 位置实例：小节点预算耗尽 vs 完整求解")
    singles12, pairs12 = generated_instance(12, seed=2)
    small = solve(singles12, pairs12, budget=300)
    full = solve(singles12, pairs12)
    print("rotamer 数:", [len(c) for c in singles12])
    print(f"预算 300   : 状态={small.status.value}, 展开={small.nodes_expanded}")
    print(f"             当前最优已知能量 = {small.energy}")
    print(f"             全局下界         = {small.global_lower_bound}"
          f"（真实最优不低于它）")
    print(f"完整求解   : 状态={full.status.value}, 展开={full.nodes_expanded}, "
          f"最优能量 = {full.energy}")
    assert small.status is SolveStatus.BUDGET_EXHAUSTED
    assert small.global_lower_bound <= full.energy <= small.energy
    print(f"口径校验   : {small.global_lower_bound} <= {full.energy} "
          f"<= {small.energy} 成立；预算耗尽时只给界，不声称精确。")

    # ---- 4. 类别标签与全局数量配额 ----
    section("4. 类别配额：局部最低能构象违反全局数量区间")
    qsingles = [[0, 1], [0, 1], [0, 1]]
    qpairs = [
        (0, 1, [[0, 0], [0, -10]]),
        (1, 2, [[0, 0], [0, -10]]),
    ]
    qcats = [["core", "surface"]] * 3
    qquotas = {"surface": (1, 2)}  # surface 类全局恰选 1..2 次
    plain = solve(qsingles, qpairs)
    qres = optimize(qsingles, qpairs,
                    rotamer_categories=qcats, quotas=qquotas)
    print("3 位置 x 2 rotamer，对能 (1,1) 处各 -10；类别: core/surface")
    print("配额: surface 全局选取次数 ∈ [1, 2]（合成设计指标示例）")
    print(f"无配额最优   : {plain.assignment}  能量 {plain.energy}"
          f"（surface 选了 3 次，越出配额上界）")
    print(f"带配额最优   : {qres.assignment}  能量 {qres.energy}  "
          f"类别计数 {qres.category_counts}  [{qres.status.value}]")
    print(f"带配额次优   : {qres.second_assignment}  能量 "
          f"{qres.second_energy}  类别计数 {qres.second_category_counts}")
    print("硬冲突仍是硬约束：配额只剪必然不可行的分支，不用罚分替代。")

    # ---- 5. 实际触发的拒绝边界 ----
    section("5. 输入拒绝边界（现场触发，非静默纠正）")
    bad_singles = [[0, 1], [2, 3, 4]]
    bad_matrix = [[0, 0, 0]]  # 位置 0 有 2 个 rotamer，矩阵却只有 1 行
    try:
        solve(bad_singles, [(0, 1, bad_matrix)])
    except RotamerInputError as exc:
        print("错误矩阵维度被拒绝:")
        print(f"  RotamerInputError: {exc}")

    try:
        solve([[0, True], [1, 2]], [])  # bool 不得冒充整数能量
    except RotamerInputError as exc:
        print("bool 冒充整数被拒绝:")
        print(f"  RotamerInputError: {exc}")

    try:
        solve([[0, 1], [1, 2]], [],
              rotamer_categories=[["A", "B"], ["A", "B"]],
              quotas={"A": (3, 1)})  # 空闭区间
    except RotamerInputError as exc:
        print("非法配额区间被拒绝:")
        print(f"  RotamerInputError: {exc}")

    elapsed = time.perf_counter() - t0
    section(f"完成：全部为现场计算，实际耗时 {elapsed:.3f} 秒（无 sleep、无预录）")


if __name__ == "__main__":
    main()
