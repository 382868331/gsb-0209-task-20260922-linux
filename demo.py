"""固定输入演示：rotamer_pack 的正常结果与一个真实触发的拒绝边界。

运行：python demo.py
全部数据在本文件内硬编码，确定性、离线、无第三方依赖。
"""

import time

from rotamer_pack import RotamerProblem, ProblemValidationError


def print_header(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_greedy_failure() -> None:
    """正常结果：局部贪心失败，分支限界找到真正最优。

    3 个位置，每位置 2 个 rotamer（单体能 singles，相邻成对项 pairs）。
    None 为硬冲突。

    位置 0：rotamer 1 的单体能 (-1) 优于 rotamer 0 (0)，但 rotamer 1
    会与位置 1 的 rotamer 0 硬碰撞，且拿不到 (0,1)/(1,2) 两段强负成对能。
    逐位置贪心（只看单体能 + 与上一位置的相互作用）会先选 rotamer 1。
    """
    print_header("正常结果：局部贪心失败，分支限界给出全局最优")
    singles = [
        [0, -1],     # 位置 0
        [0, 0],      # 位置 1
        [0, 0],      # 位置 2
    ]
    pairs = [
        ((0, 1), [[-5, None],
                  [None, 0]]),
        ((1, 2), [[-6, -6],
                  [0, 0]]),
    ]
    prob = RotamerProblem(singles, pairs)

    # 局部贪心：位置 0 先挑单体能最低的 rotamer 1；之后位置 1 被迫选 1。
    greedy = [1, 1, 0]  # 位置 2 对两条路径都取 0
    greedy_energy = prob.evaluate(greedy)

    t0 = time.perf_counter()
    rep = prob.solve()
    elapsed = time.perf_counter() - t0

    print(f"局部贪心构象        : {greedy}  能量 = {greedy_energy}")
    print(f"分支限界最优构象    : {list(rep.assignment)}  能量 = {rep.energy}"
          f"  (status={rep.status}, {elapsed*1000:.1f} ms)")
    print(f"第二个不同构象      : {list(rep.second_assignment)}  "
          f"能量 = {rep.second_energy}")
    assert rep.energy < greedy_energy, "本算例必须证明贪心次优"
    print(f"=> 贪心比全局最优高 {greedy_energy - rep.energy} 个能量单位；"
          f"短视的单体选择锁死了两段负成对能。")

    print()
    print("每位置固定为指定 rotamer 时的最优代价增量 (None=固定后无解)：")
    for i, row in enumerate(rep.costs):
        print(f"  位置 {i}: {list(row)}")
    print("解读：例如把位置 0 固定为贪心所选的 rotamer 1，"
          f"最小代价增量 = {rep.costs[0][1]}；固定位置 0=1 且位置 1=0 "
          "则直接硬碰撞。")

    print()
    print("预算中断演示（budget=3，不声称精确）：")
    rep_b = prob.solve(budget=3)
    print(f"  status={rep_b.status!r}, 已知最优={rep_b.energy}, "
          f"全局下界={rep_b.lower_bound}, 展开节点={rep_b.nodes_expanded}")
    assert rep_b.status == "budget_exhausted"
    assert rep_b.energy is None
    assert rep_b.lower_bound <= rep.energy
    print(f"  预算内尚未找到可行叶，但全局下界 {rep_b.lower_bound} 不超过")
    print(f"  事后证明的最优 {rep.energy}：中断报告的证书仍合法，且 status 与")
    print("  'infeasible'（证明无解）严格区分。")


def demo_rejection() -> None:
    """边界：成对矩阵维度与 rotamer 数不一致，必须可定位地拒绝。"""
    print_header("拒绝边界：错误的矩阵维度（不静默纠正）")
    singles = [[0, 1], [2, 3]]
    bad_pairs = [
        ((0, 1), [[0, 0, 0],      # 位置 0 只有 2 个 rotamer，这里却给 3 行
                  [0, 0, 0],
                  [0, 0, 0]]),
    ]
    print("位置数=2，每位置 rotamer 数=(2,2)，却提交 3x3 的 (0,1) 成对矩阵。")
    try:
        RotamerProblem(singles, bad_pairs)
    except ProblemValidationError as exc:
        print(f"捕获 {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("非法矩阵必须被拒绝")

    print()
    print("另：bool 被显式拒绝为整数参数（True 不会被静默当成 1）：")
    try:
        RotamerProblem([[0, True], [0, 0]])
    except ProblemValidationError as exc:
        print(f"捕获 {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("bool 能量必须被拒绝")


def main() -> None:
    t0 = time.perf_counter()
    demo_greedy_failure()
    print()
    demo_rejection()
    print("=" * 70)
    print(f"演示完成，总耗时 {time.perf_counter() - t0:.2f} 秒。")


if __name__ == "__main__":
    main()
