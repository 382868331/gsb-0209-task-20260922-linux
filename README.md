# rotamer_select — 肽链侧链构象的离散能量选择

离线结构生物学小工具：给定一条最多 12 个位置的肽链，每位置 2..5 个侧链
rotamer，用**分支限界（branch-and-bound）**求总能量最小的构象，并给出：

- 字典序最小的最优构象与总能量（能量并列时按 rotamer 下标序列字典序）；
- **第二个不同构象**及其能量（次优并列时同样取字典序最小）；
- **逐位置固定代价增量**：把位置 p 强制固定到任意另一个 rotamer 后，
  可达到的最小总能量相对最优的增量（硬冲突 → 证明无解）；
- 节点预算耗尽时返回**当前最优已知解 + 可信全局下界**，严格区分于“证明无解”，
  绝不声称精确。

本次增量新增**构象淘汰**入口：在求解前先证明某些 rotamer 不可能出现在
任何最优构象中（严格支配 + 逐项最不利差界），把它们去掉并给出可独立复核的
淘汰证书；旧求解器可接收剩余候选映射（allowed）并把构象还原为原始 ID。

## 运行环境

- **Python 3.14.7**（标准库 only），Linux 原生。
- 安装 Python 后**无需安装任何第三方依赖**（`requirements.txt` 为空占位）。

venv 准备（可选但推荐；无任何第三方包要装）：

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```
- 核心算法、测试、演示全部离线：不访问网络、账号、数据库、云 API、Docker。
- 示例数据由仓库内固定种子代码确定性生成，不依赖任何外部附件。

## 准确命令

```bash
# 必要测试（53 个用例，约 3 秒）
python -m unittest discover -s tests -v

# 基线演示（现场真实计算，约 0.05 秒；无 sleep、无预录结果）
python demo.py

# 构象淘汰演示（现场真实计算，约 0.02 秒，上限 90 秒；无 sleep、无预录结果）
python demo_iteration.py
```

在仓库根目录（本文件所在目录）执行。测试中“独立参考”
（`tests/reference.py`：≤6 位置全枚举 + 链式动态规划）**不导入被测核心**，
期望值不由被测代码产生；与 `energy_of` 的往返一致只作为补充断言。

## 输入数据结构

```python
single_energies = [
    [e00, e01, ...],          # 位置 0 的各 rotamer 单体能
    ...                        # 每位置 2..5 项
]
pair_matrices = [
    (i, j, [[...], [...]]),   # 必须 i < j；矩阵 k_i 行 × k_j 列，
]                             # 行对应位置 i 的 rotamer，列对应位置 j
```

- 能量元素：**`int` 或 `None`**。`None` 表示硬冲突（该组合禁止出现）。
  可为负（吸引）。只显式给出“相邻指定位置对”的矩阵（也允许任意非相邻对）；
  未给矩阵的位置对能量视为 0。
- 规模上限：位置数 **1 ≤ n ≤ 12**；每位置 rotamer 数 **2..5**。
- 非法输入抛 `RotamerInputError`（`ValueError` 子类），错误信息带定位坐标，
  不做静默纠正/重排：
  - 矩阵行列数与两位置 rotamer 数不符、位置对越界/`i==j`/`i>j`/重复；
  - 位置数或每位置 rotamer 数越界；
  - **能量拒绝 `float`**，因此 NaN / Infinity 直接被拒（信息中注明）；
  - **`bool` 不得冒充整数**（`True/False` 虽是 Python int 子类，一律拒绝）；
  - `budget`、`fixed` 中的下标同样要求真 `int` 且在界内。

## 公开接口

```python
from rotamer_select import solve, optimize, energy_of, RotamerProblem, SolveStatus

# 1) 只求最优
res = solve(single_energies, pair_matrices,
            fixed=None, budget=200_000)
res.status                # SolveStatus.EXACT / INFEASIBLE / BUDGET_EXHAUSTED
res.assignment            # tuple[int,...]，字典序最小最优构象；无解/未得解时 None
res.energy                # 最优（或当前最优）总能量，可能 None
res.global_lower_bound    # 真实最优的可信整数下界；INFEASIBLE 时 None
res.nodes_expanded        # 实际展开节点数
res.exact                 # bool：最优性是否已证明

# 附加固定约束求解（如 fixed={2: 0} 表示位置 2 必须取 rotamer 0）
solve(singles, pairs, fixed={2: 0})

# 2) 完整分析：最优 + 次优 + 全部固定代价增量
ana = optimize(singles, pairs, budget=200_000)
ana.assignment, ana.energy
ana.second_assignment, ana.second_energy     # 不存在第二个可行构象时为 None
ana.second_status
ana.fixed_costs[p][r]                        # 最优 rotamer 处为 None，
                                             # 其余为 FixedCostEntry：
entry.status            # OPTIMAL / BEST_KNOWN(预算耗尽) / INFEASIBLE
entry.minimal_energy    # 固定 p=r 下的最小总能量（INFEASIBLE 为 None）
entry.delta             # minimal_energy - 最优能量（代价增量上界）
entry.lower_bound_delta# 仅 BEST_KNOWN：真实代价增量的整数下界

# 3) 手工构象能量（含硬冲突检测）
energy_of(singles, pairs, assignment)

# 4) 重复求解（编译一次缓存）
prob = RotamerProblem(singles, pair_matrices)
prob.solve(); prob.solve(fixed={0: 1}); prob.optimize(); prob.energy_of(a)

# 5) 构象淘汰：先证明不可能最优，再删去
from rotamer_select import eliminate

res = eliminate(single_energies, pair_matrices)
res.status                 # "ok" / "infeasible"（某残基没有可行候选）
res.remaining              # remaining[p] = 位置 p 幸存候选的原始 ID 元组
res.certificates           # 每条 EliminationCertificate：
cert.position, cert.eliminated, cert.dominator   # 支配者；自身硬冲突时为 None
cert.terms                 # 逐项差界（单体 + 每条位置对的最小差与取界选择）
cert.lower_bound_diff      # 能量差下界，严格 > 0 或 INF（"+inf" 哨兵，非浮点）
cert.context               # 取证时刻各位置剩余候选快照（供独立复核）
res.iterations             # 迭代轮数（直到不动点）
res.infeasible_position, res.infeasible_evidence  # 无解位置与证据证书

# 6) 旧求解器接收剩余候选映射并还原原始 ID
solve(singles, pairs, allowed={p: list(c) for p, c in enumerate(res.remaining)})
optimize(singles, pairs, allowed=...)        # 同上；fixed_costs 只覆盖幸存候选
RotamerProblem(singles, pairs, allowed=...)  # solve/optimize 返回原始 ID 构象
```

**淘汰规则**：对位置 p 的候选 r 与支配者 s，若
single(r)-single(s) + 每条位置对 min_x[pair(r,x)-pair(s,x)] 之和 > 0（严格），
则 r 在任何其余选择下都严格更差，可删去。硬冲突（None）按不可行组合
处理：两侧同 None 记差 0；r 侧 None 记 +inf；s 侧 None 使该界失败
（绝不做无穷大相减产生 NaN）。**相等能量不淘汰**（下界 == 0 保留），
避免丢掉并列最优。迭代到不动点；若某位置候选被清空，返回
status="infeasible" 与导致清空的全部证书作为无解证据。

`budget` 是每次子搜索各自可展开的节点数（正整数）；`optimize` 至多发起
`2 + Σ(k_p−1) ≤ 50` 个子搜索，`nodes_expanded` 为它们的实际展开数之和。

### 状态语义（预算耗尽 ≠ 证明无解）

| 状态 | 含义 | 可用结论 |
|---|---|---|
| `EXACT` | 搜索完整结束（或剩余前沿全部被最优界剪掉） | `energy` 即已证最优，`global_lower_bound == energy` |
| `INFEASIBLE` | 证明不存在任何无硬冲突构象 | `assignment/energy/global_lower_bound` 均为 `None` |
| `BUDGET_EXHAUSTED` | 预算用完、未完成证明 | `global_lower_bound ≤ 真实最优 ≤ energy`（`energy` 可能为 `None`，即连一个可行叶都没找到；此时只给下界） |

## 算法与设计取舍

- **DFS 分支限界**，按位置 0..n−1 顺序固定 rotamer，子节点按 rotamer 下标
  升序访问；因此叶节点按构象序列字典序产出，第一个取得最优能量的叶即
  **字典序最小最优构象**，无需并列时的二次比较。
- **可采纳下界（整数）**：节点下界 = 已固定位置的精确能量（单体 + 两端已定
  位置对）+ 每个未定位置的最小单体能 + 每条未完全固定位置对的最小对能
  （一端已固定时只在该行内取最小）。`None` 不参与取最小；某位置/某行没有
  任何有限取值即剪枝。该下界永远不超过真实最优，所以剪枝不会漏掉最优。
- **预算口径**：预算按“展开的节点数”计。耗尽时取栈中所有仍可能更优前沿
  节点下界的最小值（再与当前最优取 min）作为全局下界；它是可证明的
  `≤ 真实最优`。若剩余前沿已全部被最优界剪光，则即使预算没用完/刚好用完
  也如实报 `EXACT`。
- **次优构象**：对最优构象做“整构象排除”后再搜一次（只在叶深度排除，
  绝不按前缀剪枝，避免误杀仅前缀相同的构象）。
- **固定代价增量**：对每个 (位置, 非最优 rotamer) 做一次带固定约束的搜索，
  增量由该约束最优与全局最优的整数差给出；无解单独标记 `INFEASIBLE`。
- **无浮点**：能量与界全为整数，求解全程不产生浮点误差，也不需要容差。
- **失败不留部分变更**：本库对象是不可变结果（frozen dataclass），求解不
  修改用户传入的数据（输入只读遍历，内部转 tuple）；输入校验在任何搜索前
  完成，非法调用不会产生半成品状态。
- 实现刻意简单可读：显式栈 DFS + 预计算的行/全矩阵最小值，不用第三方求解器。

## 数值与误差口径

- 输入、输出、下界、增量全部是精确整数（或 `None`）；**不存在浮点误差**，
  README 与代码中均无“宽容差”。
- “精确”一词仅在 `EXACT` 状态下使用，含义是分支限界由可采纳下界剪枝
  完整性保证的整数最优；`BUDGET_EXHAUSTED` 下的 `energy` 只是上界。
- 本库只实现本题契约的数学子集（离散、整数、至多 12 位置 × 5 rotamer、
  位置对上显式给矩阵），不宣称兼容任何完整行业能量函数/协议标准。

## 已知限制

- 规模上限即接口边界：n ≤ 12、每位置 2..5 rotamer；超限直接拒绝。
  最坏情况仍是指数搜索（5^12 ≈ 2.4 亿构象），靠负能/硬冲突剪枝与预算兜底；
  预算耗尽只保证界，不保证找到最优。
- 只支持成对位置项（无三体项等更高阶能量）；未给矩阵的位置对能量按 0 处理。
- “相邻”由用户显式给矩阵声明；库不假设距离/拓扑，也不做几何计算。
- 次优与固定增量在预算耗尽时分别给 `second_status` / `BEST_KNOWN` 与下界，
  增量同样可能只是界，不做跨子搜索的联合收紧。
- 不提供命令行产品/服务/文件格式；仅作为可 import 的库与固定 demo.py /
  demo_iteration.py。
- 淘汰是单候选对单候选的严格支配（一阶 DEE 式界），不做多候选联合支配、
  不做组合式不可行推理：某些实际不可行的候选可能幸存（不影响正确性，
  只影响缩减幅度）；缩减效果依实例而定，演示以候选数减少为准，
  不以耗时提升作为功能验收。
- allowed 映射要求每位置至少保留 1 个候选；某位置无可行候选时请使用
  eliminate() 返回的无解证据，而不是继续求解。

## 仓库结构

```
rotamer_select/          # 题库（核心算法）
  __init__.py            # 公开接口
  errors.py              # RotamerInputError
  validation.py          # 严格输入校验与上限
  solver.py              # 分支限界核心（纯整数）
  model.py               # 结果类型、solve/optimize/RotamerProblem、allowed 映射
  eliminate.py           # 构象淘汰：严格支配判定、证书、迭代到不动点
tests/
  reference.py           # 独立参考：全枚举(≤6) + 链式 DP + 证书复核（不 import 核心）
  datagen.py             # 固定种子确定性数据生成
  test_solver.py         # 求解器 36 个用例
  test_eliminate.py      # 淘汰与 allowed 映射 17 个用例
demo.py                  # 固定输入演示（贪心失败 / 预算耗尽 / 拒绝边界）
demo_iteration.py        # 构象淘汰演示（候选缩减、证书复核、无解证据）
```
