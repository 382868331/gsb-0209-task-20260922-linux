# rotamer_select — 肽链侧链构象的离散能量选择

离线结构生物学小工具：给定一条最多 12 个位置的肽链，每位置 2..5 个侧链
rotamer，用**分支限界（branch-and-bound）**求总能量最小的构象，并给出：

- 字典序最小的最优构象与总能量（能量并列时按 rotamer 下标序列字典序）；
- **第二个不同构象**及其能量（次优并列时同样取字典序最小）；
- **逐位置固定代价增量**：把位置 p 强制固定到任意另一个 rotamer 后，
  可达到的最小总能量相对最优的增量（硬冲突 → 证明无解）；
- 节点预算耗尽时返回**当前最优已知解 + 可信全局下界**，严格区分于“证明无解”，
  绝不声称精确。

可选扩展：**rotamer 类别标签与全局数量配额**。每个 rotamer 可被打上恰好一个
类别标签（单一归属），调用方可为任意类别指定全局选取次数的闭区间
`{类别: (min, max)}`（非负整数）；求解在“每个位置仍恰选一个 rotamer”的
原约束下，额外保证各类别计数落在区间内。该配额是**合成设计指标**
（例如人为要求的组成配比），**不得解释成真实蛋白稳定性结论**；
硬冲突（`None`）仍然是硬约束，任何情况下都不用罚分替代。

## 运行环境

- **Python 3.14.7**（标准库 only），Linux 原生。
- 安装 Python 后**无需安装任何第三方依赖**（`requirements.txt` 为空占位）。
- 核心算法、测试、演示全部离线：不访问网络、账号、数据库、云 API、Docker。
- 示例数据由仓库内固定种子代码确定性生成，不依赖任何外部附件。

## 准确命令

```bash
# 必要测试（51 个用例，约 1 秒）
python -m unittest discover -s tests -v

# 演示（现场真实计算，约 0.05 秒，上限约 8 秒预算内；无 sleep、无预录结果）
python demo.py
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

### 类别标签与数量配额（可选）

```python
rotamer_categories = [
    ["core", "surface"],   # 与 single_energies 同形：每位置每 rotamer 一个标签
    ...
]
quotas = {"surface": (1, 4), "core": (2, 12)}   # {类别: (min, max)} 闭区间
```

- 标签：`str` 或 `int`（拒绝 `bool`/`None`/其他类型）；每个 rotamer **恰好一个**类别。
- 配额边界：**非负整数** 且 `min <= max`；`max = 0` 即整类禁用。
  给 `quotas` 必须同时给 `rotamer_categories`；配额键允许是未出现的类别
  （其 `min > 0` 时由搜索证明 `INFEASIBLE`，而不是校验期静默改写）。
- 只给类别、不给配额：解与旧行为完全一致，仅额外输出类别计数。
- 非法输入（区间为空、负数、非整数、标签形状不符等）抛 `RotamerInputError`。

## 公开接口

```python
from rotamer_select import solve, optimize, energy_of, RotamerProblem, SolveStatus

# 1) 只求最优
res = solve(single_energies, pair_matrices,
            fixed=None, budget=200_000)
res.status                # SolveStatus.EXACT / INFEASIBLE / BUDGET_EXHAUSTED
res.assignment            # tuple[int,...]，字典序最小最优构象；无解/未得解时 None
res.energy                # 最优（或当前最优）总能量，可能 None
res.category_counts       # dict{类别: 次数}；未给类别或无构象时 None
res.global_lower_bound    # 真实最优的可信整数下界；INFEASIBLE 时 None
res.nodes_expanded        # 实际展开节点数
res.exact                 # bool：最优性是否已证明

# 附加固定约束求解（如 fixed={2: 0} 表示位置 2 必须取 rotamer 0）
solve(singles, pairs, fixed={2: 0})

# 带类别配额的求解（旧的无配额调用方式完全保留）
solve(singles, pairs, rotamer_categories=cats, quotas={"surface": (1, 4)})

# 2) 完整分析：最优 + 次优 + 全部固定代价增量
ana = optimize(singles, pairs, budget=200_000)
ana.assignment, ana.energy
ana.second_assignment, ana.second_energy     # 不存在第二个可行构象时为 None
ana.second_status
ana.category_counts, ana.second_category_counts   # 最优/次优构象的类别计数
ana.fixed_costs[p][r]                        # 最优 rotamer 处为 None，
                                             # 其余为 FixedCostEntry：
entry.status            # OPTIMAL / BEST_KNOWN(预算耗尽) / INFEASIBLE
entry.minimal_energy    # 固定 p=r 下的最小总能量（INFEASIBLE 为 None）
entry.delta             # minimal_energy - 最优能量（代价增量上界）
entry.lower_bound_delta# 仅 BEST_KNOWN：真实代价增量的整数下界

# 3) 手工构象能量（含硬冲突检测）
energy_of(singles, pairs, assignment)

# 4) 重复求解（编译一次缓存）
prob = RotamerProblem(singles, pair_matrices,
                      rotamer_categories=None, quotas=None)
prob.solve(); prob.solve(fixed={0: 1}); prob.optimize(); prob.energy_of(a)
```

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
- **配额剪枝（可行性剪枝，与能量下界正交）**：搜索逐节点维护各类别已选
  次数，两类必然不可行的分支直接剪掉——(a) 某类别计数**已超上限**；
  (b) 剩余位置中仍可能取该类别的位置数（忽略对能冲突的乐观后缀估计）
  加上当前计数**仍达不到该类下界**。两者都只剪必然不可行的分支，
  不丢弃任何负成对能量，能量下界保持可采纳，因此不会制造错误剪枝。
  配额导致的不可满足同样报 `INFEASIBLE`，与预算耗尽严格区分。
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
- 不提供命令行产品/服务/文件格式；仅作为可 `import` 的库与固定 `demo.py`。
- 类别配额是**合成设计指标**接口：库只保证计数落在给定闭区间，
  不把配额结果解释成任何真实蛋白稳定性/物理结论。
- 配额的下界可行性检查是逐类别独立的必要条件（不做跨类别的联合
  匹配可行性判定），因此剪枝保守、正确，但理论上可能少剪一些分支。

## 仓库结构

```
rotamer_select/          # 题库（核心算法）
  __init__.py            # 公开接口
  errors.py              # RotamerInputError
  validation.py          # 严格输入校验与上限
  solver.py              # 分支限界核心（纯整数）
  model.py               # 结果类型与 solve/optimize/RotamerProblem
tests/
  reference.py           # 独立参考：全枚举(≤6) + 链式 DP（不 import 核心）
  datagen.py             # 固定种子确定性数据生成
  test_solver.py         # 36 个用例（无配额旧行为）
  test_quotas.py         # 15 个用例（类别配额新行为）
demo.py                  # 固定输入演示（贪心失败 / 预算耗尽 / 类别配额 / 拒绝边界）
```
