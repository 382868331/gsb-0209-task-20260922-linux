# rotamer_select — 肽链侧链构象的离散能量选择

离线结构生物学小工具，包含两个共用同一数据约定的能力：

**A. 单骨架分支限界**：给定一条最多 12 个位置的肽链，每位置 2..5 个侧链
rotamer，用**分支限界（branch-and-bound）**求总能量最小的构象，并给出：

- 字典序最小的最优构象与总能量（能量并列时按 rotamer 下标序列字典序）；
- **第二个不同构象**及其能量（次优并列时同样取字典序最小）；
- **逐位置固定代价增量**：把位置 p 强制固定到任意另一个 rotamer 后，
  可达到的最小总能量相对最优的增量（硬冲突 → 证明无解）；
- 节点预算耗尽时返回**当前最优已知解 + 可信全局下界**，严格区分于“证明无解”，
  绝不声称精确。

**B. 多骨架共享 rotamer 全枚举（本迭代新增）**：2..3 个给定骨架**共用同一组
rotamer**（最多 6 位点、每点 2..3 选项），各骨架有独立的单点与成对整数能量，
为所有骨架选择**同一个构象**：

- 先最小化**最坏骨架能量** `max_b E_b`（极小极大）；
- 并列时再最小化给定正整数权重下的**加权平均** `Σ w_b E_b / Σ w_b`
  （比较用整数加权和，全程无浮点）；仍并列取构象字典序最小；
- 构象必须在**每个**骨架上都无硬冲突，否则该构象不可行；全部冲突 → 明确
  `INFEASIBLE`，不硬凑带冲突的解；
- 返回**每个骨架的得分**、达到最坏能量的骨架、枚举/可行构象计数。
  规模至多 3^6 = 729，直接全枚举即可给出**已证明**的最优，无预算状态。

## 运行环境

- **Python 3.14.7**（标准库 only），Linux 原生。
- 安装 Python 后**无需安装任何第三方依赖**（`requirements.txt` 为空占位）。
- 核心算法、测试、演示全部离线：不访问网络、账号、数据库、云 API、Docker。
- 示例数据由仓库内固定种子代码确定性生成，不依赖任何外部附件。

## 准确命令

```bash
# 必要测试（63 个用例，约 1 秒）
python -m unittest discover -s tests -v

# 演示（现场真实计算，约 0.05 秒，上限约 90 秒预算内；无 sleep、无预录结果）
python demo.py
```

在仓库根目录（本文件所在目录）执行。测试中“独立参考”
（`tests/reference.py`：单骨架 ≤6 位置全枚举 + 链式动态规划；
多骨架独立全枚举 `brute_backbones`）**不导入被测核心**，
期望值不由被测代码产生；与 `energy_of`/`scores_of` 的往返一致只作为补充断言。

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
```

`budget` 是每次子搜索各自可展开的节点数（正整数）；`optimize` 至多发起
`2 + Σ(k_p−1) ≤ 50` 个子搜索，`nodes_expanded` 为它们的实际展开数之和。

## 多骨架共享 rotamer 接口（本迭代）

### 输入数据结构

```python
backbones = [
    {"singles": single_energies, "pairs": pair_matrices},   # 骨架 0
    {"singles": single_energies, "pairs": pair_matrices},   # 骨架 1
    # 至多第 3 个骨架
]
```

- 每个骨架的 `singles`/`pairs` 与单骨架契约**完全相同**（int 或 `None`，
  `i < j`，矩阵行列对应等）；键名也兼容单骨架的 `single_energies`/
  `pair_matrices`。
- **共用同一组 rotamer**：所有骨架的位点数与逐位点 rotamer 数必须完全一致。
- 多骨架边界更紧：位点数 **1 ≤ n ≤ 6**，每点 **2..3** 个 rotamer，
  骨架数 **2..3**。越界/形状不一致抛 `RotamerInputError`，信息带
  `backbones[b]` 下标定位（内部能量错误同样冠以骨架下标）。
- `weights=None` 表示各骨架等权；否则必须是与骨架等长的**正整数**序列
  （拒绝 bool/float/0/负数）。权重只影响第二级目标。
- `fixed={p: r}` 可选，给所有骨架附加同一个共享固定约束。

### 公开接口

```python
from rotamer_select import solve_backbones, MultiBackboneProblem, SolveStatus

res = solve_backbones(backbones, weights=None, fixed=None)
res.status                # EXACT（全枚举已证）或 INFEASIBLE
res.assignment            # tuple[int,...] 最优共享构象；无解 None
res.scores                # (E_0, E_1, ...) 所选构象在每骨架上的精确能量
res.worst_energy          # max(scores)：第一级目标
res.weighted_sum          # Σ w_b·score_b：第二级目标（整数）
res.weight_total          # Σ w_b；加权平均 = weighted_sum / weight_total
res.weighted_average      # 仅展示用的 float；比较/优化一律用整数加权和
res.worst_backbones       # 达到 worst_energy 的骨架下标元组（可并列多个）
res.conformations_scanned # 枚举的构象数（笛卡尔积大小，受 fixed 限制后）
res.conformations_feasible# 在所有骨架上都无硬冲突的构象数
res.exact                 # bool

# 重复求解（编译一次缓存）；scores_of 对任一骨架硬冲突都整体抛错，不返回半组分数
prob = MultiBackboneProblem(backbones)
prob.solve(); prob.solve(weights=(2, 1)); prob.scores_of(assignment)
```

多骨架模式下结果只可能是 `EXACT` 或 `INFEASIBLE`：规模上限保证全枚举一定跑完，
因此**没有** `BUDGET_EXHAUSTED`，`worst_energy` 即已证的极小极大最优。

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

### 多骨架全枚举（极小极大 + 加权平均）

- 规模上限 6 位点 × 每点 3 选项 = 至多 729 个构象，直接用
  `itertools.product` 按位点顺序全枚举；产出顺序即构象序列字典序，故
  `(worst, weighted_sum)` 两级并列时保留的第一个构象就是字典序最小最优。
- 每个构象在每个骨架上朴素求和（单体 + 给定位置对），任一能量为 `None`
  即该构象在该骨架不可行、整体跳过。
- 比较键为整数元组 `(max_b E_b, Σ_b w_b E_b)`；分母 `Σ w_b` 是与构象无关的
  常数，故最小化加权平均等价于最小化加权和，**全程整数、无浮点、无容差**。
  `weighted_average` 仅为展示便利提供的 float，不参与任何判定。
- 多骨架校验复用单骨架的 `validate_problem`（逐骨架冠以 `backbones[b]`
  定位），再额外检查骨架数、更紧的 n/k 上限与跨骨架形状一致；非法输入在
  枚举前拒绝。`scores_of` 对任一骨架冲突都整体抛错，不返回半组分数。

## 数值与误差口径

- 输入、输出、下界、增量全部是精确整数（或 `None`）；**不存在浮点误差**，
  README 与代码中均无“宽容差”。
- “精确”一词仅在 `EXACT` 状态下使用，含义是分支限界由可采纳下界剪枝
  完整性（单骨架）或全枚举完整性（多骨架）保证的整数最优；
  `BUDGET_EXHAUSTED` 下的 `energy` 只是上界（该状态仅单骨架会出现）。
- 本库只实现本题契约的数学子集（离散、整数；单骨架至多 12 位置 × 5 rotamer、
  多骨架 2..3 骨架 × 至多 6 位置 × 每点 3 rotamer、位置对上显式给矩阵），
  不宣称兼容任何完整行业能量函数/协议标准。

## 已知限制

- 规模上限即接口边界：n ≤ 12、每位置 2..5 rotamer；超限直接拒绝。
  最坏情况仍是指数搜索（5^12 ≈ 2.4 亿构象），靠负能/硬冲突剪枝与预算兜底；
  预算耗尽只保证界，不保证找到最优。
- 只支持成对位置项（无三体项等更高阶能量）；未给矩阵的位置对能量按 0 处理。
- “相邻”由用户显式给矩阵声明；库不假设距离/拓扑，也不做几何计算。
- 次优与固定增量在预算耗尽时分别给 `second_status` / `BEST_KNOWN` 与下界，
  增量同样可能只是界，不做跨子搜索的联合收紧。
- 多骨架选择只实现题面限定子集：**2..3 个骨架、n ≤ 6、每点 2..3 rotamer**；
  超过此范围（更大实例）本迭代不支持，多骨架接口会直接拒绝而非退化为
  不保证完整的搜索。多骨架必须共用逐位点 rotamer 数完全相同的一组构象自由度，
  不支持各骨架位点/选项不同的情形；目标固定为“先最坏、再加权平均”两级，
  不提供其他聚合方式（如乘积、分位数）。
- 不提供命令行产品/服务/文件格式；仅作为可 `import` 的库与固定 `demo.py`。

## 仓库结构

```
rotamer_select/          # 题库（核心算法）
  __init__.py            # 公开接口
  errors.py              # RotamerInputError
  validation.py          # 严格输入校验与上限（含多骨架/权重）
  solver.py              # 单骨架分支限界核心（纯整数）
  model.py               # 单骨架结果类型与 solve/optimize/RotamerProblem
  multi.py               # 多骨架全枚举：极小极大 + 加权平均（本迭代）
tests/
  reference.py           # 独立参考：全枚举(≤6) + 链式 DP + 多骨架全枚举
                         #   （均不 import 核心）
  datagen.py             # 固定种子确定性数据生成（单/多骨架）
  test_solver.py         # 36 个单骨架用例
  test_multi.py          # 27 个多骨架用例（本迭代）
demo.py                  # 固定输入演示（贪心失败 / 预算耗尽 / 多骨架 / 拒绝边界）
```
