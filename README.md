# rotamer_select — 肽链侧链构象的离散能量选择

离线结构生物学小工具，提供两个层次的能力：

1. **单肽链**：给定一条最多 12 个位置的肽链，每位置 2..5 个侧链
   rotamer，用**分支限界（branch-and-bound）**求总能量最小的构象，并给出：

   - 字典序最小的最优构象与总能量（能量并列时按 rotamer 下标序列字典序）；
   - **第二个不同构象**及其能量（次优并列时同样取字典序最小）；
   - **逐位置固定代价增量**：把位置 p 强制固定到任意另一个 rotamer 后，
     可达到的最小总能量相对最优的增量（硬冲突 → 证明无解）；
   - 节点预算耗尽时返回**当前最优已知解 + 可信全局下界**，严格区分于“证明无解”，
     绝不声称精确。

2. **多个给定骨架共用一组 rotamer**：同一位点结构（最多 6 位点、每点
   1..3 个 rotamer）要同时放到 1..3 个给定骨架上，每个骨架各有自己的
   单体能与成对能。选出**一个共用构象**，按两级目标**全枚举**：
   - 先**最小化最坏骨架能量** `max_b E_b(a)`；
   - 最坏能量并列时，再**最小化给定正整数权重的加权平均**
     （按整数加权和比较，不引入浮点）；
   - 再并列取 rotamer 下标序列字典序最小者；
   - 返回构象、**每个骨架的得分**、枚举/可行构象数；无共用可行构象时
     明确报 `INFEASIBLE`。

## 运行环境

- **Python 3.14.7**（标准库 only），Linux 原生。
- 安装 Python 后**无需安装任何第三方依赖**（`requirements.txt` 为空占位）。
- 核心算法、测试、演示全部离线：不访问网络、账号、数据库、云 API、Docker。
- 示例数据由仓库内固定种子代码确定性生成，不依赖任何外部附件。

## 准确命令

```bash
# 必要测试（62 个用例，约 1 秒）
python -m unittest discover -s tests -v

# 演示（现场真实计算，整体约 0.1 秒；无 sleep、无预录结果）
python demo.py
```

在仓库根目录（本文件所在目录）执行。测试中“独立参考”
（`tests/reference.py`：单肽 ≤6 位置全枚举 + 链式 DP、多骨架全枚举）
**不导入被测核心**，期望值不由被测代码产生；与 `energy_of`/`scores_of`
的往返一致只作为补充断言。

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

### 状态语义（预算耗尽 ≠ 证明无解）

| 状态 | 含义 | 可用结论 |
|---|---|---|
| `EXACT` | 搜索完整结束（或剩余前沿全部被最优界剪掉） | `energy` 即已证最优，`global_lower_bound == energy` |
| `INFEASIBLE` | 证明不存在任何无硬冲突构象 | `assignment/energy/global_lower_bound` 均为 `None` |
| `BUDGET_EXHAUSTED` | 预算用完、未完成证明 | `global_lower_bound ≤ 真实最优 ≤ energy`（`energy` 可能为 `None`，即连一个可行叶都没找到；此时只给下界） |

## 多骨架接口（先最小化最坏能量，再最小化加权平均）

```python
from rotamer_select import (
    select_for_backbones, MultiBackboneProblem, BackboneSolveStatus,
)

# backbones：每骨架一个 (single_energies, pair_matrices) 二元组，
# 能量格式与单骨架完全相同（int 或 None，i<j 矩阵，行 i 列 j）；
# 所有骨架必须有相同位点数与每位点 rotamer 数（共用一组 rotamer）。
backbones = [
    (singles_b0, pairs_b0),
    (singles_b1, pairs_b1),
    (singles_b2, pairs_b2),
]
res = select_for_backbones(backbones)                 # 等权
res = select_for_backbones(backbones, weights=(1, 3, 2))  # 正整数权重

res.status                # BackboneSolveStatus.EXACT / INFEASIBLE
res.assignment            # 最优共用构象 tuple；无解为 None
res.scores                # 每骨架在该构象下的总能量（按输入顺序）；无解为 None
res.worst_energy          # max(scores)：被最小化的最坏骨架能量
res.weighted_sum          # sum(w_b * score_b)；加权平均 = weighted_sum / sum(weights)
res.weights               # 实际使用的权重（未给时全为 1）
res.assignments_evaluated # 枚举构象总数（乘积 k_0*...*k_{n-1}，至多 729）
res.feasible_assignments  # 对所有骨架都无硬冲突的构象数

# 重复评估（校验/归一化一次缓存；权重可在每次 select 时改变）
prob = MultiBackboneProblem(backbones)
prob.n, prob.counts, prob.backbone_count
prob.select(weights=(2, 1, 1))
prob.scores_of((0, 2, 1, 0))   # 指定构象的每骨架能量；硬冲突报错并指明骨架下标
```

**两级目标与并列规则**：枚举按构象字典序进行，保留首个达到最优目标的构象：

1. 最小化 `worst = max_b score_b`；
2. `worst` 并列时最小化整数加权和 `Σ w_b·score_b`
   （分母 `Σ w_b` 是与构象无关的常数，故最小化加权平均等价于最小化该整数和，
   全程无浮点；`weighted_average` 仅作展示用的分数属性）；
3. 两者都并列取字典序最小构象。

任一骨架上单体能或对能为 `None` 的构象即整体不可行；全部不可行时
`status=INFEASIBLE`、各结果字段为 `None`，这是枚举证明的结论而非错误。

### 多骨架规模与校验

- 位点数 **1..6**、每位点 rotamer 数 **1..3**、骨架数 **1..3**
  （常量 `MULTI_MAX_POSITIONS / MULTI_MIN_ROTAMERS / MULTI_MAX_ROTAMERS
  / MAX_BACKBONES`）；构象总数至多 3^6 = 729，直接全枚举，结果精确。
- 复用单骨架的严格校验：能量只接受真 `int`/`None`（拒绝 `float`、NaN/
  Infinity、`bool`），矩阵行列必须匹配，位置对 `i<j` 且不重复。
- 骨架间位点数或任一位置 rotamer 数不一致 → `RotamerInputError`
  （信息列出骨架 0 与出错骨架各自的结构）。
- `weights` 省略为等权；给出时必须与骨架数等长、元素为正整数
  （零/负/float/bool 一律拒绝，错误带 `weights[i]` 定位）。
- 校验在任何枚举前完成；非法输入不构造对象、不产生半成品状态；
  求解只读遍历输入（内部转 tuple），不修改用户数据。

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
  多骨架的加权平均比较也只用整数加权和；`BackboneSelection.weighted_average`
  是唯一的展示用浮点派生属性，不参与任何决策。
- **多骨架：全枚举两级字典序优化**。题面把规模限定在 6 位点 × 每点 3
  rotamer × 3 骨架（至多 729 构象），因此直接 `itertools.product` 按
  字典序枚举：逐构象在每骨架上独立计分（单体 + 成对，遇 `None` 即不可行），
  仅在“最坏能量严格更小”或“最坏相等且加权和严格更小”时更新保留解，
  天然得到字典序最小的两级最优。枚举精确，不设预算、无暂停/恢复、
  无分支下界认证，也不为更大实例扩展搜索。
- **失败不留部分变更**：本库对象是不可变结果（frozen dataclass），求解不
  修改用户传入的数据（输入只读遍历，内部转 tuple）；输入校验在任何搜索前
  完成，非法调用不会产生半成品状态。
- 实现刻意简单可读：显式栈 DFS + 预计算的行/全矩阵最小值，不用第三方求解器。

## 数值与误差口径

- 输入、输出、下界、增量全部是精确整数（或 `None`）；**不存在浮点误差**，
  README 与代码中均无“宽容差”。
- “精确”一词仅在 `EXACT` 状态下使用，含义是分支限界由可采纳下界剪枝
  完整性保证的整数最优；`BUDGET_EXHAUSTED` 下的 `energy` 只是上界。
- 本库只实现本题契约的数学子集（离散、整数；单肽至多 12 位置 × 5 rotamer、
  多骨架至多 6 位点 × 3 rotamer × 3 骨架、位置对上显式给矩阵），不宣称
  兼容任何完整行业能量函数/协议标准。

## 已知限制

- 单肽规模上限即接口边界：n ≤ 12、每位置 2..5 rotamer；超限直接拒绝。
  最坏情况仍是指数搜索（5^12 ≈ 2.4 亿构象），靠负能/硬冲突剪枝与预算兜底；
  预算耗尽只保证界，不保证找到最优。
- **多骨架能力是有意收窄的可枚举子集**：只接受 1..6 位点、每点 1..3
  rotamer、1..3 个骨架，全枚举至多 729 构象。本次迭代不做暂停/恢复、
  分支下界认证或更大实例的搜索；超过该范围的多骨架输入直接拒绝。
- 只支持成对位置项（无三体项等更高阶能量）；未给矩阵的位置对能量按 0 处理。
- “相邻”由用户显式给矩阵声明；库不假设距离/拓扑，也不做几何计算。
- 次优与固定增量在预算耗尽时分别给 `second_status` / `BEST_KNOWN` 与下界，
  增量同样可能只是界，不做跨子搜索的联合收紧。
- 多骨架选择要求所有骨架共用完全相同的位点结构与 rotamer 数（“共用一组
  rotamer”的字面含义）；它不搜索跨骨架的独立构象——要选的正是**一个**
  共用构象。硬冲突按构象整体过滤；权重只作用于加权平均，不影响硬约束。
- 不提供命令行产品/服务/文件格式；仅作为可 `import` 的库与固定 `demo.py`。

## 仓库结构

```
rotamer_select/          # 题库（核心算法）
  __init__.py            # 公开接口
  errors.py              # RotamerInputError
  validation.py          # 严格输入校验与上限（单肽/多骨架两套边界）
  solver.py              # 分支限界核心（纯整数）
  model.py               # 结果类型与 solve/optimize/RotamerProblem
  backbones.py           # 多骨架全枚举：minimax → 加权平均 → 字典序
tests/
  reference.py           # 独立参考：单肽全枚举(≤6)+链式 DP、多骨架全枚举
  datagen.py             # 固定种子确定性数据生成（含多骨架生成器）
  test_solver.py         # 36 个原用例（全部保留）
  test_backbones.py      # 26 个多骨架用例
demo.py                  # 固定输入演示（贪心失败 / 预算耗尽 / 多骨架 / 拒绝）
```
