# rotamer_select — 肽链侧链构象的离散能量选择

离线结构生物学小工具：给定一条最多 12 个位置的肽链，每位置 2..5 个侧链
rotamer，用**分支限界（branch-and-bound）**求总能量最小的构象，并给出：

- 字典序最小的最优构象与总能量（能量并列时按 rotamer 下标序列字典序）；
- **可选类别配额**：每个 rotamer 归属唯一类别，调用方可按类别限定全局
  选择数量的闭区间 `[lo, hi]`（非负整数）；结果附带类别计数；
- **第二个不同构象**及其能量（次优并列时同样取字典序最小）；
- **逐位置固定代价增量**：把位置 p 强制固定到任意另一个 rotamer 后，
  可达到的最小总能量相对最优的增量（硬冲突 → 证明无解）；
- 节点预算耗尽时返回**当前最优已知解 + 可信全局下界**，严格区分于“证明无解”，
  绝不声称精确。

## 运行环境

- **Python 3.14.7**（标准库 only），Linux 原生。
- 安装 Python 后**无需安装任何第三方依赖**（`requirements.txt` 为空占位）。
- 核心算法、测试、演示全部离线：不访问网络、账号、数据库、云 API、Docker。
- 示例数据由仓库内固定种子代码确定性生成，不依赖任何外部附件。

## 准确命令

```bash
# 必要测试（56 个用例，约 1 秒）
python -m unittest discover -s tests -v

# 演示（现场真实计算，约 0.05 秒，上限约 8 秒预算内；无 sleep、无预录结果）
python demo.py
```

在仓库根目录（本文件所在目录）执行。测试中“独立参考”
（`tests/reference.py`：≤6 位置全枚举 + 链式动态规划）**不导入被测核心**，
期望值不由被测代码产生；与 `energy_of` 的往返一致只作为补充断言。

> **口径说明**：类别配额是**合成设计指标**（例如“某类侧链至少出现 k 次”
> 的工程约束），用于组合约束求解演示；本库结果**不构成真实蛋白稳定性结论**。

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

### 类别与配额（可选）

```python
categories = [["hyd", "pol"], ["hyd", "pol"], ...]  # 与 single_energies 同形
quotas = {"pol": (2, 3)}                            # {类别: (lo, hi)}
```

- 每个 rotamer **恰好归属一个类别**（单一归属）；标签为非空 `str` 或 `int`
  （`bool`/`None`/空串一律拒绝）。
- `quotas` 限定某类别在**整条链全局**被选中次数的闭区间 `[lo, hi]`：
  边界为非负整数（`bool` 拒绝）、`lo <= hi`、键必须是出现过的类别；
  未列出的类别不限制数量。零配额 `(0, 0)` 表示禁用该类别。
- 配额是**硬约束**：违反配额的构象直接不可行，不以罚分近似；
  硬冲突（`None`）同样仍然禁止，不与配额混为罚分。
- 每个位置仍恰好选一个 rotamer 的原要求不变。

## 公开接口

```python
from rotamer_select import solve, optimize, energy_of, RotamerProblem, SolveStatus

# 1) 只求最优
res = solve(single_energies, pair_matrices,
            fixed=None, categories=None, quotas=None, budget=200_000)
res.status                # SolveStatus.EXACT / INFEASIBLE / BUDGET_EXHAUSTED
res.assignment            # tuple[int,...]，字典序最小最优构象；无解/未得解时 None
res.energy                # 最优（或当前最优）总能量，可能 None
res.global_lower_bound    # 真实最优的可信整数下界；INFEASIBLE 时 None
res.nodes_expanded        # 实际展开节点数
res.exact                 # bool：最优性是否已证明
res.category_counts       # {类别: 数量}；未设 categories 或无构象时为 None

# 附加固定约束求解（如 fixed={2: 0} 表示位置 2 必须取 rotamer 0）
solve(singles, pairs, fixed={2: 0})

# 2) 完整分析：最优 + 次优 + 全部固定代价增量
ana = optimize(singles, pairs, categories=None, quotas=None,
               budget=200_000)
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
prob = RotamerProblem(singles, pair_matrices, categories=None, quotas=None)
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
- **配额剪枝（纯计数，不动能量下界）**：分支时若某类别已选数将超过 `hi`
  则剪枝；若剩余位置中能取某类别的位置数加上已选数仍不足 `lo` 则剪枝；
  叶节点再终检一次所有下界。根节点先做计数预检（下界总和 > n、
  各类别上限总和 < n、某类别可选位置不足 `lo` 均直接判 `INFEASIBLE`）。
  能量下界保持原有可采纳性：负成对能照常计入矩阵最小值，绝不为了剪枝
  丢掉负对能；预算耗尽时上界（当前最优）与下界（全局下界）照旧分开报告。
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
- 配额可行性剪枝只做计数层面（类别可选位置数），不把对能冲突纳入
  预检；冲突导致的不可行仍由搜索证明，最坏情况不减少指数开销。
- 类别为单一归属（每 rotamer 一个标签），不支持多标签/层级类别。
- 配额只约束全局总数，不支持“每段/每窗口”类局部配额。
- 不提供命令行产品/服务/文件格式；仅作为可 `import` 的库与固定 `demo.py`。

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
  test_solver.py         # 无配额核心行为用例
  test_quotas.py         # 类别配额用例（含局部最低能破坏配额、
                         # 类别只在冲突位置可选、零配额、同分布局）
demo.py                  # 固定输入演示（贪心失败 / 预算耗尽 / 拒绝边界）
```
