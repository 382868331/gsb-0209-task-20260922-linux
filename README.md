# rotamer_pack —— 肽链侧链构象的离散能量选择

离线、零第三方依赖的可复用 Python 库：给定若干链上位置各自的候选
rotamer、单体能矩阵与指定相邻位置对的成对能矩阵（整数，`None` 表示硬
冲突），用**分支限界搜索**求总能量最小的 rotamer 组合，并给出：

- 全局最优构象及其精确能量；
- 第二个**不同**构象的能量（能量并列时按 rotamer 索引序列字典序取最小）；
- 每个位置固定为每个 rotamer 时，相对最优的最优代价增量（固定后无解为
  `None`）；
- 预算（节点展开数）耗尽时的最优已知解与**全局可采纳下界**，状态与
  “已证明无解”严格区分，绝不把未完成搜索报告为精确结果。

## 运行环境

- Python **3.14.7**（仅使用标准库；低版本 3.10+ 大概率可用，但以 3.14.7
  为验收环境）。
- Linux 原生运行；装好 Python 后**无需安装任何其他依赖**，无需网络、
  账号、密钥、数据库、中间件、云 API 或 Docker。
- 全部示例数据由仓库内确定性代码生成/硬编码。

## 快速开始

```python
from rotamer_pack import RotamerProblem

singles = [                 # 每位置 2..5 个 rotamer 的单体能
    [0, -1],
    [0, 0],
    [0, 0],
]
pairs = [                   # 指定位置对 (i, j), i < j；未列出的对能量为 0
    ((0, 1), [[-5, None],  # None = 该 rotamer 组合硬碰撞
              [None, 0]]),
    ((1, 2), [[-6, -6],
              [0, 0]]),
]

rep = RotamerProblem(singles, pairs).solve()
print(rep.status)               # 'optimal'
print(rep.assignment, rep.energy)        # (0, 0, 0) -11
print(rep.second_assignment, rep.second_energy)  # (0, 0, 1) -11
print(rep.costs[0])             # (0, 10)：位置0固定为 rotamer1 的代价增量
```

便捷函数 `minimize_energy(singles, pairs=(), budget=None)` 等价于
`RotamerProblem(singles, pairs).solve(budget)`。

## 运行测试与演示

```bash
python -m unittest discover -s tests -v   # 必要测试（约 1 秒）
python demo.py                            # 固定输入演示（远小于 8 秒）
```

- `tests/reference_bruteforce.py` 是**独立全枚举参考**（最多 6 位置），
  不导入被测包、不调用被测核心；最优/次优/固定代价的期望值全部由它
  独立产生。`evaluate` 的往返一致性只作为补充断言。
- `demo.py` 打印：(1) 一个局部贪心失败、分支限界得到全局最优的正常
  结果（含次优、代价增量、预算中断）；(2) 一个真实触发的拒绝
  （错误矩阵维度）与 bool 拒绝。无 sleep、无预录结果。

## 公开接口与数据结构

### `RotamerProblem(singles, pairs=())`

不可变问题实例，可重复 `solve()`，调用间无共享状态。

| 参数 | 形态 | 约束 |
|---|---|---|
| `singles` | 序列的序列（list/tuple） | 外层长度 1..12；每个位置 2..5 个 rotamer；元素为 `int` 或 `None` |
| `pairs` | `(key, matrix)` 项的序列 | `key=(i,j)` 且 `0<=i<j<n`、不可重复；matrix 有 `n_i` 行，每行恰好 `n_j` 项，元素为 `int` 或 `None` |

- 能量必须是整数 `int`，**允许负数**；`None` 为硬冲突（单体 `None`
  表示该 rotamer 不可用；矩阵项 `None` 表示该对组合碰撞）。
- 未在 `pairs` 中出现的位置对，相互作用视为 0（因此可以只给“相邻指定
  位置对”，也可以给任意 i<j 的对）。
- `problem.evaluate(assignment)` 返回完整构象的精确整数能量；构象含
  越界下标或硬冲突时抛 `ProblemValidationError`。

### `solve(budget=None) -> SearchReport`

`budget` 为最多展开的搜索节点数（根节点计 1；该预算覆盖主搜索与全部
固定代价子搜索）。`None` 或足够大时给出精确结果。

`SearchReport`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `status` | `'optimal'` / `'infeasible'` / `'budget_exhausted'` 三选一 |
| `energy`, `assignment` | 已知最优能量与其 rotamer 索引元组；无解/预算内未找到可行叶时为 `None` |
| `second_energy`, `second_assignment` | 与最优不同的次优构象；并列时为字典序最小的第二个；唯一可行构象时为 `None` |
| `costs` | `costs[i][r]`：位置 i 固定为 rotamer r 时最优总能量减 `energy`（非负整数）；固定后无解为 `None`；最优槽位为 0。预算在该阶段耗尽则**整体为 `None`**（不交付半成品） |
| `lower_bound` | 终止时对真实最优能量的全局可采纳下界；`optimal` 时等于 `energy`；`infeasible` 时为 `None` |
| `energy_proven` | `energy/assignment` 是否已被证明全局最优（独立于代价增量阶段是否耗尽） |
| `costs_complete` | 代价增量表是否完整 |
| `nodes_expanded`, `budget` | 实际展开节点数与本次预算 |
| `is_optimal` / `is_infeasible` / `budget_exhausted` / `feasible` | 便捷属性；`feasible` 为 True/False/None（None=预算内可行性未知） |

## 输入上限

- 位置数：**1..12**；每位置 rotamer 数：**2..5**。
- 这些是接口边界（构造时强制校验）。上限规模（12×5）下朴素枚举不可行；
  若需要在超时前得到有证书的答案，请显式传 `budget`，用
  `status='budget_exhausted'` 的 `energy`（可行上界）与 `lower_bound`
  （可采纳下界）夹住真实最优。

## 错误语义

- 所有公开错误继承 `RotamerError`；输入问题统一抛
  `ProblemValidationError`（同时是 `ValueError`），消息带可定位下标，
  例如 `singles[1][1]: 能量不能是 NaN`、
  `pairs[0] matrix: 行数 3 与位置 0 的 rotamer 数 2 不一致`。
- 数值输入拒绝 `NaN`/`+Infinity`/`-Infinity`；能量矩阵只接受整数
  `int` 与 `None`，**浮点值（含 1.0）一律拒绝**，不做隐式取整。
- 整数参数拒绝 `bool`（`True` 不会被静默当作 1），包括能量项与下标。
- 矩阵行列数、位置下标越界、`i>=j`、重复位置对、rotamer 数越界、
  非序列结构、负/非整数 `budget` 均被拒绝；不静默纠正、不补默认值。
- 失败原子性：纯函数式校验与求解，输入非法时不构造对象；预算在代价
  增量阶段耗尽时 `costs` 整体为 `None`，不留下未说明的部分结果。
- **预算耗尽 ≠ 证明无解**：前者 `status='budget_exhausted'`（可能附
  部分结果与下界），后者 `status='infeasible'`（已完整证明不存在可行
  构象，能量字段为 `None`）。

## 算法与设计取舍

1. **深度优先分支限界**，按位置顺序、rotamer 索引顺序展开，第一个被
   完整证明的最优叶即字典序最小构象；次优通过跟踪两个不同 incumbent
   获得，能量相等时不替换（DFS 顺序保证字典序）。
2. **可采纳下界**（节点状态：前 d 个位置已定，路径代价 g）：
   - 已定项：精确贡献（在 g 中）；
   - 每个未定位置：该列最小单体能；
   - 一端已定的成对项：所选行的最小非 `None` 值；
   - 两端未定的成对项：整张矩阵的最小非 `None` 值。
   全部最小值不存在（全 `None`）时该子问题判不可行并剪枝。
3. **前向检查**：每个未定位置必须至少有一个与*已分配*邻居相容的
   rotamer。这是便宜的额外剪枝，不影响下界正确性。
4. **预算中断的全局下界**：开放前沿 = 已生成未展开子节点的懒删除最小
   堆 ∪ 中断瞬间活动调用路径上各节点的下界（路径覆盖尚未惰性生成的
   兄弟）；被能量剪枝的分支由 incumbent 背书。取其最小值（若已有可行
   叶，再与 incumbent 能量取小）即为合法全局下界。测试对 12 组实例 ×
   预算 0..30 逐一验证 `lower_bound <= 独立枚举真实最优`。
5. **每位置固定代价增量**：对每个非最优 (i,r) 以单固定约束再跑一次
   分支限界，减最优能量；与最优相同槽位直接为 0。预算在主搜索与全部
   子搜索间共享。
6. 搜索中 g 增量只累加“与已分配邻居”的成对项；每项恰在其第二个端点
   被分配时计入一次，无重复无遗漏。
7. 纯整数运算；除演示打印的耗时外，库本身不产生浮点结果。

## 误差口径

本库输出的能量与代价增量全部是**精确整数**（Python 任意精度 int），
不存在浮点舍入误差，也没有容差参数。`demo.py` 中显示的毫秒/秒耗时
来自 `time.perf_counter()`，仅用于展示，不参与任何计算或判定。

## 已知限制

- 只实现本题契约的数学子集：位置 1..12、每位置 2..5 rotamer、成对
  相互作用（任意指定的 i<j 位置对）；不宣称兼容任何行业标准（如
  CHARMM/ROSETTA 能量项格式）、不做力场计算、不读结构文件。
- 上界规模（12×5、约 2.44 亿组合）最坏情况下无预算搜索可能耗时约十秒
  量级；需要硬时限时使用 `budget`，接受带证书的近似答案。
- 代价增量只做“单点固定”，不计算同时固定多个位置的联合增量。
- 不做多线程/增量复用；每次 `solve()` 独立。
