"""输入解析与严格校验。

契约要点：
- 位置数 n: 1 <= n <= 12；每个位置 rotamer 数 2..5。
- 单体能: 每位置一个长度为 k_i 的序列，元素为 int 或 None。
- 对能: 每条 ``(i, j, matrix)``，要求 0 <= i < j < n，
  matrix 为 k_i 行 k_i 列（k_j 列）的矩形，元素 int 或 None。
  同一位置对至多给一个矩阵。
- 所有数值必须是真正的 int：拒绝 float（含 NaN/Infinity——能量域根本不接受浮点），
  特别地拒绝 bool（``True``/``False`` 是 Python 的 int 子类，按契约不得冒充整数）。
- 预算必须是正整数（不接受 bool）。
"""

import math

from .errors import RotamerInputError

MAX_POSITIONS = 12
MIN_ROTAMERS = 2
MAX_ROTAMERS = 5


def _is_real_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_energy_cell(x, where: str):
    if x is None:
        return
    if isinstance(x, bool) or not isinstance(x, int):
        if isinstance(x, float) and math.isnan(x):
            raise RotamerInputError(f"{where}: 拒绝 NaN（能量只接受 int 或 None）")
        if isinstance(x, float) and math.isinf(x):
            raise RotamerInputError(f"{where}: 拒绝 Infinity（能量只接受 int 或 None）")
        kind = type(x).__name__
        raise RotamerInputError(
            f"{where}: 能量元素必须是 int 或 None, 得到 {kind}（bool 不允许冒充整数）"
        )


def _check_positive_int(value, name: str):
    if isinstance(value, bool) or not isinstance(value, int):
        kind = type(value).__name__
        raise RotamerInputError(f"{name}: 必须是 int, 得到 {kind}（bool 不接受）")
    if value <= 0:
        raise RotamerInputError(f"{name}: 必须为正整数, 得到 {value}")


def validate_problem(single_energies, pair_matrices):
    """校验并归一化问题输入。

    返回 ``(n, counts, singles, pairs)``：

    - ``n``: 位置数
    - ``counts``: 每位置 rotamer 数（tuple[int, ...]）
    - ``singles``: ``tuple[tuple[int|None, ...], ...]`` 单体能
    - ``pairs``: ``tuple[(i, j, tuple[tuple[int|None, ...], ...]), ...]``，
      按 (i, j) 排序去重
    """
    # ---- 单体能 / 位置数 ----
    if not isinstance(single_energies, (list, tuple)):
        raise RotamerInputError(
            f"single_energies: 必须是 list/tuple 的序列, 得到 {type(single_energies).__name__}"
        )
    n = len(single_energies)
    if n == 0:
        raise RotamerInputError("single_energies: 至少需要 1 个位置")
    if n > MAX_POSITIONS:
        raise RotamerInputError(
            f"single_energies: 位置数 {n} 超过上限 {MAX_POSITIONS}"
        )

    counts = []
    singles = []
    for i, col in enumerate(single_energies):
        where = f"single_energies[{i}] (位置 {i})"
        if not isinstance(col, (list, tuple)):
            raise RotamerInputError(
                f"{where}: 必须是 list/tuple, 得到 {type(col).__name__}"
            )
        k = len(col)
        if not (MIN_ROTAMERS <= k <= MAX_ROTAMERS):
            raise RotamerInputError(
                f"{where}: rotamer 数 {k} 不在 [{MIN_ROTAMERS}, {MAX_ROTAMERS}]"
            )
        norm = []
        for r, v in enumerate(col):
            _validate_energy_cell(v, f"{where}[{r}]")
            norm.append(v)
        counts.append(k)
        singles.append(tuple(norm))

    # ---- 对能矩阵 ----
    if not isinstance(pair_matrices, (list, tuple)):
        raise RotamerInputError(
            f"pair_matrices: 必须是 list/tuple, 得到 {type(pair_matrices).__name__}"
        )

    seen = {}
    normalized_pairs = []
    for idx, item in enumerate(pair_matrices):
        where0 = f"pair_matrices[{idx}]"
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise RotamerInputError(
                f"{where0}: 每条必须是 (i, j, matrix) 三元组"
            )
        i, j, matrix = item
        if not _is_real_int(i) or not _is_real_int(j):
            raise RotamerInputError(
                f"{where0}: 位置下标 i, j 必须是 int（bool 不接受）"
            )
        if not (0 <= i < n and 0 <= j < n):
            raise RotamerInputError(
                f"{where0}: 位置对 ({i}, {j}) 越界, 合法范围 0..{n - 1}"
            )
        if i == j:
            raise RotamerInputError(f"{where0}: 位置对不能引用同一位置 (i == j == {i})")
        if i > j:
            raise RotamerInputError(
                f"{where0}: 位置对必须满足 i < j（matrix 行对应 i、列对应 j）; "
                f"得到 ({i}, {j}), 请改用 ({j}, {i}) 并相应转置矩阵"
            )
        a, b = i, j
        if (a, b) in seen:
            raise RotamerInputError(
                f"{where0}: 位置对 ({a}, {b}) 重复定义（另见 pair_matrices[{seen[(a, b)]}]）"
            )
        seen[(a, b)] = idx

        mwhere = f"{where0} (位置对 ({a}, {b}))"
        if not isinstance(matrix, (list, tuple)):
            raise RotamerInputError(
                f"{mwhere}: matrix 必须是 list/tuple, 得到 {type(matrix).__name__}"
            )
        ki, kj = counts[a], counts[b]
        if len(matrix) != ki:
            raise RotamerInputError(
                f"{mwhere}: 期望 {ki} 行（位置 {a} 的 rotamer 数）, 实际 {len(matrix)} 行"
            )
        rows = []
        for r, row in enumerate(matrix):
            if not isinstance(row, (list, tuple)):
                raise RotamerInputError(
                    f"{mwhere}[{r}]: 行必须是 list/tuple, 得到 {type(row).__name__}"
                )
            if len(row) != kj:
                raise RotamerInputError(
                    f"{mwhere}[{r}]: 期望 {kj} 列（位置 {b} 的 rotamer 数）, "
                    f"实际 {len(row)} 列"
                )
            nrow = []
            for c, v in enumerate(row):
                _validate_energy_cell(v, f"{mwhere}[{r}][{c}]")
                nrow.append(v)
            rows.append(tuple(nrow))
        normalized_pairs.append((a, b, tuple(rows)))

    normalized_pairs.sort(key=lambda t: (t[0], t[1]))
    return n, tuple(counts), tuple(singles), tuple(normalized_pairs)


def validate_budget(budget) -> int:
    _check_positive_int(budget, "budget")
    return budget


def validate_fixed(n: int, counts, fixed) -> dict:
    """校验固定约束映射 ``{position: rotamer}``。"""
    if fixed is None:
        return {}
    if not isinstance(fixed, dict):
        raise RotamerInputError(
            f"fixed: 必须是 dict {{position: rotamer}}, 得到 {type(fixed).__name__}"
        )
    result = {}
    for p, r in fixed.items():
        if not _is_real_int(p):
            raise RotamerInputError(
                f"fixed 的位置键 {p!r}: 必须是 int（bool 不接受）"
            )
        if not (0 <= p < n):
            raise RotamerInputError(f"fixed: 位置 {p} 越界, 合法范围 0..{n - 1}")
        if not _is_real_int(r):
            raise RotamerInputError(
                f"fixed[{p}]: rotamer 必须是 int（bool 不接受）, 得到 {type(r).__name__}"
            )
        if not (0 <= r < counts[p]):
            raise RotamerInputError(
                f"fixed[{p}]: rotamer {r} 越界, 位置 {p} 合法范围 0..{counts[p] - 1}"
            )
        result[p] = r
    return result


def _validate_label(label, where: str):
    if isinstance(label, bool):
        raise RotamerInputError(f"{where}: 类别标签不接受 bool")
    if isinstance(label, str):
        if not label:
            raise RotamerInputError(f"{where}: 类别标签不能为空字符串")
        return
    if isinstance(label, int):
        return
    raise RotamerInputError(
        f"{where}: 类别标签必须是非空 str 或 int, 得到 {type(label).__name__}"
    )


def validate_categories(n: int, counts, categories):
    """校验类别标签表，返回 ``tuple[tuple[label, ...], ...]``（与单体能同形）。"""
    if categories is None:
        return None
    if not isinstance(categories, (list, tuple)):
        raise RotamerInputError(
            f"categories: 必须是 list/tuple, 得到 {type(categories).__name__}"
        )
    if len(categories) != n:
        raise RotamerInputError(
            f"categories: 长度必须等于位置数 {n}, 实际 {len(categories)}"
        )
    out = []
    for p, col in enumerate(categories):
        where = f"categories[{p}] (位置 {p})"
        if not isinstance(col, (list, tuple)):
            raise RotamerInputError(
                f"{where}: 必须是 list/tuple, 得到 {type(col).__name__}"
            )
        if len(col) != counts[p]:
            raise RotamerInputError(
                f"{where}: 长度必须等于该位置 rotamer 数 {counts[p]}, 实际 {len(col)}"
            )
        row = []
        for r, label in enumerate(col):
            _validate_label(label, f"{where}[{r}]")
            row.append(label)
        out.append(tuple(row))
    return tuple(out)


def validate_quotas(quotas, used_labels):
    """校验配额映射 ``{label: (lo, hi)}``，返回规范化的 dict（保持插入序）。

    ``used_labels`` 是类别表中实际出现过的标签集合（无 categories 时为 None）；
    配额键必须是出现过类别的子集。
    """
    if quotas is None:
        return None
    if used_labels is None:
        raise RotamerInputError("quotas: 指定了配额但未提供 categories")
    if not isinstance(quotas, dict):
        raise RotamerInputError(
            f"quotas: 必须是 dict {{类别: (lo, hi)}}, 得到 {type(quotas).__name__}"
        )
    out = {}
    for label, bounds in quotas.items():
        _validate_label(label, "quotas 的键")
        if label not in used_labels:
            raise RotamerInputError(
                f"quotas: 类别 {label!r} 未在 categories 中出现"
            )
        where = f"quotas[{label!r}]"
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise RotamerInputError(f"{where}: 必须是 (lo, hi) 二元组")
        lo, hi = bounds
        for name, v in (("lo", lo), ("hi", hi)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise RotamerInputError(
                    f"{where}: {name} 必须是非负 int（bool 不接受）, "
                    f"得到 {type(v).__name__}"
                )
            if v < 0:
                raise RotamerInputError(f"{where}: {name} 必须为非负整数, 得到 {v}")
        if lo > hi:
            raise RotamerInputError(f"{where}: 下界 {lo} 大于上界 {hi}")
        out[label] = (lo, hi)
    return out
