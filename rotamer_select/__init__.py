"""肽链侧链 rotamer 离散能量选择库。

只依赖 Python 3.14 标准库；能量均为整数或 ``None``（硬冲突），
求解全程整数运算，无浮点误差。

典型用法::

    from rotamer_select import optimize

    result = optimize(single_energies, pair_matrices)
    print(result.assignment, result.energy, result.status)
"""

from .errors import RotamerError, RotamerInputError
from .validation import (
    MAX_POSITIONS,
    MIN_ROTAMERS,
    MAX_ROTAMERS,
    MULTI_MAX_POSITIONS,
    MULTI_MAX_ROTAMERS,
    MIN_BACKBONES,
    MAX_BACKBONES,
)
from .solver import DEFAULT_NODE_BUDGET
from .model import (
    RotamerProblem,
    SolveStatus,
    SolveResult,
    FixedCostStatus,
    FixedCostEntry,
    AnalysisResult,
    solve,
    optimize,
    energy_of,
)
from .multi import (
    MultiBackboneProblem,
    BackboneResult,
    solve_backbones,
)

__all__ = [
    "MAX_POSITIONS",
    "MIN_ROTAMERS",
    "MAX_ROTAMERS",
    "MULTI_MAX_POSITIONS",
    "MULTI_MAX_ROTAMERS",
    "MIN_BACKBONES",
    "MAX_BACKBONES",
    "DEFAULT_NODE_BUDGET",
    "RotamerError",
    "RotamerInputError",
    "RotamerProblem",
    "SolveStatus",
    "SolveResult",
    "FixedCostStatus",
    "FixedCostEntry",
    "AnalysisResult",
    "solve",
    "optimize",
    "energy_of",
    "MultiBackboneProblem",
    "BackboneResult",
    "solve_backbones",
]
