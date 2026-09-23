"""异常类型。

注意：“证明无解”不是异常——它由 ``SolveStatus.INFEASIBLE`` 表达，
以便与 ``SolveStatus.BUDGET_EXHAUSTED``（预算耗尽、未能证明）严格区分。
"""


class RotamerError(Exception):
    """本库所有异常的公共基类。"""


class RotamerInputError(RotamerError, ValueError):
    """输入不满足契约（维度、取值、NaN/Infinity、bool 冒充整数等）。

    错误信息尽量带可定位坐标，例如::

        pair_matrix[3] (位置对 (1, 4)): 期望 5 行, 实际 4 行
    """
