import numpy as np
import scipy.sparse as sparse
from typing import List, Union, Sequence, Optional, Type
from .expressions import Expr, ExprNumpy

__all__ = [
    'Model',
    'Expr',
    'ExprNumpy',
]


class Model:

    def __init__(self,
                 max_vars: int = 5,
                 expr: Expr = ExprNumpy,
                 dtype: np.dtype = np.float):
        self.max_vars: int = max_vars + 1  # One extra for the constant.
        self.next_var_idx: int = 0  # What number variable model is up to.
        self.dtype: np.dtype = dtype
        self.expr: Expr = expr
        self.int_vars = np.zeros(max_vars, dtype=np.bool)  # True where var is int

        # Constraints are represented as linear expressions which must be <= 0
        self.cons: List[Expr] = []

        self.k: Expr = self.var()  # A linear expression equal to 1.0

        self.objective: Optional[Expr] = None

    def var(self, n=1, integer: bool = False) -> Expr:
        """ Return linear expressions representing new variables. """
        start_idx = self.next_var_idx
        self.next_var_idx += n
        exprs = self.expr.zeros(n, self.max_vars, dtype=self.dtype)
        var_idxs = np.arange(n)
        exprs[var_idxs, start_idx + var_idxs] = 1.0
        self.int_vars[var_idxs] = integer
        return exprs

    def add_constr(self, constr: Expr):
        self.cons.append(constr)

    def __iadd__(self, other: Expr):
        self.add_constr(other)
        return self

    def combine_cons(self) -> Expr:
        return self.cons[0].vstack(self.cons)

    def solve(self):
        from scipy.optimize import linprog
        cons = self.combine_cons()

        res = linprog(
            c=self.objective[1:],  # Remove constants,
            A_ub=cons[:, 1:],  # All but the constants
            b_ub=cons[:, 0] * -1.0,  # Constants only
            bounds=(None, None)
        )
        x = np.hstack([[0], res.x])
        return res, x
