import numpy as np
import scipy.sparse as sparse
from typing import List, Union, Sequence, Optional, Type
from .expressions import Expr, ExprNumpy


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


def main():
    m = Model()
    a = m.var(2)
    b = m.var(2)

    # Check sum with constant and with vector.
    assert np.allclose(np.array(a + 2.34)[:, 0], 2.34)
    assert np.allclose(np.array(a + np.array([2.34, 1.23]))[:, 0], np.array([2.34, 1.23]))
    assert np.allclose(np.array(a + np.array([1.13, 3.01]))[:, 0], [1.13, 3.01])

    print(a*2)
    print(a*np.array([2, 3]))

    print((a + 1) * np.array([2, 3]))

    d = a + np.array([1, 3]) + 3
    m += a <= b
    m += a >= b
    m += a == b
    m += a == d
    print(m.combine_cons())

    m = Model(max_vars=4)
    a = m.var(3)
    m += a <= np.array([1, 2, 3])
    m += a >= 0
    # m += a[1] + a[2] <= 2
    # m += a[0] + a[1] <= 1
    m.objective = -1 * a.sum()  # Maximise sum of them.
    res, x = m.solve()
    print(x)
    print(a.raw() @ x)


if __name__ == '__main__':
    main()
