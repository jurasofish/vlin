import numpy as np
import scipy.sparse as sparse
from typing import List, Union, Sequence, Optional, Type, Iterable
from .expressions import Expr, ExprNumpy
import vlin


__all__ = [
    "Model",
]


class Model:
    def __init__(
        self, max_vars: int = 5, expr: Expr = ExprNumpy, dtype: np.dtype = np.float
    ):
        self.max_vars: int = max_vars + 1  # One extra for the constant.
        self.next_var_idx: int = 0  # What number variable model is up to.
        self.dtype: np.dtype = dtype
        self.expr: Expr = expr
        self.int_vars = np.zeros(max_vars, dtype=np.bool)  # True where var is int

        # Constraints are represented as linear expressions which must be <= 0
        self.cons: List[Expr] = []

        self.k: Expr = self.var()  # A linear expression equal to 1.0

        self.objective: Optional[Expr] = None

    def var(self, n=1, integer: Union[bool, Iterable[bool]] = False) -> Expr:
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
        return self.expr.vstack(self.cons)

    def solve_scipy(self):
        if not np.isclose(self.int_vars.sum(), 0):
            raise vlin.IntegerNotSupported(
                "scipy linprog does not support integer variables."
            )
        from scipy.optimize import linprog

        cons = self.combine_cons()

        res = linprog(
            c=self.objective[1:],  # Remove constants,
            A_ub=cons[:, 1:],  # All but the constants
            b_ub=cons[:, 0] * -1.0,  # Constants only
            bounds=(None, None),
        )
        x = np.hstack([[0], res.x])
        return res, x

    def solve_cylp(self, verbose: bool = True):
        """ Inspired by the cvxpy cbc layer, ty :) """
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray

        cons = self.combine_cons()
        n = int(self.next_var_idx - 1)  # Number of variables.

        # Maximize c@x s.t. A@x <= b  (variable bounds done by constraints)
        c = self.objective[1 : n + 1].raw()  # Objective coefficients, no constants.
        A = cons[:, 1 : n + 1].raw()  # Constraint coefficients.
        b = cons[:, 0].raw() * -1.0  # Constraint constants.

        model = CyLPModel()
        x = model.addVariable("x", n)  # Variables
        model.objective = c

        # I hate this so much. Casting A to a matrix causes A.__mul__ to be
        # be called, which raises NotImplemented, so then x.__rmul__ is called
        # which gets the job done. Using np.matmul or np.dot doesn't
        # trigger x.__rmul__
        # model.addConstraint(np.matrix(A) * x <= CyLPArray(b))  # Works
        model.addConstraint(x.__rmul__(A) <= CyLPArray(b))

        model = CyClpSimplex(model)  # Convert model

        model.logLevel = 0 if not verbose else model.logLevel

        is_integer_model = not np.isclose(self.int_vars.sum(), 0)
        if is_integer_model:
            model.setInteger(x[np.argwhere(self.int_vars)])
            cbcModel = model.getCbcModel()
            cbcModel.solve()
            status = cbcModel.status
            solmodel = cbcModel
        else:
            status = model.initialSolve()
            solmodel = model

        sol_x = np.hstack(  # Pad back to original shape
            (
                [1.0],  # Special constant 1.0 variable.
                solmodel.primalVariableSolution["x"],  # Actual solution
                np.zeros(self.max_vars - n - 1),  # Unused vars
            )
        )

        solution = {"status": status, "primal": sol_x, "value": solmodel.objectiveValue}
        return solution, solution["primal"]
