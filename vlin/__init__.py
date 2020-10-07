import numpy as np
import scipy.sparse as sparse
from typing import List, Union, Sequence, Optional
from numbers import Real


class Expr:
    """ An array of linear expressions: each expression is a vector of var coefficients. """

    def raw(self):
        """ Convert expression to its raw underlying data type. """
        raise NotImplementedError

    def sum(self) -> 'Expr':
        """ Return sum of vector as a vector with one element. """
        raise NotImplementedError

    @classmethod
    def zeros(self, shape: int, max_vars: int, dtype=np.float64) -> 'Expr':
        """ Return vector instance of self all zeros. """
        raise NotImplementedError

    @classmethod
    def vstack(self, tup: Sequence['Expr']) -> 'Expr':
        """ apply vstack to given expressions. """
        raise NotImplementedError

    def __add__(self, other: Union['Expr', Real, np.ndarray]) -> 'Expr':
        """ Implement me """
        raise NotImplementedError

    def __mul__(self, other: Union[Real, np.ndarray]) -> 'Expr':
        """ Implement me """
        raise NotImplementedError

    def __le__(self, other: Union['Expr', Real]) -> 'Expr':
        """ x <= y  =>  x-y <= 0 """
        raise NotImplementedError

    def __ge__(self, other: Union['Expr', Real]) -> 'Expr':
        """ Negative of less than or equal. """
        return -1.0 * self.__le__(other)

    def __eq__(self, other: Union['Expr', Real]) -> 'Expr':
        """ x == y  =>  x-y >= 0 AND x-y <= 0 """
        con = other <= self
        return self.vstack([con, -1.0*con])

    def __sub__(self, other: Union['Expr', Real]) -> 'Expr':
        return self.__add__(-1.0 * other)

    def __truediv__(self, other: Real) -> 'Expr':
        return self.__mul__(1.0/other)

    def __radd__(self, other: Union['Expr', Real]) -> 'Expr':
        return self.__add__(other)

    def __rsub__(self, other: Union['Expr', Real]) -> 'Expr':
        return (-1*self).__add__(other)

    def __rmul__(self, other: Real) -> 'Expr':
        return self.__mul__(other)

    def __iadd__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError


class ExprNumpy(Expr, np.ndarray):

    def __new__(cls, input_array, info=None):
        # See https://numpy.org/devdocs/user/basics.subclassing.html
        obj = np.asarray(input_array).view(cls)
        obj.base_cast = np.array
        return obj

    def __array_finalize__(self, obj):
        # See https://numpy.org/devdocs/user/basics.subclassing.html
        if obj is None: return
        self.base_cast = getattr(obj, 'base_cast', None)

    def raw(self) -> np.ndarray:
        """ Convert expression to numpy array """
        return np.array(self)

    def sum(self) -> 'ExprNumpy':
        return self.raw().sum(axis=-2)

    @classmethod
    def zeros(cls, shape: int, max_vars: int, dtype=np.float64) -> 'ExprNumpy':
        return cls(np.zeros((shape, max_vars), dtype=dtype))

    @classmethod
    def vstack(cls, tup: Sequence['ExprNumpy']) -> 'ExprNumpy':
        return cls(np.vstack(tup))

    def __add__(self, other: Union['ExprNumpy', Real, np.ndarray]) -> 'ExprNumpy':
        """ Implement me """
        if not isinstance(other, Expr):
            expr = np.zeros(self.shape, dtype=self.dtype)
            expr[..., 0] = 1  # Last axis
            other = np.array(other)
            expr *= np.expand_dims(other, tuple(range(other.ndim, self.ndim)))
        else:
            expr = other.raw()
        return self.__class__(np.add(self, expr))

    def __mul__(self, other: Union[Real, np.ndarray]) -> 'ExprNumpy':
        """ Implement me """
        return np.multiply(self, np.expand_dims(other, -1))

    def __le__(self, other: Union['ExprNumpy', Real, float]) -> 'ExprNumpy':
        """ x <= y  =>  x-y <= 0 """
        if not isinstance(other, Expr):
            expr = np.zeros(self.shape, dtype=self.dtype)
            expr[..., 0] = 1  # Last axis
            other = np.array(other)
            expr *= np.expand_dims(other, tuple(range(other.ndim, self.ndim)))
        else:
            expr = other.raw()
        return np.subtract(self, expr)


class Model:

    def __init__(self, max_vars: int = 5):
        self.max_vars: int = max_vars + 1  # One extra for the constant.
        self.next_var_idx: int = 0  # What number variable model is up to.
        self.dtype = np.float64
        self.int_vars = np.zeros(max_vars, dtype=np.bool)  # True where var is int

        # Constraints are represented as linear expressions which must be <= 0
        self.cons: List[Expr] = []

        self.k = self.var()  # A linear expression equal to 1.0

        self.objective: Optional[Expr] = None

    def var(self, n=1, integer: bool = False) -> Expr:
        """ Return linear expressions representing new variables. """
        start_idx = self.next_var_idx
        self.next_var_idx += n
        exprs = ExprNumpy.zeros(n, self.max_vars, dtype=self.dtype)
        var_idxs = np.arange(n)
        exprs[var_idxs, start_idx + var_idxs] = 1.0
        self.int_vars[var_idxs] = integer
        return exprs

    def add_constr(self, constr: 'ExprNumpy'):
        self.cons.append(constr)

    def __iadd__(self, other: 'ExprNumpy'):
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
