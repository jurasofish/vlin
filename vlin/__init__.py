import numpy as np
import scipy.sparse as sparse
from typing import List, Union
from numbers import Real


class ExprBase:
    """ An array of linear expressions: each expression is a vector of var coefficients. """

    def raw(self):
        """ Convert expression to raw sparse matrix. """
        raise NotImplementedError

    def __le__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x <= y  =>  x-y <= 0 """
        return [self - other]

    def __ge__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x >= y  =>  x-y >= 0  =>  y-x <= 0 """
        return [other - self]

    def __eq__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x == y  =>  x-y >= 0 AND x-y <= 0 """
        return (other <= self) + (self <= other)  # List of two linear expressions.

    def __add__(self, other: Union['Expr', Real, np.ndarray]) -> 'Expr':
        raise NotImplementedError

    def __mul__(self, other: Union[Real, np.ndarray]) -> 'Expr':
        raise NotImplementedError

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

    # TODO: yada yada fill out the rest of these...


class Expr(sparse.csr_matrix, ExprBase):
    """ A vector of linear expressions: Each row is a vector of var coefficients. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = sparse.csr_matrix

    def raw(self) -> sparse.csr_matrix:
        """ Convert expression to raw sparse matrix. """
        return self.base(self)

    def __add__(self, other: Union['Expr', Real, np.ndarray]) -> 'Expr':
        if isinstance(other, (Real, np.ndarray)):
            expr = self.base(self.shape, dtype=self.dtype)
            expr[:, 0] = 1
            if isinstance(other, np.ndarray):
                other = np.expand_dims(other, -1)
            expr = expr.multiply(other)  # `expr * k` doesn't work ...
            return super().__add__(expr)
        return super().__add__(other)

    def __mul__(self, other: Union[Real, np.ndarray]) -> 'Expr':
        if isinstance(other, np.ndarray):
            other = np.expand_dims(other, -1)
        return self.multiply(other)


class Model:

    def __init__(self, max_vars: int = 5):
        self.max_vars: int = max_vars + 1  # One extra for the constant.
        self.next_var_idx: int = 0  # What number variable model is up to.
        self.dtype = np.float64
        self.int_vars = np.zeros(max_vars, dtype=np.bool)  # True where var is int

        # Constraints are represented as linear expressions which must be <= 0
        self.cons: List[Expr] = []

        self.k = self.var()  # A linear expression equal to 1.0

    def var(self, n=1, integer: bool = False) -> Expr:
        """ Return linear expressions representing new variables. """
        start_idx = self.next_var_idx
        self.next_var_idx += n
        # exprs = np.zeros((n, self.max_vars), dtype=self.dtype)
        exprs = Expr((n, self.max_vars), dtype=self.dtype)
        var_idxs = np.arange(n)
        exprs[var_idxs, start_idx + var_idxs] = 1.0
        self.int_vars[var_idxs] = integer
        return exprs

    def add_constr(self, constr: List[Expr]):
        self.cons += constr

    def __iadd__(self, other: List[Expr]):
        self.add_constr(other)
        return self

    def combine_cons(self) -> Expr:
        return sparse.vstack(self.cons)


def main():
    m = Model()
    a = m.var(2)
    b = m.var(2)

    # Check sum with constant and with vector.
    assert np.allclose((a + 2.34).todense()[:, 0], 2.34)
    assert np.allclose((a + np.array([1.13, 3.01])).todense()[:, 0], [[1.13], [3.01]])

    print(a.todense())
    print((a*2).todense())
    print((a*np.array([2, 3])).todense())

    print(((a + 1) * np.array([2, 3])).todense())

    d = a + np.array([1, 3]) + 3
    m += a <= b
    m += a >= b
    m += a == b
    m += a == d
    print(m.combine_cons().todense())


if __name__ == '__main__':
    main()
