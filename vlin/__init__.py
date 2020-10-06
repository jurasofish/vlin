import numpy as np
import scipy.sparse as sparse
from typing import List, Union
from numbers import Real


class Expr(sparse.csr_matrix):
    """ A vector of linear expressions: Each row is a vector of var coefficients. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def raw(self) -> sparse.csr_matrix:
        """ Convert expression to raw sparse matrix. """
        return sparse.csr_matrix(self)

    def promote_constant(self, k: Real) -> 'Expr':
        """ Convert a Real number into a constant-valued linear expression. """
        expr = sparse.csr_matrix(self.shape, dtype=self.dtype)
        expr[:, 0] = 1
        expr *= k
        return self.__class__(expr)

    def __le__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x <= y  =>  x-y <= 0 """
        return [self - other]

    def __ge__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x >= y  =>  x-y >= 0  =>  y-x <= 0 """
        return [other - self]

    def __eq__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x == y  =>  x-y >= 0 AND x-y <= 0 """
        return (other <= self) + (self <= other)  # List of two linear expressions.

    def __add__(self, other: Union['Expr', Real]) -> 'Expr':
        if isinstance(other, Real):
            other = self.promote_constant(other)
        return super().__add__(other)

    def __mul__(self, other: Real) -> 'Expr':
        if not isinstance(other, Real):
            raise NotImplementedError
        return super().__mul__(other)

    def __sub__(self, other: Union['Expr', Real]) -> 'Expr':
        return self.__add__(-1.0 * other)

    def __truediv__(self, other: Real) -> 'Expr':
        return self.__mul__(1.0/other)

    def __radd__(self, other: Union['Expr', Real]) -> 'Expr':
        return self.__add__(other)

    def __rsub__(self, other: Union['Expr', Real]) -> 'Expr':
        return (-self).__add__(other)

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
        return sparse.vstack(self.cons, format='csr')


def main():
    m = Model()
    a = m.var(2)
    b = m.var(2)
    c = a + 2
    m += a <= b
    m += a >= b
    m += a == b
    m += c == a
    print(m.combine_cons().todense())


if __name__ == '__main__':
    main()