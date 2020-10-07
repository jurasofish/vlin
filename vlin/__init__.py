import numpy as np
import scipy.sparse as sparse
from typing import List, Union, Sequence
from numbers import Real


class Expr:
    """ An array of linear expressions: each expression is a vector of var coefficients. """

    def raw(self):
        """ Convert expression to its raw underlying data type. """
        raise NotImplementedError

    @classmethod
    def zeros(self, shape, dtype=np.float64):
        """ Return instance of self all zeros. """
        raise NotImplementedError

    @classmethod
    def vstack(self, tup: Sequence['Expr']):
        """ apply vstack to given expressions. """
        raise NotImplementedError

    def __add__(self, other: Union['Expr', Real, np.ndarray]) -> 'Expr':
        """ Implement me """
        raise NotImplementedError

    def __mul__(self, other: Union[Real, np.ndarray]) -> 'Expr':
        """ Implement me """
        raise NotImplementedError

    def __le__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x <= y  =>  x-y <= 0 """
        raise NotImplementedError

    def __ge__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ Negative of less than or equal. """
        return [-1.0*x for x in self.__le__(other)]

    def __eq__(self, other: Union['Expr', Real]) -> List['Expr']:
        """ x == y  =>  x-y >= 0 AND x-y <= 0 """
        con = (other <= self)[0]
        return [con, -1.0 * con]

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

    def raw(self):
        """ Convert expression to numpy array """
        return np.array(self)

    @classmethod
    def zeros(cls, shape, dtype=np.float64):
        return cls(np.zeros(shape, dtype=dtype))

    @classmethod
    def vstack(self, tup: Sequence['ExprNumpy']):
        return np.vstack(tup)

    def __add__(self, other: Union['Expr', Real, np.ndarray]) -> 'Expr':
        """ Implement me """
        expr = np.zeros(self.shape, dtype=self.dtype)
        expr[..., 0] = 1  # Last axis
        expr *= np.expand_dims(other, -1)
        return self.__class__(np.add(self, expr))

    def __mul__(self, other: Union[Real, np.ndarray]) -> 'Expr':
        """ Implement me """
        return np.multiply(self, np.expand_dims(other, -1))


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
        exprs = ExprNumpy.zeros((n, self.max_vars), dtype=self.dtype)
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
        return self.cons[0].vstack(self.cons)


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
    print(m.combine_cons().todense())

    m = Model(max_vars=10)
    x = m.var(3)
    y = m.var(2)

    A = np.array([[1., 2., 0], [1., 0, 1.]])
    B = np.array([[1., 0, 0], [0, 0, 1.]])
    D = np.array([[1., 2.], [0, 1]])
    a = np.array([5, 2.5])
    b = np.array([4.2, 3])
    x_u = np.array([2., 3.5])

    m += A * x <= a


if __name__ == '__main__':
    main()
