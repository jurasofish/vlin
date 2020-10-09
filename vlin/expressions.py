import numpy as np
import scipy.sparse as sparse
from typing import List, Union, Sequence, Optional, Type, Tuple
from numbers import Real
from abc import ABC


__all__ = [
    "Expr",
    "ExprNumpy",
]


class Expr(ABC):
    """ An array of linear expressions: each expression is a vector of var coefficients. """

    shape: Tuple

    def raw(self):
        """ Convert expression to its raw underlying data type. """
        raise NotImplementedError

    def sum(self) -> "Expr":
        """ Return sum of vector as a vector with one element. """
        raise NotImplementedError

    @classmethod
    def zeros(self, shape: int, max_vars: int, dtype: np.dtype = np.float64) -> "Expr":
        """ Return vector instance of self all zeros. """
        raise NotImplementedError

    @classmethod
    def vstack(self, tup: Sequence["Expr"]) -> "Expr":
        """ apply vstack to given expressions. """
        raise NotImplementedError

    def __add__(self, other: Union["Expr", Real, np.ndarray]) -> "Expr":
        """ Implement me """
        raise NotImplementedError

    def __mul__(self, other: Union[Real, np.ndarray]) -> "Expr":
        """ Implement me """
        raise NotImplementedError

    def __le__(self, other: Union["Expr", Real]) -> "Expr":
        """ x <= y  =>  x-y <= 0 """
        raise NotImplementedError

    def __ge__(self, other: Union["Expr", Real]) -> "Expr":
        """ Negative of less than or equal. """
        return -1.0 * self.__le__(other)

    def __eq__(self, other: Union["Expr", Real]) -> "Expr":
        """ x == y  =>  x-y >= 0 AND x-y <= 0 """
        con = other <= self
        return self.vstack([con, -1.0 * con])

    def __sub__(self, other: Union["Expr", Real]) -> "Expr":
        return self.__add__(-1.0 * other)

    def __truediv__(self, other: Real) -> "Expr":
        return self.__mul__(1.0 / other)

    def __radd__(self, other: Union["Expr", Real]) -> "Expr":
        return self.__add__(other)

    def __rsub__(self, other: Union["Expr", Real]) -> "Expr":
        return (-1 * self).__add__(other)

    def __rmul__(self, other: Real) -> "Expr":
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

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)


class ExprNumpy(Expr, np.ndarray):
    def __new__(cls, input_array, info=None):
        # See https://numpy.org/devdocs/user/basics.subclassing.html
        obj = np.asarray(input_array).view(cls)
        obj.base_cast = np.array
        return obj

    def __array_finalize__(self, obj):
        # See https://numpy.org/devdocs/user/basics.subclassing.html
        if obj is None:
            return
        self.base_cast = getattr(obj, "base_cast", None)

    def raw(self) -> np.ndarray:
        """ Convert expression to numpy array """
        return np.asarray(self).view(np.ndarray)

    def sum(self) -> "ExprNumpy":
        return self.__class__(self.raw().sum(axis=-2))

    @classmethod
    def zeros(
        cls, shape: int, max_vars: int, dtype: np.dtype = np.float64
    ) -> "ExprNumpy":
        return cls(np.zeros((shape, max_vars), dtype=dtype))

    @classmethod
    def vstack(cls, tup: Sequence["ExprNumpy"]) -> "ExprNumpy":
        return cls(np.vstack(tup))

    def __add__(self, other: Union["ExprNumpy", Real, np.ndarray]) -> "ExprNumpy":
        if not isinstance(other, Expr):
            expr = np.zeros(self.shape, dtype=self.dtype)
            expr[..., 0] = 1  # Last axis
            other = np.array(other)
            expr *= np.expand_dims(other, tuple(range(other.ndim, self.ndim)))
        else:
            expr = other.raw()
        return self.__class__(np.add(self, expr))

    def __mul__(self, other: Union[Real, np.ndarray]) -> "ExprNumpy":
        try:
            return np.multiply(self, other)
        except ValueError:
            # Exception rarely hit, so is faster than instance/dimension check.
            return np.multiply(self, np.expand_dims(other, -1))

    def __le__(self, other: Union["ExprNumpy", Real, float]) -> "ExprNumpy":
        """ x <= y  =>  x-y <= 0 """
        if not isinstance(other, Expr):
            expr = np.zeros(self.shape, dtype=self.dtype)
            expr[..., 0] = 1  # Last axis
            other = np.array(other)
            expr *= np.expand_dims(other, tuple(range(other.ndim, self.ndim)))
        else:
            expr = other.raw()
        return np.subtract(self, expr)
