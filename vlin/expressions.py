import numpy as np
import scipy.sparse as sparse
from typing import List, Union, Sequence, Optional, Type, Tuple
from numbers import Real
from abc import ABC


__all__ = [
    "Expr",
    "ExprNumpy",
    "ExprCSR",
]


class Expr(ABC):
    """ An array of linear expressions: each expression is a vector of var coefficients. """

    shape: Tuple

    def raw(self):
        """ Convert expression to its raw underlying data type. """
        raise NotImplementedError

    def rawdense(self) -> np.ndarray:
        """ Convert expression to np.ndarray """
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

    def __le__(self, other: Union["Expr", Real, float, np.ndarray]) -> "Expr":
        """ x <= y  =>  x-y <= 0 """
        return self - other

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
        return np.asarray(self).view(np.ndarray).copy()

    def rawdense(self) -> np.ndarray:
        """ Convert expression to np.ndarray """
        return self.raw()

    def sum(self) -> "ExprNumpy":
        x = np.asarray(self).view(np.ndarray)
        x = x.sum(axis=0)
        x = np.atleast_2d(x)
        return self.__class__(x)

    @classmethod
    def zeros(
        cls, shape: int, max_vars: int, dtype: np.dtype = np.float64
    ) -> "ExprNumpy":
        return cls(np.zeros((shape, max_vars), dtype=dtype))

    @classmethod
    def vstack(cls, tup: Sequence["ExprNumpy"]) -> "ExprNumpy":
        return cls(np.vstack(tup))

    def __add__(self, other: Union["ExprNumpy", Real, np.ndarray]) -> "ExprNumpy":
        if isinstance(other, Expr):
            return self.__class__(np.add(self, other))

        # Mutate self in-place
        # TODO: use this for __iadd__
        # a = np.asarray(self).view(np.ndarray)
        # a[:, 0] += other

        a = self.raw()
        a[:, 0] += other
        return self.__class__(a)

    def __mul__(self, other: Union[Real, np.ndarray]) -> "ExprNumpy":
        try:
            return np.multiply(self, other)
        except ValueError:
            # Exception rarely hit, so is faster than instance/dimension check.
            return np.multiply(self, np.expand_dims(other, -1))

    def __getitem__(self, key):
        """ Force result to be 2D. """
        return np.atleast_2d(super().__getitem__(key))


class ExprCSR(Expr, sparse.csr_matrix):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def raw(self) -> sparse.csr_matrix:
        """ Convert expression to numpy array """
        return sparse.csr_matrix(self)

    def rawdense(self) -> np.ndarray:
        """ Convert expression to np.ndarray """
        return self.raw().toarray()

    def sum(self) -> "ExprCSR":
        return self.__class__(sparse.csr_matrix.sum(self.raw(), axis=0))

    @classmethod
    def zeros(
        cls, shape: int, max_vars: int, dtype: np.dtype = np.float64
    ) -> "ExprCSR":
        return cls(sparse.csr_matrix((shape, max_vars), dtype=dtype))

    @classmethod
    def vstack(cls, tup: Sequence["ExprCSR"]) -> "ExprCSR":
        return cls(sparse.vstack(tup))

    def __add__(self, other: Union["ExprCSR", Real, np.ndarray]) -> "ExprCSR":
        if isinstance(other, Expr):
            # use vstack to broadcast shapes (I'm sorry)
            if self.shape == other.shape:
                return self.__class__(self.raw() + other.raw())
            elif other.shape[0] == 1:
                a, b = self, other
            elif self.shape[0] == 1:
                a, b = other, self
            else:
                raise ValueError(f'inconsistent shapes: {self.shape}, {other.shape}')
            return self.__class__(a.raw() + b.vstack([b]*a.shape[0]).raw())

        a = self.raw()
        k = np.ones(self.shape[0]) * other + a[:, 0].toarray()[:, 0]
        a[:, 0] = np.expand_dims(k, -1)
        return self.__class__(a)

    def __mul__(self, other: Union[Real, np.ndarray]) -> "ExprNumpy":
        try:
            return self.multiply(other)
        except ValueError:
            # Exception rarely hit, so is faster than instance/dimension check.
            return np.multiply(self, np.expand_dims(other, -1))

    def __repr__(self):
        return f'<{self.__class__.__name__}: {super().__repr__()[1:]}'
