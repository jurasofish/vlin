import vlin
import numpy as np
import pytest


def test_addition_of_var_and_constant():

    m = vlin.Model()
    a = m.var(2)

    v = (a + 2.34)
    assert v.shape == a.raw().shape
    assert np.allclose(v.raw()[:, 0], 2.34)
    assert isinstance(v, vlin.Expr)

    v = (a + np.array([2.34]))
    assert v.shape == a.raw().shape
    assert np.allclose(v.raw()[:, 0], 2.34)
    assert isinstance(v, vlin.Expr)

    v = (a + np.array([2.34, 1.23]))
    assert v.shape == a.raw().shape
    assert np.allclose(v.raw()[:, 0], [2.34, 1.23])
    assert isinstance(v, vlin.Expr)

    with pytest.raises(ValueError) as execinfo:
        a + np.array([1, 2, 3])
    assert 'operands could not be broadcast together' in str(execinfo)

    with pytest.raises(NotImplementedError):
        a += 1


def test_slicing_type():
    m = vlin.Model()
    a = m.var(4)
    assert isinstance(a[1:3], vlin.Expr)
    assert isinstance(a[1], vlin.Expr)


def test_slicing_shape():
    m = vlin.Model()
    a = m.var(4)
    assert a[1:3].shape == (2, m.max_vars)
    assert a[1].shape == (1, m.max_vars)  # TODO: FIX. should return a 2D array.


def test_addition_of_var_and_var():

    m = vlin.Model(max_vars=7)
    a = m.var(2)
    b = m.var(2)
    c = m.var(3)

    assert np.allclose((a + b).raw(), a.raw() + b.raw())

    with pytest.raises(ValueError) as execinfo:
        a + c
    assert 'operands could not be broadcast together' in str(execinfo)


def test_multiplication_of_var_and_constant():

    m = vlin.Model()
    a = m.var(2)

    v = (a * 2.34)
    assert v.shape == a.raw().shape
    assert np.allclose(v.raw().sum(axis=1), 2.34)
    assert isinstance(v, vlin.Expr)

    v = (a * np.array([2.34]))
    assert v.shape == a.raw().shape
    assert np.allclose(v.raw().sum(axis=1), 2.34)
    assert isinstance(v, vlin.Expr)

    v = (a * np.array([2.34, 1.23]))
    assert v.shape == a.raw().shape
    assert np.allclose(v.raw().sum(axis=1), [2.34, 1.23])
    assert isinstance(v, vlin.Expr)

    with pytest.raises(ValueError) as execinfo:
        a * np.array([1, 2, 3])
    assert 'operands could not be broadcast together' in str(execinfo)

    with pytest.raises(NotImplementedError):
        a *= 1
