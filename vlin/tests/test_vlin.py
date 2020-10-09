import vlin
import numpy as np
import pytest


def test_addition_of_var_and_constant():

    m = vlin.Model()
    a = m.var(2)

    v = (a + 2.34).raw()
    assert v.shape == a.raw().shape
    assert np.allclose(v[:, 0], 2.34)

    v = (a + np.array([2.34])).raw()
    assert v.shape == a.raw().shape
    assert np.allclose(v[:, 0], 2.34)

    v = (a + np.array([2.34, 1.23])).raw()
    assert v.shape == a.raw().shape
    assert np.allclose(v[:, 0], [2.34, 1.23])

    with pytest.raises(ValueError) as execinfo:
        a + np.array([1, 2, 3])
    assert 'operands could not be broadcast together' in str(execinfo)

    with pytest.raises(NotImplementedError):
        a += 1


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

    v = (a * 2.34).raw()
    assert v.shape == a.raw().shape
    assert np.allclose(v.sum(axis=1), 2.34)

    v = (a * np.array([2.34])).raw()
    assert v.shape == a.raw().shape
    assert np.allclose(v.sum(axis=1), 2.34)

    v = (a * np.array([2.34, 1.23])).raw()
    assert v.shape == a.raw().shape
    assert np.allclose(v.sum(axis=1), [2.34, 1.23])

    with pytest.raises(ValueError) as execinfo:
        a * np.array([1, 2, 3])
    assert 'operands could not be broadcast together' in str(execinfo)

    with pytest.raises(NotImplementedError):
        a *= 1
