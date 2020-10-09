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
