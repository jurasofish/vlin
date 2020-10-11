import vlin
import numpy as np
import pytest


@pytest.mark.parametrize("expr", vlin.expressions)
def test_addition_of_var_and_constant(expr):

    m = vlin.Model(expr=expr)
    a = m.var(2)

    v = a + 2.34
    assert v.shape == a.shape
    assert np.allclose(v.rawdense()[:, 0], 2.34)
    assert isinstance(v, expr)

    v = a + np.array([2.34])
    assert v.shape == a.shape
    assert np.allclose(v.rawdense()[:, 0], 2.34)
    assert isinstance(v, expr)

    v = a + np.array([2.34, 1.23])
    assert v.shape == a.shape
    assert np.allclose(v.rawdense()[:, 0], [2.34, 1.23])
    assert isinstance(v, expr)

    with pytest.raises(ValueError) as execinfo:
        a + np.array([1, 2, 3])
    assert "operands could not be broadcast together" in str(execinfo)

    with pytest.raises(NotImplementedError):
        a += 1


@pytest.mark.parametrize("expr", vlin.expressions)
def test_slicing_type(expr):
    m = vlin.Model(expr=expr)
    a = m.var(4)
    assert isinstance(a[1:3], expr)
    assert isinstance(a[1], expr)


@pytest.mark.parametrize("expr", vlin.expressions)
def test_slicing_shape(expr):
    m = vlin.Model(expr=expr)
    a = m.var(4)
    assert a[1:3].shape == (2, m.max_vars)
    assert a[1].shape == (1, m.max_vars)


@pytest.mark.parametrize("expr", vlin.expressions)
def test_sum(expr):
    m = vlin.Model(expr=expr, max_vars=4)
    a = m.var(1)
    b = m.var(3)

    assert np.allclose(a.sum().rawdense(), [[0, 1, 0, 0, 0]])

    # Broadcasting~~
    # This is like [0, 1, 2] + 1 => [1, 2, 3]
    # and then summed: [1, 2, 3].sum() = 6 == [0, 1, 2].sum() * 1*len([0, 1, 2])
    assert np.allclose((a + b).sum().rawdense(), [[0, 3, 1, 1, 1]])

    assert a.sum().shape == (1, m.max_vars)
    assert b.sum().shape == (1, m.max_vars)


@pytest.mark.parametrize("expr", vlin.expressions)
def test_addition_of_var_and_var(expr):

    m = vlin.Model(expr=expr, max_vars=8)
    a = m.var(2)
    b = m.var(2)
    c = m.var(3)
    d = m.var(1)

    assert np.allclose((a + b).rawdense(), a.rawdense() + b.rawdense())

    # Force broadcasting.
    assert np.allclose((a + d).rawdense(), a.rawdense() + d.rawdense())
    assert np.allclose((d + a).rawdense(), a.rawdense() + d.rawdense())

    with pytest.raises(ValueError) as execinfo:
        a + c
    assert "operands could not be broadcast together" in str(
        execinfo
    ) or "inconsistent shapes" in str(execinfo)


@pytest.mark.parametrize("expr", vlin.expressions)
def test_multiplication_of_var_and_constant(expr):

    m = vlin.Model(expr=expr)
    a = m.var(2)

    v = a * 2.34
    assert v.shape == a.shape
    assert np.allclose(v.rawdense().sum(axis=1), 2.34)
    assert isinstance(v, expr)

    v = a * np.array([2.34])
    assert v.shape == a.shape
    assert np.allclose(v.rawdense().sum(axis=1), 2.34)
    assert isinstance(v, expr)

    v = a * np.array([2.34, 1.23])
    assert v.shape == a.shape
    assert np.allclose(v.rawdense().sum(axis=1), [2.34, 1.23])
    assert isinstance(v, expr)

    with pytest.raises(ValueError) as execinfo:
        a * np.array([1, 2, 3])
    assert "operands could not be broadcast together" in str(
        execinfo
    ) or "inconsistent shapes" in str(execinfo)

    with pytest.raises(NotImplementedError):
        a *= 1
