import vlin
import numpy as np
import pytest


@pytest.mark.parametrize("expr", vlin.expressions)
def test_trivial_fully_linear(expr):
    m = vlin.Model(expr=expr, max_vars=4)
    a = m.var(3)
    m += a <= np.array([1.1, 2.3, 3.3])
    m += a >= 0
    m.objective = -1 * a.sum()  # Maximise sum of them.
    res, x = m.solve_cylp()
    assert np.allclose(a.raw() @ x, np.array([1.1, 2.3, 3.3]))


@pytest.mark.parametrize("expr", vlin.expressions)
def test_trivial_milp(expr):
    m = vlin.Model(expr=expr, max_vars=4)
    a = m.var(3, integer=np.array([False, True, False]))
    m += a <= np.array([1.1, 2.3, 3.3])
    m += a >= 0
    m.objective = -1 * a.sum()  # Maximise sum of them.
    res, x = m.solve_cylp()
    assert np.allclose(a.raw() @ x, np.array([1.1, 2.0, 3.3]))
