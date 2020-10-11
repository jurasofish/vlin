import vlin
import numpy as np
import pytest


@pytest.mark.parametrize("expr", vlin.expressions)
@pytest.mark.parametrize("solver", ["cylp"])
def test_knapsack(expr, solver):
    # https://docs.python-mip.com/en/latest/examples.html
    m = vlin.Model(expr=expr, max_vars=20)
    p = np.array([10, 13, 18, 31, 7, 15])
    w = np.array([11, 15, 20, 35, 10, 33])
    c = 47

    x = m.var(len(w), integer=True)
    m += x >= 0
    m += x <= 1
    m.objective = -1 * (x * p).sum()

    m += (x * w).sum() <= c

    res, sol_x = m.solve(solver=solver)
    print(x.raw() @ sol_x)
    assert np.isclose(-41, res["value"])
    # Note there are multiple optimal solutions.
