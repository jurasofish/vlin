import numpy as np
from vlin import Model


def knapsack():
    # https://docs.python-mip.com/en/latest/examples.html
    m = Model(max_vars=20)
    p = np.array([10, 13, 18, 31, 7, 15])
    w = np.array([11, 15, 20, 35, 10, 33])
    c = 47

    x = m.var(len(w), integer=True)
    m += x >= 0
    m += x <= 1
    m.objective = -1 * (x * p).sum()

    m += (x * w).sum() <= c

    res, sol_x = m.solve_cylp()
    print(x.raw() @ sol_x)
    assert np.isclose(-41, res['value'])


def main():
    m = Model()
    a = m.var(2)
    b = m.var(2)

    # Check sum with constant and with vector.
    assert np.allclose(np.array(a + 2.34)[:, 0], 2.34)
    assert np.allclose(
        np.array(a + np.array([2.34, 1.23]))[:, 0], np.array([2.34, 1.23])
    )
    assert np.allclose(np.array(a + np.array([1.13, 3.01]))[:, 0], [1.13, 3.01])

    knapsack()

    print(a * 2)
    print(a * np.array([2, 3]))

    print((a + 1) * np.array([2, 3]))

    d = a + np.array([1, 3]) + 3
    m += a <= b
    m += a >= b
    m += a == b
    m += a == d
    print(m.combine_cons())

    m = Model(max_vars=4)
    a = m.var(3, integer=np.array([False, True, False]))
    m += a <= np.array([1, 2, 3])
    m += a >= 0
    # m += a[1] + a[2] <= 2
    # m += a[0] + a[1] <= 1
    m.objective = -1 * a.sum()  # Maximise sum of them.
    res, x = m.solve_cylp()
    print(x)
    print(a.raw() @ x)


if __name__ == "__main__":
    main()
