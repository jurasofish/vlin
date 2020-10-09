import numpy as np
import vlin


def main():

    m = vlin.Model(max_vars=4)
    a = m.var(3)
    m += a <= np.array([1.1, 2.3, 3.3])
    m += a >= 0
    m.objective = -1 * a.sum()  # Maximise sum of them.
    res, x = m.solve_cylp()
    assert np.allclose(a.raw() @ x, np.array([1.1, 2.3, 3.3]))

    m = vlin.Model()
    a = m.var(2)
    b = m.var(2)

    print(a * 2)
    print(a * np.array([2, 3]))

    print((a + 1) * np.array([2, 3]))

    d = a + np.array([1, 3]) + 3
    m += a <= b
    m += a >= b
    m += a == b
    m += a == d
    print(m.combine_cons())


if __name__ == "__main__":
    main()
