import numpy as np


class Model:

    def __init__(self, max_vars: int = 5, max_constrs: int = 5):
        self.max_vars: int = max_vars + 1  # One extra for the constant.
        self.next_var_idx: int = 0  # What number variable model is up to.
        self.next_con_idx: int = 0  # What number constraint model is up to.
        self.dtype = np.float64
        self.int_vars = np.zeros(max_vars, dtype=np.bool)  # True where var is int

        # Constraints are represented as linear expressions which must be <= 0
        self.cons = np.zeros((max_constrs, max_vars), dtype=self.dtype)

        self.k = self.var()  # A linear expression equal to 1.0

    def var(self, n=1, integer: bool = False) -> np.ndarray:
        """ Return linear expressions representing new variables. """
        start_idx = self.next_var_idx
        self.next_var_idx += n
        exprs = np.zeros((n, self.max_vars), dtype=self.dtype)
        var_idxs = np.arange(n)
        exprs[var_idxs, start_idx + var_idxs] = 1.0
        self.int_vars[var_idxs] = integer
        return exprs

    def le(self, x: np.ndarray, y: np.ndarray):
        """ x <= y  =>  x-y <= 0 """
        n = x.shape[0]
        start_idx = self.next_con_idx
        self.next_con_idx += n
        self.cons[start_idx + np.arange(n)] = x - y

    def ge(self, x: np.ndarray, y: np.ndarray):
        """ x >= y  =>  x-y >= 0  =>  y-x <= 0 """
        n = x.shape[0]
        start_idx = self.next_con_idx
        self.next_con_idx += n
        self.cons[start_idx + np.arange(n)] = y - x

    def eq(self, x: np.ndarray, y: np.ndarray):
        """ x == y  =>  x-y >= 0 AND x-y <= 0 """
        self.le(x, y)
        self.ge(x, y)


def main():
    m = Model()
    a = m.var(2)
    b = m.var(2)
    m.le(a, b)
    m.ge(a, b)
    m.eq(a, b)


if __name__ == '__main__':
    main()
