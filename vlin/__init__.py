import numpy as np


class Model:

    def __init__(self, max_vars: int = 5, max_constrs: int = 5):
        self.max_vars: int = max_vars
        self.next_var_idx: int = 0  # What number variable model is up to.
        self.dtype = np.float64

        # Constraints are represented as linear expressions which must be <= 0
        self.constrs = np.zeros((max_constrs, max_vars), dtype=self.dtype)

        # A linear expression equal to 1.0
        k = self.var().squeeze()

    def var(self, n=1, integer: bool = False) -> np.ndarray:
        """ Return linear expressions representing new variables. """
        if integer:
            raise NotImplementedError
        idx = self.next_var_idx
        self.next_var_idx += n
        exprs = np.zeros((n, self.max_vars), dtype=self.dtype)
        exprs[np.arange(n), idx + np.arange(n)] = 1
        return exprs


def main():
    m = Model()
    a = m.var(2)


if __name__ == '__main__':
    main()
