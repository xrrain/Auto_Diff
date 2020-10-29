import numpy as np

eps = np.spacing(1.0)  # np.finfo(np.float64).eps


def five_point_diff(f, h=(eps / 4) ** (1 / 5)):
    # use numpy to compute
    def g(x):
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        nx = len(x)
        cors = [-25, 48, -36, 16, -3]
        res = []
        for i in range(nx):
            delta_array = np.zeros_like(x)
            delta_array[i] = h
            fs = [[cors[i] * y for y in f(x + i * delta_array)] for i in range(5)]
            ys = list(zip(*fs))
            col = [sum(y) / (12 * h) for y in ys]
            res.append(col)

        return np.array(res).T

    return g
