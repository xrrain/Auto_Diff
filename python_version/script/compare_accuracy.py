import time

import numpy as np
from ADDiff.base import *
from Numerical_diff.base import five_point_diff


def f(x):
    a = 1
    b = 1
    for i in range(len(x)):
        y = 0.3 * sin(a) + 0.4 * b
        z = 0.1 * a + 0.3 * cos(b) + x[i]
        a = y
        b = z
    return [a, b]


def gf(x):
    n = len(x)
    a = 1
    b = 1
    da = np.zeros(n, dtype=float)
    db = np.zeros(n, dtype=float)
    db[0] = 1
    for i in range(n):
        db[i] = 1
        temp_a = da.copy()
        da[0:i] = 0.3 * cos(a) * da[0:i] + 0.4 * db[0:i]
        db[0:i] = 0.1 * temp_a[0:i] - 0.3 * sin(b) * db[0:i]
        y = 0.3 * sin(a) + 0.4 * b
        z = 0.1 * a + 0.3 * cos(b) + x[i]
        a = y
        b = z
    return np.vstack((da, db))


def run(n=2020, verbose=True):
    x = [1.0] * n
    since = time.time()
    res = gf(x)
    if verbose:
        print("forward time cost:{:.6f} s".format(time.time() - since))
        print(res.shape)
        print(res)

    g = JacobianForward(f, return_array=True)
    since = time.time()
    res_ad = g(x)
    if verbose:
        print("forward time cost:{:.6f} s".format(time.time() - since))
        print(res_ad.shape)
        print("l2 norm : {}".format(np.linalg.norm(res - res_ad)))

    g = JacobianBackward(f, return_array=True)
    since = time.time()
    res_bk = g(x)
    if verbose:
        print("forward time cost:{:.6f} s".format(time.time() - since))
        print(res_bk.shape)
        print("l2 norm : {}".format(np.linalg.norm(res - res_bk)))
        print(res_bk)

    since = time.time()
    eps = np.spacing(1.0)  # np.finfo(np.float64).eps
    h = (eps / 4) ** (1 / 5)
    g = five_point_diff(f, h)
    res_fd = g(x)
    if verbose:
        print("time cost:{:.6f} s".format(time.time() - since))
        print(res_fd.shape)
        print("l2 norm : {}".format(np.linalg.norm(res - res_fd)))
        print(res_fd)


if __name__ == '__main__':
    run(n=2020, verbose=True)
