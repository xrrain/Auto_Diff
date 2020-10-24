import time

from Numerical_diff.base import five_point_diff
import numpy as np
from numpy import sin as sin
from numpy import cos as cos


def f(x):
    a = 1
    b = 1
    for i in range(len(x)):
        y = 0.3 * sin(a) + 0.4 * b
        z = 0.1 * a + 0.3 * cos(b) + x[i]
        a = y
        b = z
    return [a, b]


if __name__ == '__main__':
    x = [1.0] * 2020
    since = time.time()
    eps = np.spacing(1.0)  # np.finfo(np.float64).eps
    h = (eps / 4) ** (1 / 5)
    g = five_point_diff(f, h)
    res_ad = g(x)
    print("time cost:{:.2f} s".format(time.time() - since))
    print(res_ad.shape)
    print(res_ad)
