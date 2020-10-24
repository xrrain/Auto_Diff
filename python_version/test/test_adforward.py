import time

from ForwardDiff.base import *


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
    x = [1] * 2020
    since = time.time()
    g = Jacobian(f, return_array=True)
    res_ad = g(x)
    print("time cost:{:.2f} s".format(time.time() - since))
    print(res_ad.shape)
    print(res_ad)
