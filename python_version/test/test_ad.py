import time

from ADDiff.base import *


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
    g = JacobianForward(f, return_array=True)
    since = time.time()
    res_ad = g(x)
    print("forward time cost:{:.2f} s".format(time.time() - since))
    print(res_ad.shape)

    # h = JacobianBackward(f, return_array=True)
    # since = time.time()
    # res_bk = h(x)
    # print("backward time cost:{:.2f} s".format(time.time() - since))
    # print(res_bk.shape)

    print(res_ad)
    # print(res_bk)
