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


count = [0]
if __name__ == '__main__':
    x = [1] * 4
    # since = time.time()
    # g = JacobianForward(f, return_array=True)
    # res_ad = g(x)
    # print("time cost:{:.2f} s".format(time.time() - since))
    # print(res_ad.shape)

    since = time.time()
    # h = JacobianBackward(f, return_array=True)
    # res_bk = h(x)
    h = ADbackward(f, return_array=True)
    d = [0, 1]
    # print(res_ad)
    res_bk = h(x, d, count)
    print(res_bk)
    print(count)
    print("time cost:{:.2f} s".format(time.time() - since))
    # print(print((res_ad == res_bk).all()))
