import argparse
import time

from ADDiff.base import *

parser = argparse.ArgumentParser(description='Auto Diff')
parser.add_argument('--verbose', action='store_true', default=False, help='print results')
parser.add_argument('--scale', type=int, default=2020, help='input dimension')
args = parser.parse_args()


def f(x):
    a = 1
    b = 1
    for i in range(len(x)):
        y = 0.3 * sin(a) + 0.4 * b
        z = 0.1 * a + 0.3 * cos(b) + x[i]
        a = y
        b = z
    return [a, b]



def run(n=2020, verbose=True):
    x = [1.0] * n
    g = JacobianForward(f, return_array=True)
    since = time.time()
    res_ad = g(x)
    if verbose:
        print("forward time cost:{:.2f} s".format(time.time() - since))
        print(res_ad.shape)
        print(res_ad)
    g = JacobianBackward(f, return_array=True)
    since = time.time()
    res_bk = g(x)
    if verbose:
        print("forward time cost:{:.2f} s".format(time.time() - since))
        print(res_bk.shape)
        print(res_bk)


if __name__ == '__main__':
    # run(args.scale, args.verbose)
    x = [1.0]*2020
    gf(x)

