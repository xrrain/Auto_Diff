import time

from test import test_adforward, test_adbackward, test_finitediff


def test(f, iters, disp=True, return_ms=True, **kwargs):
    res = 0.0
    for i in range(iters):
        since = time.time()
        f(**kwargs)
        end = time.time()
        duration = (end - since) * 1000 if return_ms else (end - since)
        res += duration
        if disp:
            print("the {} runing cost {}{}".format(i, duration, "ms" if return_ms else "s"))
    return res / iters


def run():
    test_function = {"forward": test_adforward.run, "backward": test_adbackward.run, "finitediff": test_finitediff.run}
    runing_parameter = {"verbose": False, "n": 2020}
    iters = 10
    return_ms = True
    for name, function in test_function.items():
        print("begin to test on {}".format(name))
        res = test(function, iters, disp=True, return_ms=return_ms, **runing_parameter)
        print("finish to test on {}, average cost: {}{}".format(name, res, "ms" if return_ms else "s"))


if __name__ == '__main__':
    run()
