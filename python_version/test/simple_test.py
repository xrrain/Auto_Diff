from ForwardDiff.base import *


def f(x):
    return [sin(cos(x[0] - x[1]))]


print(ADforward(f)([2, 1], [1, 0]))
print(ADforward(f)([2, 1], [0, 1]))
print(-cos(cos(1)) * sin(1))
print(cos(cos(1)) * sin(1))

from BackwardDiff.base import *


def f(x):
    return [sin(cos(x[0] - x[1]))]


print(ADbackward(f)([2, 1], [1]))
print(-cos(cos(1)) * sin(1))
print(cos(cos(1)) * sin(1))

from Numerical_diff.base import *


def f(x):
    return [sin(cos(x[0] - x[1]))]


print(five_point_diff(f)([2, 1]))
print(-cos(cos(1)) * sin(1))
print(cos(cos(1)) * sin(1))
