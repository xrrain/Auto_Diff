from collections import defaultdict

import numpy as np


class Ops:
    """Base Class for all operations, support +, -, *, /, sin, cos
        mainly refer to https://sidsite.com/posts/autodiff/ """

    def __init__(self, value: float = 0.0, grad: float = 0.0, grad_back: float = 0.0, children: list = None,
                 op: str = "None"):
        self.value = value
        self.grad = grad  # store forward grad
        self.op = op

        self.grad_back = grad_back  # store backward grad
        self.children = children  # store children nodes

    def __add__(self, other):
        return self._add(self, other)

    def __radd__(self, other):
        return self._add(self, other)

    def __neg__(self):
        return self._neg(self)

    def __sub__(self, other):
        return self._sub(self, other)

    def __rsub__(self, other):
        return self._add(self._neg(self), other)

    def __mul__(self, other):
        return self._mul(self, other)

    def __rmul__(self, other):
        return self._mul(self, other)

    def __invert__(self):
        return self._inv(self)

    def __truediv__(self, other):
        if isinstance(other, Ops):
            return self._mul(self, self._inv(other))
        else:
            return self._mul(self, 1 / other)

    def __rtruediv__(self, other):
        return self._mul(self._inv(self), other)

    def __str__(self):
        return "value:{} grad:{} grad_back:{} operation:{}".format(self.value, self.grad, self.grad_back, self.op)

    @staticmethod
    def _add(a, b):
        if isinstance(b, Ops):
            return Ops(value=a.value + b.value, grad=a.grad + b.grad, children=[(a, 1), (b, 1)], op="add")
        else:
            # add constant
            return Ops(value=a.value + b, grad=a.grad, children=[(a, 1)], op="add")

    @staticmethod
    def _sub(a, b):
        if isinstance(b, Ops):
            return Ops(value=a.value - b.value, grad=a.grad - b.grad, children=[(a, 1), (b, -1)], op="sub")
        else:
            # sub constant
            return Ops(value=a.value - b, grad=a.grad, children=[(a, 1)], op="sub")

    @staticmethod
    def _neg(var):
        return Ops(value=-var.value, grad=-var.grad, children=[(var, -1)], op="neg")

    @staticmethod
    def _mul(a, b):
        if isinstance(b, Ops):
            return Ops(value=a.value * b.value, grad=a.value * b.grad + b.value * a.grad,
                       children=[(a, b.value), (b, a.value)], op="mul")
        else:
            # mul constant
            return Ops(value=a.value * b, grad=a.grad * b, children=[(a, b)], op="mul")

    @staticmethod
    def _inv(var):
        return Ops(value=1 / var.value, grad=-var.grad / (var.value ** 2), children=[(var, - 1 / (var.value ** 2))],
                   op="inv")


class Var(Ops):
    def __init__(self, value, grad=0.0, grad_back=0.0, children=None):
        super(Var, self).__init__(value=value, grad=grad, grad_back=grad_back, children=children, op="var")


def sin(var):
    if isinstance(var, Ops):
        return Ops(value=np.sin(var.value), grad=np.cos(var.value) * var.grad, children=[(var, np.cos(var.value))],
                   op="sin")
    else:
        return np.sin(var)


def cos(var):
    if isinstance(var, Ops):
        return Ops(value=np.cos(var.value), grad=-np.sin(var.value) * var.grad, children=[(var, -np.sin(var.value))],
                   op="cos")
    else:
        return np.cos(var)


def ADforward(f, return_array=False):
    def g(x, d):
        if not isinstance(x, list):
            x = [x]
        assert len(x) == len(d)
        X = [Var(x[i], d[i]) for i in range(len(x))]
        Y = f(X)
        if not isinstance(Y, list):  # return one value
            Y = [Y]
        if return_array:
            return np.array([y.grad for y in Y]).T
        else:
            return [y.grad for y in Y]

    return g


def JacobianForward(f, return_array=False):
    def g(x):
        if not isinstance(x, list):
            x = [x]
        h = ADforward(f, return_array)
        n = len(x)
        res = []
        for i in range(n):
            d = [0] * n
            d[i] = 1
            col = h(x, d)
            res.append(col)
        if return_array:
            return np.array(res).T
        else:
            return res

    return g


def ADbackward(f, return_array=False):
    def h(x, d, count):
        if not isinstance(x, list):
            x = [x]
        X = [Var(value=x[i]) for i in range(len(x))]
        Y = f(X)
        assert len(d) == len(Y)
        # clear grad
        for i, y in enumerate(Y):
            # bfs update
            node_stack = list()
            y.grad_back = d[i]
            node_stack.append(y)

            while node_stack:
                var = node_stack.pop()
                count[0] += 1
                if var.children is not None:
                    for child, local_grad in var.children:
                        if isinstance(child, Var):
                            child.grad_back += var.grad_back * local_grad
                        else:
                            child.grad_back = var.grad_back * local_grad
                            node_stack.append(child)
        ans = [x.grad_back for x in X]
        return np.array(ans) if return_array else ans

    return h


def JacobianBackward(f, return_array=False):
    def h(x):
        if not isinstance(x, list):
            x = [x]
        y = f(x)
        n = len(y) if isinstance(y, list) else 1
        res = []
        for seed in range(n):
            d = [0] * n
            d[seed] = 1
            g = ADbackward(f, return_array)
            ans = g(x, d)
            res.append(ans)
        return np.array(res) if return_array else res

    return h
