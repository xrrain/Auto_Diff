from collections import defaultdict
import numpy as np


class Ops:
    """Base Class for all operations, support +, -, *, /, sin, cos
        mainly refer to https://sidsite.com/posts/autodiff/ """

    def __init__(self, value: float = None, children: list = None):
        self.value = value
        self.children = children

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

    def __repr__(self):
        return "value:{} operation:{}".format(self.value, [child.__class__.name for child in self.children])

    @staticmethod
    def _add(a, b):
        if isinstance(b, Ops):
            return Ops(a.value + b.value, [(a, 1), (b, 1)])
        else:
            # add constant
            return Ops(a.value + b, [(a, 1)])

    @staticmethod
    def _sub(a, b):
        if isinstance(b, Ops):
            return Ops(a.value - b.value, [(a, 1), (b, -1)])
        else:
            # sub constant
            return Ops(a.value - b, [(a, 1)])

    @staticmethod
    def _neg(var):
        return Ops(-var.value, [(var, -1)])

    @staticmethod
    def _mul(a, b):
        if isinstance(b, Ops):
            return Ops(a.value * b.value, [(a, b.value), (b, a.value)])
        else:
            # mul constant
            return Ops(a.value * b, [(a, b)])

    @staticmethod
    def _inv(var):
        return Ops(1 / var.value, [(var, - 1 / (var.value ** 2))])


class Var(Ops):
    def __init__(self, value, children=None):
        super(Var, self).__init__(value, children)


def sin(var):
    if isinstance(var, Ops):
        return Ops(np.sin(var.value), [(var, np.cos(var.value))])
    else:
        return np.sin(var)


def cos(var):
    if isinstance(var, Ops):
        return Ops(np.cos(var.value), [var, -np.sin(var.value)])
    else:
        return np.cos(var)


def ADbackward(f, return_array=False):
    def h(x, d):
        X = [Var(x[i]) for i in range(len(x))]
        Y = f(X)
        assert len(d) == len(Y)
        for i, y in enumerate(Y):
            gradients = defaultdict(lambda: 0)
            # bfs update
            node_stack = list()
            node_stack.append([y, d[i]])
            while node_stack:
                var, local_grad = node_stack.pop()
                gradients[var] += local_grad
                node_stack += [(child[0], child[1] * local_grad) for child in var.children]
        ans = [dict(gradients)[x] for x in X]
        return np.array(ans) if return_array else ans

    return h


def Jacobian(f, return_array=False):
    def h(x):
        X = [Var(x[i]) for i in range(len(x))]
        Y = f(X)
        n = len(Y)
        res = []
        for i in range(n):
            d = [0] * n
            d[i] = 1
            for i, y in enumerate(Y):
                gradients = defaultdict(lambda: 0)
                # bfs update
                node_stack = list()
                node_stack.append([y, d[i]])
                while node_stack:
                    var, local_grad = node_stack.pop()
                    gradients[var] += local_grad
                    node_stack += [(child[0], child[1] * local_grad) for child in var.children]
            ans = [dict(gradients)[x] for x in X]
            res.append(ans)
        return np.array(res) if return_array else res

    return h
