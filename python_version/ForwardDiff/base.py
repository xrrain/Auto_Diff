import numpy as np

class Ops:
    """Base Class for all operations, support +, -, *, /, sin, cos
        mainly refer to https://sidsite.com/posts/autodiff/ """

    def __init__(self, value: float = 0.0, grad: float = 0.0, op: str = None):
        self.value = value
        self.grad = grad
        self.op = op

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
        return "value:{} grad:{} operation:{}".format(self.value, self.grad, self.op)

    @staticmethod
    def _add(a, b):
        if isinstance(b, Ops):
            return Ops(a.value + b.value, a.grad + b.grad, op="add")
        else:
            # add constant
            return Ops(a.value + b, a.grad, op="add")

    @staticmethod
    def _sub(a, b):
        if isinstance(b, Ops):
            return Ops(a.value - b.value, a.grad - b.grad, op="sub")
        else:
            # sub constant
            return Ops(a.value - b, a.grad, op="sub")

    @staticmethod
    def _neg(var):
        return Ops(-var.value, -var.grad, op="neg")

    @staticmethod
    def _mul(a, b):
        if isinstance(b, Ops):
            return Ops(a.value * b.value, a.value * b.grad + b.value * a.grad, op="mul")
        else:
            # mul constant
            return Ops(a.value * b, a.grad * b, op="mul")

    @staticmethod
    def _inv(var):
        return Ops(1 / var.value, -var.grad / (var.value ** 2), op="inv")


class Var(Ops):
    def __init__(self, value, grad=1):
        super(Var, self).__init__(value, grad, op="var")


def sin(var):
    if isinstance(var, Ops):
        return Ops(np.sin(var.value), np.cos(var.value) * var.grad, op="sin")
    else:
        return np.sin(var)


def cos(var):
    if isinstance(var, Ops):
        return Ops(np.cos(var.value), -np.sin(var.value) * var.grad, op="cos")
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


def Jacobian(f, return_array=False):
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
