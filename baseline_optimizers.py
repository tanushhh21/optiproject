import numpy as np

# --------------------------------------------------------
# SGD with MOMENTUM
# --------------------------------------------------------
class SGDM:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def step(self, x, grad_fn):
        g = grad_fn(x)
        if self.v is None:
            self.v = np.zeros_like(g)
        self.v = self.beta * self.v + (1 - self.beta) * g
        x_new = x - self.lr * self.v
        return x_new, np.linalg.norm(g)


# --------------------------------------------------------
# ADAM
# --------------------------------------------------------
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.t = 0

    def step(self, x, grad_fn):
        g = grad_fn(x)
        if self.m is None:
            self.m = np.zeros_like(g)
            self.v = np.zeros_like(g)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        x_new = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return x_new, np.linalg.norm(g)


# --------------------------------------------------------
# RMSPROP
# --------------------------------------------------------
class RMSprop:
    def __init__(self, lr=0.001, beta=0.999, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None

    def step(self, x, grad_fn):
        g = grad_fn(x)
        if self.v is None:
            self.v = np.zeros_like(g)
        self.v = self.beta * self.v + (1 - self.beta) * (g ** 2)
        x_new = x - self.lr * g / (np.sqrt(self.v) + self.eps)
        return x_new, np.linalg.norm(g)
