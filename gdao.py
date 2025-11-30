# gdao.py
import numpy as np

class GDAO:
    """
    Gradient Direction Alignment Optimizer (NumPy version)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 gamma=0.5, epsilon=1e-8, theta_min=0.1):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.eps = epsilon
        self.theta_min = theta_min

        self.m = None
        self.v = None
        self.g_prev = None
        self.t = 0

    def _init_state(self, x):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
            self.g_prev = np.zeros_like(x)

    def _alignment(self, g):
        if self.t == 0:
            return 1.0

        norm_g = np.linalg.norm(g)
        norm_prev = np.linalg.norm(self.g_prev)

        if norm_g < self.eps or norm_prev < self.eps:
            return 1.0

        cos = np.dot(g, self.g_prev) / (norm_g * norm_prev)
        return np.clip(cos, self.theta_min, 1.0)

    def step(self, x, grad_fn):
        self._init_state(x)

        g = grad_fn(x)
        self.t += 1

        align = self._alignment(g)

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        lr_t = self.lr * (self.gamma * align + (1 - self.gamma))

        x_new = x - lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

        self.g_prev = g.copy()

        return x_new, {
            "alignment": align,
            "effective_lr": lr_t,
            "grad_norm": np.linalg.norm(g)
        }

    def optimize(self, x0, grad_fn, iters):
        x = x0
        history = []
        for _ in range(iters):
            x, metrics = self.step(x, grad_fn)
            history.append(metrics)
        return x, history
