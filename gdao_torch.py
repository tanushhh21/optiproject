# gdao_torch.py
"""
PyTorch version of GDAO optimizer
"""

import torch

class GDAOTorch(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 gamma=0.5, eps=1e-8, theta_min=0.1):

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        gamma=gamma, eps=eps, theta_min=theta_min)

        super(GDAOTorch, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr     = group['lr']
            beta1  = group['beta1']
            beta2  = group['beta2']
            gamma  = group['gamma']
            eps    = group['eps']
            theta_min = group['theta_min']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                # state init
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['g_prev'] = torch.zeros_like(p.data)

                state['t'] += 1
                t = state['t']

                # Compute gradient alignment
                g_prev = state['g_prev']
                g_norm = torch.norm(g)
                g_prev_norm = torch.norm(g_prev)

                if t == 1 or g_norm < eps or g_prev_norm < eps:
                    alignment = 1.0
                else:
                    cos_sim = torch.dot(g.flatten(), g_prev.flatten()) / (g_norm * g_prev_norm)
                    alignment = torch.clamp(cos_sim, theta_min, 1.0)

                # Adam-style moments
                m = state['m'] = beta1 * state['m'] + (1 - beta1) * g
                v = state['v'] = beta2 * state['v'] + (1 - beta2) * (g * g)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Alignment-scaled learning rate
                lr_t = lr * (gamma * alignment + (1 - gamma))

                # Update
                p.data -= lr_t * m_hat / (torch.sqrt(v_hat) + eps)

                # Store gradient
                state['g_prev'] = g.clone()
