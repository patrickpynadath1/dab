import torch
import torch.nn as nn
import torch.distributions as dists


class LangevinSampler(nn.Module):
    def __init__(self, step_size, mh=True, beta1=0.9, beta2=0.999):
        super().__init__()
        self.step_size = step_size

        self.mh = mh
        self.a_s = []
        self.hops = []
        self.beta1 = 0.9
        self.beta2 = 0.999

    def compute_delta(self, x, model):
        if self.grad_mu is None:
            self.grad_mu = torch.zeros(2, x[0], x[1], x[2])
        if self.grad_sigma is None:
            self.grad_sigma = torch.zeros(2, x[0], x[1], x[2])
        x = x.requires_grad_()
        model_out = model(x)
        loss, output_ids, gpt_logit, senti_losses = model_out
        gx = torch.autograd.grad(loss, x, allow_unused=True)[0]
        #
        self.grad_mu = self.beta1 * self.grad_mu + (1 - self.beta1) * gx
        self.grad_sigma = self.beta2 * self.grad_sigma + (1 - self.beta2) * (gx**2)

        wx = gx * (2.0 * x - 1)
        return wx.detach(), loss, output_ids.cpu(), senti_losses.cpu()

    def step(self, x, model):
        x_cur = x
        EPS = 1e-10
        forward_delta, loss, output_ids, senti_losses = self.compute_delta(x_cur, model)
        term2 = 1.0 / (2 * self.step_size)  # for binary {0,1}, the L2 norm is always 1
        flip_prob = torch.exp(forward_delta - term2) / (
            torch.exp(forward_delta - term2) + 1
        )
        rr = torch.rand_like(x_cur)
        ind = (rr < flip_prob) * 1
        x_delta = (1.0 - x_cur) * ind + x_cur * (1.0 - ind)
        if self.mh:
            probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
            lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)
            reverse_delta, _, _, _ = self.compute_delta(x_delta, model)
            flip_prob = torch.exp(reverse_delta - term2) / (
                torch.exp(reverse_delta - term2) + 1
            )
            probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
            lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)
            delta_energy, _, _, _ = model(x_delta)
            cur_energy, _, _, _ = model(x_cur)
            m_term = delta_energy.squeeze() - cur_energy.squeeze()
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            self.a_s.append(a.mean().item())
            x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
        else:
            # print("hops")
            # print((x_delta - x_cur).sum(dim=-1).mean())
            x_cur = x_delta
        return x_cur, loss, output_ids, senti_losses
