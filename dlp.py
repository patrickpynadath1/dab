import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np

class LangevinSampler(nn.Module):
    def __init__(self, step_size, mh=True):
        super().__init__()
        self.step_size = step_size  

        self.mh = mh
        self.a_s = []
        self.hops = []

    def compute_delta(self, x, model): 
        x = x.requires_grad_()
        model_out = model(x)
        # for output of sentiment 
        # will be loss, output ids, gpt logits, senti losses 
        # for output of toxicity, will be same 
        # for output of keyword generations, will be just 
        # loss and output ids 
        loss, output_ids, *otheroutputs = model_out
        gx = torch.autograd.grad(loss, x, allow_unused=True)[0]
        wx = gx * (2. * x - 1)
        return wx.detach(), loss, output_ids.cpu(), otheroutputs

    def step(self, x, model):
        x_cur = x
        EPS = 1e-10
        forward_delta, cur_energy, output_ids, otheroutputs = self.compute_delta(x_cur, model)
        term2 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1        
        flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)
        rr = torch.rand_like(x_cur)
        ind = (rr<flip_prob)*1
        x_delta = (1. - x_cur)*ind + x_cur * (1. - ind)
        if self.mh:
            probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
            lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
            reverse_delta, delta_energy, _, _ = self.compute_delta(x_delta, model)
            flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
            probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
            lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)
            m_term = (delta_energy.squeeze() - cur_energy.squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            self.a_s.append(a.mean().item())
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
        else:
            # print("hops")
            # print((x_delta - x_cur).sum(dim=-1).mean())
            x_cur = x_delta
        return x_cur, cur_energy, output_ids, otheroutputs