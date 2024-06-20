import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np

class LangevinSampler(nn.Module):
    def __init__(self, 
                 weight_val, 
                 k_val, 
                 proposal_temp,
                 device, 
                 **kwargs):
        super().__init__()
        self.weight_val = weight_val
        self.a_s = []
        self.hops = []
        self.k_val = k_val 
        self.temp = proposal_temp
        self.device = device
    
    def initialize_batch(self,
                         model, 
                         sentiment,
                         batch_size,
                         seq_length,
                         prompt_length,
                         **kwargs):
        
        model.set_biases(batch_size=batch_size, 
                         seq_len=seq_length,
                         prompt_length=prompt_length, 
                         attribute=sentiment,
                         device=self.device)
        initial_bias = torch.zeros(batch_size, 
                           seq_length - prompt_length, 
                           50257).to(self.device)
        self.embed_map = model.get_input_embeddings()
        return initial_bias

    def compute_p_lm(self, 
                     cur_bias, 
                     energy_fn): 
        
        loss, output_ids, onehot, logits, senti_losses = energy_fn(cur_bias)
        gx = torch.autograd.grad(loss, onehot)[0].detach()
        topk_ids = torch.topk(logits, self.k_val, dim=-1).indices
        gx_topk = gx[torch.arange(gx.size(0))[:, None, None], 
                     torch.arange(gx.size(1))[None, :, None], 
                     topk_ids]
        # gx_topk = torch.gather(gx, -1, topk_ids)
        token_dist = torch.ones_like(gx_topk).to(self.device) 
        token_dist[:, :, 0] = 0
        logits = gx_topk * token_dist
        proposal_dist = torch.distributions.Categorical(logits = -1 * logits / self.temp)
        sampled_dist_ids = proposal_dist.sample()
        actual_ids = topk_ids[torch.arange(topk_ids.size(0))[:, None],
                              torch.arange(topk_ids.size(1))[None, :],
                                sampled_dist_ids]
        return loss, output_ids, actual_ids, senti_losses
    
    def compute_bias(self, 
                     sampled_ids):
        # this is batch x seq_len x embed_dim
        cur_embeds = self.embed_map(sampled_ids)

        # compute ||embed - sampled_embed||^2 using foil 
        t1 = torch.einsum('ve -> v', [self.embed_map.weight ** 2])[None, None, :]
        t2 = torch.einsum('bse, ve -> bsv', [cur_embeds, self.embed_map.weight])
        t3 = torch.einsum('bse -> bs', [cur_embeds ** 2]).unsqueeze(-1)
        bias = -1 * self.weight_val * (t1 - 2 * t2 + t3)
        return bias 


    # first, compute the autoregressive generation
    # this should give the one hot vector, the loss, and the gradient
    # use the gradient to compute the distribution over the top-k tokens 
    def step(self, x, energy_fn, **kwargs):
        cur_bias = x
        loss, output_ids, sampled_ids, senti_losses = self.compute_p_lm(cur_bias, energy_fn)
        bias = self.compute_bias(sampled_ids)
        return bias, loss, output_ids, [senti_losses]