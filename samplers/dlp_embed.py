import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
EPS = 1e-10
class LangevinSampler(nn.Module):
    def __init__(self, 
                 weight_val, 
                 proposal_temp,
                 device,
                 filter_type,
                 p_val,
                 k_val,
                 filter_logit,
                 **kwargs):
        super().__init__()
        self.weight_val = weight_val
        self.a_s = []
        self.hops = []
        self.k_val = k_val 
        self.temp = proposal_temp
        self.device = device
        self.filter_type = filter_type
        self.p_val = p_val
        self.filter_logit= filter_logit
    
    def initialize_batch(self,
                         model, 
                         sentiment,
                         batch_size,
                         seq_length,
                         prompt_length,
                         inputs,
                         **kwargs):
        self.prompt_length = prompt_length
        model.set_biases(batch_size=batch_size, 
                        seq_len=seq_length,
                        prompt_length=prompt_length, 
                        attribute=sentiment,
                        device=self.device)
        initial_bias = torch.zeros(batch_size, 
                        seq_length - prompt_length, 
                        50257).to(self.device)
        self.embed_map = model.get_input_embeddings()
        return inputs, initial_bias
        

    def calc_grad(self, loss, onehot): 
        gx = torch.autograd.grad(loss, onehot, allow_unused=True)
        gx = gx[0].detach()[:, self.prompt_length:, :]
        return gx
    
    def get_unfiltered_dist(self, gx, cur_token_ids):
        token_dist = torch.ones_like(gx).to(self.device)
        token_dist[torch.arange(token_dist.size(0))[:, None, None],
                    torch.arange(token_dist.size(1))[None, :, None], 
                    cur_token_ids[:, self.prompt_length:].unsqueeze(-1)] = EPS 
        unfiltered_dist = - gx * token_dist
        return unfiltered_dist


    def get_top_k_dlp_dist(self, 
                         loss, 
                         onehot,
                         cur_token_ids,
                         logits): 
        gx = self.calc_grad(loss, onehot)
        logits = logits[:, self.prompt_length:, :]
        unfiltered_dist = self.get_unfiltered_dist(gx, cur_token_ids)
        if self.filter_logit == "gpt":
            topk_ids = torch.topk(logits, self.k_val, dim=-1).indices
        else: 
            topk_ids = torch.topk(unfiltered_dist, self.k_val, dim=-1).indices
        filtered_dist_logits = unfiltered_dist[torch.arange(unfiltered_dist.size(0))[:, None, None],
                                                  torch.arange(unfiltered_dist.size(1))[None, :, None],
                                                    topk_ids]
        return filtered_dist_logits, topk_ids
    

    # first need to sort the logits 
    # then need to do torch cum sum in order to figure out where the p cut off is 
    # then need to take the top p logits, where the dimension is the largest out of all sequences 
    # 
    def get_top_p_dlp_dist(self, 
                         loss,
                         onehot,
                         cur_token_ids,
                         logits):
        gx = self.calc_grad(loss, onehot)
        logits = logits[:, self.prompt_length:, :]
        unfiltered_dist = self.get_unfiltered_dist(gx, cur_token_ids)
        if self.filter_logit == "gpt":
            logits_to_filter = logits.softmax(dim=-1)
        else: 
            logits_to_filter = unfiltered_dist.softmax(dim=-1)
        
        sorted_logits, sorted_indices = torch.sort(logits_to_filter, dim=-1, descending=True)
        cumsum_logits = torch.cumsum(sorted_logits, dim=-1)
        cumsum_cutoff_idx = ((cumsum_logits > self.p_val) * 1.0).argmax(dim=-1)
        cumsum_cutoff_idx = torch.where(cumsum_cutoff_idx == 0, 1, cumsum_cutoff_idx)
        max_cutoff_idx = cumsum_cutoff_idx.max()
        topk_ids = sorted_indices[:, :, :max_cutoff_idx]
        dist_logits_premask = unfiltered_dist[torch.arange(unfiltered_dist.size(0))[:, None, None],
                                              torch.arange(unfiltered_dist.size(1))[None, :, None],
                                              topk_ids]

        # now need to mask out the logits that are not in the top p by setting to -inf
        tmp_mask = dist_logits_premask / ((cumsum_logits < self.p_val)*1.0)[:, :, :max_cutoff_idx]
        masked_logits = torch.where(tmp_mask == float('inf'), 
                                    torch.tensor(float('-inf'), 
                                                 dtype=tmp_mask.dtype), 
                                    tmp_mask)
        # making sure at least one index is a scalar value, or else the torch.distributions.Categorical will throw an error
        masked_logits[:, :, 0] = dist_logits_premask[:, :, 0]
        return masked_logits, topk_ids

    def compute_p_lm(self, 
                     cur_bias, 
                     energy_fn): 

        loss, output_ids, onehot, logits, senti_losses = energy_fn(cur_bias)
        if self.filter_type == "topk":
            dist_logits, topk_ids = self.get_top_k_dlp_dist(loss, onehot, output_ids, logits)
        elif self.filter_type == "topp":
            dist_logits, topk_ids = self.get_top_p_dlp_dist(loss, onehot, output_ids, logits)
        proposal_dist = torch.distributions.Categorical(logits =  dist_logits / self.temp)
        sampled_dist_ids = proposal_dist.sample()
        actual_ids = topk_ids[torch.arange(topk_ids.size(0))[:, None],
                              torch.arange(topk_ids.size(1))[None, :],
                                sampled_dist_ids]
        return loss, output_ids, actual_ids, senti_losses.detach().cpu().numpy()
    
    def compute_bias(self, 
                     sampled_ids):
        with torch.no_grad():
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
