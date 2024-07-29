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
                 use_diverse_initialization=False,
                 diverse_type='beam',
                 num_beams=100, 
                 num_beam_groups=100,
                 diversity_penalty=.8, 
                 diverse_addition_length=3,
                 is_kw=False,
                 use_cnn_batchloss=False,
                 k_val=250,
                 bias_compute_method="penalty",
                 weight_strat='uniform',
                 min_weight=1,
                 max_weight=1,
                 weight_lr=1e-3,
                 filter_type='topk',
                 bias_rep_space='logit',
                 proposal_rep_space='logit',
                 step_size=.1,
                 disc_weight=.9, 
                 use_bolt_weights=True, 
                 use_scale_weights=True, 
                 **kwargs):
        super().__init__()
        self.weight_val = weight_val
        self.a_s = []
        self.hops = []
        self.k_val = int(k_val)
        self.temp = float(proposal_temp)
        self.device = str(device)
        self.use_diverse_initialization=use_diverse_initialization
        self.diverse_addition_length = diverse_addition_length
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.is_kw=is_kw
        self.use_cnn_batchloss = use_cnn_batchloss
        self.bias_compute_method = bias_compute_method
        self.max_weight = max_weight
        self.min_weight = min_weight 
        self.weight_lr = weight_lr
        self.weight_strat = weight_strat
        self.filter_type = filter_type
        self.bias_rep_space = bias_rep_space
        self.proposal_rep_space=proposal_rep_space
        self.step_size = step_size 
        self.disc_weight = disc_weight
        self.use_bolt_weights = use_bolt_weights
        self.use_scale_weights = use_scale_weights

        # initializing sampling metrics to track
        self.sampled_tokens = []
        self.max_unnorm = []
        self.energy = []

    def calc_linear_weights(self, 
                            num_gen_tokens):
        return torch.Tensor(np.linspace(self.min_weight, self.max_weight, num_gen_tokens)).to(self.device)

    def calc_bolt_weights(self, 
                          num_tokens):
        t = torch.linspace(1, num_tokens+1, num_tokens).to(self.device)
        return self.weight_val * (1 - t / (num_tokens+1))


    def initialize_batch(self,
                         model, 
                         sentiment,
                         batch_size,
                         seq_length,
                         prompt_length,
                         inputs,
                         keyword_tokens=None,
                         **kwargs):
        self.prompt_length = prompt_length
        model.set_biases(batch_size=batch_size, 
                        seq_len=seq_length,
                        prompt_length=prompt_length, 
                        attribute=sentiment,
                        device=self.device, 
                        disc_weight=self.disc_weight,
                        use_scale_weights=self.use_scale_weights, 
                        use_bolt_weights=self.use_bolt_weights)
        logit_dim = model.get_input_embeddings().weight.size(0)
        embed_dim = model.get_input_embeddings().weight.size(1)
        if self.bias_rep_space == 'logit':
            initial_bias = torch.zeros(batch_size, 
                        seq_length - prompt_length, 
                        logit_dim).to(self.device)
        else: 
            initial_bias = torch.zeros(batch_size, 
                        seq_length - prompt_length, 
                        embed_dim).to(self.device) 
        if keyword_tokens is not None: 
            self.keyword_tokens = keyword_tokens.unsqueeze(dim=1).repeat(1, seq_length - prompt_length, 1)
        self.embed_map = model.get_input_embeddings()
        if self.weight_strat == 'uniform':
            self.weights = self.weight_val 
        elif self.weight_strat == 'linear':
            self.weights = self.calc_linear_weights(seq_length - prompt_length)
        elif self.weight_strat == 'bolt':
            self.weights = self.calc_bolt_weights(seq_length - prompt_length)
        elif self.weight_strat == 'learn': 
            self.weights = torch.ones((initial_bias.size(0), initial_bias.size(1))).to(self.device)
            self.weight_optim = torch.optim.Adam([self.weights], lr=self.weight_lr)    
        initial_bias.requires_grad = True     
        return inputs, initial_bias
        

    def calc_grad_logit(self, loss, onehot): 
        gx = torch.autograd.grad(loss, onehot, allow_unused=True)
        gx = gx[0].detach()[:, self.prompt_length:, :]
        return gx
    
    def calc_grad_embed(self, loss, embed_bias): 
        gx = torch.autograd.grad(loss, embed_bias, allow_unused=True)
        gx = gx[0].detach()
        return gx
    
    def get_unfiltered_dist(self, gx, cur_token_ids, cur_bias=None):
        # print(gx.shape)
        if self.proposal_rep_space == 'logit':
            token_dist = torch.ones_like(gx).to(self.device)
            token_dist[torch.arange(token_dist.size(0))[:, None, None],
                        torch.arange(token_dist.size(1))[None, :, None], 
                        cur_token_ids[:, self.prompt_length:].unsqueeze(-1)] = EPS 
            unfiltered_dist = gx * token_dist
        else:
            t1_1 = torch.einsum('bse, ve -> bsv', [gx, self.embed_map.weight])
            t1_2 = torch.einsum('bse, bse -> bs', [gx, cur_bias]).unsqueeze(-1)
            t2_1 = torch.einsum('ve -> v', [self.embed_map.weight ** 2])[None, None, :]
            t2_2 = torch.einsum('bse, ve -> bsv', [cur_bias, self.embed_map.weight])
            t2_3 = torch.einsum('bse -> bs', [cur_bias ** 2]).unsqueeze(-1)

            unfiltered_dist = .5 * (t1_1 - t1_2) + (t2_1 - 2 * t2_2 + t2_3) / self.step_size
        return -1 * unfiltered_dist


    def get_top_k_dlp_dist_embed(self, 
                                 loss, 
                                 cur_bias,
                                 onehot, 
                                 cur_token_ids, 
                                 logits):
        if self.proposal_rep_space == 'logit':
            gx = self.calc_grad_logit(loss, onehot)
        else:
            gx = self.calc_grad_embed(loss, cur_bias)
        logits = logits[:, self.prompt_length:, :]
        unfiltered_dist = self.get_unfiltered_dist(gx, cur_token_ids, cur_bias=cur_bias)
        topk_ids = torch.topk(logits, self.k_val, dim=-1).indices
        filtered_dist_logits = unfiltered_dist[torch.arange(unfiltered_dist.size(0))[:, None, None],
                                                  torch.arange(unfiltered_dist.size(1))[None, :, None],
                                                    topk_ids]
        return filtered_dist_logits, topk_ids

    def get_top_k_dlp_dist_logit(self, 
                         loss, 
                         onehot,
                         cur_token_ids,
                         logits): 
        gx = self.calc_grad_logit(loss, onehot)
        logits = logits[:, self.prompt_length:, :]
        unfiltered_dist = self.get_unfiltered_dist(gx, cur_token_ids)
        topk_ids = torch.topk(logits, self.k_val, dim=-1).indices
        filtered_dist_logits = unfiltered_dist[torch.arange(unfiltered_dist.size(0))[:, None, None],
                                                  torch.arange(unfiltered_dist.size(1))[None, :, None],
                                                    topk_ids]
        # print(filtered_dist_logits.var(dim=-1).mean())
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
        gx = self.calc_grad_logit(loss, onehot)
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

    def compute_p_lm_embed_soft(self, 
                                cur_bias, 
                                energy_fn): 
        loss, output_ids, onehot, logits, senti_losses = energy_fn(cur_bias)
        dist_logits, topk_ids = self.get_top_k_dlp_dist_embed(loss, cur_bias, onehot, output_ids, logits)
        proposal_dist = torch.distributions.Categorical(logits =  dist_logits / self.temp)
        sampled_dist_ids = proposal_dist.sample()
        actual_ids = topk_ids[torch.arange(topk_ids.size(0))[:, None],
                              torch.arange(topk_ids.size(1))[None, :],
                                sampled_dist_ids]
        return loss, output_ids, actual_ids, senti_losses.detach().cpu().numpy()


    def compute_p_lm_logit_soft(self, 
                     cur_bias, 
                     energy_fn): 

        loss, output_ids, onehot, logits, senti_losses = energy_fn(cur_bias)
        if self.filter_type == "topk":
            dist_logits, topk_ids = self.get_top_k_dlp_dist_logit(loss, onehot, output_ids, logits)
        elif self.filter_type == "topp":
            dist_logits, topk_ids = self.get_top_p_dlp_dist(loss, onehot, output_ids, logits)
        proposal_dist = torch.distributions.Categorical(logits =  dist_logits / self.temp)
        sampled_dist_ids = proposal_dist.sample()
        actual_ids = topk_ids[torch.arange(topk_ids.size(0))[:, None],
                              torch.arange(topk_ids.size(1))[None, :],
                                sampled_dist_ids]
        return loss, output_ids, actual_ids, senti_losses.detach().cpu().numpy()
    

    # idea: instead of just using top k on the gpt logit, do 
    # top k on the embeddings closest to key word embedding 
    # not sure how this would translate to a phrase, but lets just see if it works for now 
    def compute_closest_embedding(self, 
                                  kw_token, 
                                  kw_top_k):
        kw_embeds = self.embed_map(kw_token)
        kw_top_k = torch.topk(torch.einsum('e, ve -> v', [kw_embeds, self.embed_map.weight]), k=kw_top_k)
        return kw_top_k.indices

    def step_weights(self, 
                     cur_bias,
                     energy_fn):
        self.weight_optim.zero_grad() 
        cur_bias = cur_bias * self.weights.unsqueeze(-1)
        ppl_loss, output_ids, onehot, logits = energy_fn(cur_bias)
        ppl_loss.backward()
        self.weight_optim.step()
        return


    def compute_p_lm_kw(self, 
                        cur_bias, 
                        energy_fn, 
                        kw_tokens,
                        cur_iter): 
        ppl_loss, output_ids, onehot, logits = energy_fn(cur_bias)
        loss = ppl_loss
        kw_losses = torch.zeros_like(logits)
        if self.proposal_rep_space == 'logit':
            gx = self.calc_grad_logit(loss, onehot)
        else: 
            gx = self.calc_grad_embed(loss, cur_bias)
        unfiltered_dist = self.get_unfiltered_dist(gx, output_ids, cur_bias)
        logits = logits[:, self.prompt_length:, :]
        topk_ids = torch.topk(logits, self.k_val, -1).indices
        # ideally, this should capture the kw tokens + those that are semantically similar 
        topk_ids = torch.concat([topk_ids, self.keyword_tokens], dim=-1)
        filtered_dist_logits = unfiltered_dist[torch.arange(unfiltered_dist.size(0))[:, None, None],
                                                  torch.arange(unfiltered_dist.size(1))[None, :, None],
                                                    topk_ids] 
        proposal_dist = torch.distributions.Categorical(logits = filtered_dist_logits / self.temp)
        sampled_dist_ids = proposal_dist.sample()
        actual_ids = topk_ids[torch.arange(topk_ids.size(0))[:, None],
                              torch.arange(topk_ids.size(1))[None, :],
                                sampled_dist_ids]
        return loss, output_ids, actual_ids, kw_losses.detach().cpu().numpy()
    

    def compute_bias_l2_pen(self, 
                     sampled_ids, kw_token=None):
        with torch.no_grad():
            # this is batch x seq_len x embed_dim
            cur_embeds = self.embed_map(sampled_ids)

            # compute ||embed - sampled_embed||^2 using foil 
            t1 = torch.einsum('ve -> v', [self.embed_map.weight ** 2])[None, None, :]
            t2 = torch.einsum('bse, ve -> bsv', [cur_embeds, self.embed_map.weight])
            t3 = torch.einsum('bse -> bs', [cur_embeds ** 2]).unsqueeze(-1)
            bias = -1 * self.weight_val * (t1 - 2 * t2 + t3)
        return bias
    
    def compute_bias_l2_inv(self, 
                            sampled_ids, 
                            kw_token):
        with torch.no_grad():
            cur_embeds = self.embed_map(sampled_ids)

            # compute ||embed - sampled_embed||^2 using foil 
            t1 = torch.einsum('ve -> v', [self.embed_map.weight ** 2])[None, None, :]
            t2 = torch.einsum('bse, ve -> bsv', [cur_embeds, self.embed_map.weight])
            t3 = torch.einsum('bse -> bs', [cur_embeds ** 2]).unsqueeze(-1)
            l2_dist = (t1 - 2 * t2 + t3)
            bias = 1 / (l2_dist + EPS)
        return bias
    
    def compute_bias_dot_exp(self, 
                             sampled_ids,
                             kw_token): 
        with torch.no_grad(): 
            cur_embeds = self.embed_map(sampled_ids)
            dots = torch.einsum('bse, ve -> bsv', [cur_embeds, self.embed_map.weight])
            bias = torch.exp(dots)
        return bias


    def step(self, **kwargs):
        if self.bias_rep_space == 'logit':
            if self.is_kw: 
                return self.step_hard_logit(**kwargs)
            else: 
                return self.step_soft_logit(**kwargs)
        else: 
            return self.step_soft_embed(**kwargs)

    def step_soft_embed(self, x, energy_fn, **kwargs): 
        cur_bias = x
        if self.is_kw: 
            loss, output_ids, sampled_ids, senti_losses = self.compute_p_lm_kw(cur_bias, energy_fn, kw_tokens = kwargs['kw_tokens'], cur_iter=kwargs['cur_iter'])
        else: 
            loss, output_ids, sampled_ids, senti_losses = self.compute_p_lm_embed_soft(cur_bias, energy_fn)
        bias = self.embed_map(sampled_ids)
        return bias, loss, output_ids, [senti_losses]


    # first, compute the autoregressive generation
    # this should give the one hot vector, the loss, and the gradient
    # use the gradient to compute the distribution over the top-k tokens 
    def step_soft_logit(self, x, energy_fn, **kwargs):
        cur_bias = x
        loss, output_ids, sampled_ids, senti_losses = self.compute_p_lm_logit_soft(cur_bias, energy_fn)
        bias = self.compute_bias_l2_pen(sampled_ids)
        return bias, loss, output_ids, [senti_losses]
    

    def step_hard_logit(self, x, energy_fn, 
                kw_tokens, cur_iter, **kwargs):
        cur_bias = x
        loss, output_ids, sampled_ids, kw_losses = self.compute_p_lm_kw(cur_bias, energy_fn, kw_tokens, cur_iter)
        if self.bias_compute_method == 'l2_pen':
            bias = self.compute_bias_l2_pen(sampled_ids, kw_tokens)
        elif self.bias_compute_method == 'l2_inv': 
            bias = self.compute_bias_l2_inv(sampled_ids, kw_tokens)
        else: 
            bias = self.compute_bias_dot_exp(sampled_ids, kw_tokens)
        if self.weight_strat == 'learn':
            self.step_weights(cur_bias, energy_fn)
        self.sampled_tokens.append(sampled_ids)
        if self.weight_strat == 'learn':
            bias = bias * self.weights.unsqueeze(dim=-1)
        else: 
            bias = bias * self.weights.unsqueeze(0).unsqueeze(-1)
        return bias, loss, output_ids, [kw_losses]
    
    ### function for sampling POSITIONS along the sequence 
    def sample_position_kw(self, 
                           keyword_tokens,
                           tokens):
        # first, compute a distribution over the positions
        cur_embeds = self.embed_map(tokens)
        keyword_embeds = self.embed_map(keyword_tokens)
        # right now, doing dot product 
        # can change later 
        position_logits = torch.einsum('bse, e -> bs', [cur_embeds, keyword_embeds])
        position_dist = torch.distributions.Categorical(logits=position_logits)
        sampled_pos = position_dist.sample()
        return sampled_pos

    def keyword_loss(self, logits, target_kw_idx, kw_token): 
        kw_log_prob = logits[torch.arange(logits.size(0))[:, None, None],
                             target_kw_idx[None, :, None],
                                kw_token]
        return -1 * (kw_log_prob)
    
    def get_metrics_to_store(self): 
        return {'bias_tokens': self.sampled_tokens, 
                'max_unnorm': self.max_unnorm}
