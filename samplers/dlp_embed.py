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
                 use_diverse_initialization=False,
                 diverse_type='beam',
                 num_beams=100, 
                 num_beam_groups=100,
                 diversity_penalty=.8, 
                 diverse_addition_length=3,
                 is_kw=False, 
                 **kwargs):
        super().__init__()
        self.weight_val = weight_val
        self.a_s = []
        self.hops = []
        self.k_val = k_val 
        self.temp = proposal_temp
        self.diverse_type = diverse_type
        self.device = device
        self.use_diverse_initialization=use_diverse_initialization
        self.diverse_addition_length = diverse_addition_length
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.is_kw=is_kw
    
    def initialize_batch(self,
                         model, 
                         sentiment,
                         batch_size,
                         seq_length,
                         prompt_length,
                         inputs,
                         **kwargs):
        if self.use_diverse_initialization: 
            return self.diverse_initialization(model, 
                                                sentiment,
                                                batch_size, 
                                                seq_length, 
                                                prompt_length, 
                                                inputs)
        else: 
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
    
    def diverse_initialization(self,
                         model, 
                         sentiment,
                         batch_size,
                         seq_length,
                         prompt_length,
                         inputs,
                         **kwargs): 
        self.prompt_length = prompt_length + self.diverse_addition_length
        # inputs.input_ids = inputs.input_ids[0, :]
        # inputs.attention_mask = torch.ones_like(inputs.input_ids).to(self.device)
        new_inputs = model.generate(
            input_ids=inputs.input_ids[0, :].unsqueeze(0),
            num_return_sequences=self.num_beam_groups,
            top_k=batch_size,
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups,
            bad_words_ids=[[198], [628]],
            max_new_tokens=self.diverse_addition_length,
            diversity_penalty=self.diversity_penalty,
        )
        new_inputs = new_inputs.to(self.device)
        # running the discriminator to get the best sentences 
        # disc_embeds = model.discriminator.get_input_embeddings()(new_inputs)
        disc_logits = model.discriminator(new_inputs).logits
        loss = disc_logits[:, 1] - disc_logits[:, 0]
        best_idx = torch.topk(loss, batch_size).indices
        new_inputs = new_inputs[best_idx]
        model.set_biases(batch_size=batch_size, 
                         seq_len=seq_length,
                         prompt_length=self.prompt_length,
                         attribute=sentiment,
                         device=self.device)
        initial_bias = torch.zeros(batch_size, 
                           seq_length - self.prompt_length, 
                           50257).to(self.device)
        self.embed_map = model.get_input_embeddings()
        return new_inputs, initial_bias

    def compute_p_lm(self, 
                     cur_bias, 
                     energy_fn): 
        
        loss, output_ids, onehot, logits, senti_losses = energy_fn(cur_bias)
        gx = torch.autograd.grad(loss, onehot, allow_unused=True)
        gx = gx[0].detach()[:, self.prompt_length:, :]
        logits = logits[:, self.prompt_length:, :]
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
        return loss, output_ids, actual_ids, senti_losses.detach().cpu().numpy()
    

    def compute_p_lm_kw(self, 
                        cur_bias, 
                        energy_fn, 
                        kw_tokens,
                        cur_iter): 
        ppl_loss, output_ids, onehot, logits = energy_fn(cur_bias)
        if cur_iter == 0: 
            self.kw_target_idx = self.sample_position_kw(kw_tokens, output_ids)
        kw_losses = self.keyword_loss(onehot, self.kw_target_idx, kw_tokens)
        loss = kw_losses.sum()
        gx = torch.autograd.grad(loss, onehot, allow_unused=True)
        gx = gx[0].detach()[:, self.prompt_length:, :]
        logits = logits[:, self.prompt_length:, :]
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
        return loss, output_ids, actual_ids, kw_losses.detach().cpu().numpy()
    

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


    def step(self, **kwargs): 
        if self.is_kw: 
            return self.step_hard(**kwargs)
        else: 
            return self.step_soft(**kwargs)

    # first, compute the autoregressive generation
    # this should give the one hot vector, the loss, and the gradient
    # use the gradient to compute the distribution over the top-k tokens 
    def step_soft(self, x, energy_fn, **kwargs):
        cur_bias = x
        loss, output_ids, sampled_ids, senti_losses = self.compute_p_lm(cur_bias, energy_fn)
        bias = self.compute_bias(sampled_ids)
        return bias, loss, output_ids, [senti_losses]
    
    def step_hard(self, x, energy_fn, 
                kw_tokens, cur_iter, **kwargs):
        cur_bias = x
        loss, output_ids, sampled_ids, kw_losses = self.compute_p_lm_kw(cur_bias, energy_fn, kw_tokens[0], cur_iter)
        bias = self.compute_bias(sampled_ids)
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

    def keyword_loss(self, onehot, target_kw_idx, kw_token): 
        cur_embeds = torch.einsum('bv, ve -> be', [onehot[torch.arange(onehot.size(0)), target_kw_idx, :], 
                                  self.embed_map.weight])
        target_embeds = self.embed_map(kw_token)
        return torch.norm(cur_embeds - target_embeds, dim=-1)
