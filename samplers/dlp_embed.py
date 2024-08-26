import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np

EPS = 1e-10


class LangevinSampler(nn.Module):
    def __init__(
        self,
        weight_val,
        proposal_temp,
        device,
        is_kw=False,
        use_bolt_weights=False,
        k_val=250,
        weight_strat="uniform",
        min_weight=1,
        max_weight=1,
        disc_weight=0.9,
        use_scale_weights=True,
        initialization="random_disc",
        initialization_noise_rate=0.5,
        **kwargs
    ):
        super().__init__()
        self.weight_val = weight_val
        self.a_s = []
        self.hops = []
        self.k_val = int(k_val)
        self.temp = float(proposal_temp)
        self.device = str(device)
        self.is_kw = is_kw
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.weight_strat = weight_strat
        self.disc_weight = disc_weight

        # weighting args
        self.use_scale_weights = use_scale_weights
        self.use_bolt_weights = use_bolt_weights
        self.initialization = initialization
        self.initialization_noise_rate = initialization_noise_rate

        # initializing sampling metrics to track
        self.sampled_tokens = []
        self.max_unnorm = []
        self.disc_loss = []
        self.cur_disc_loss = None

    def initialize_batch(
        self,
        model,
        sentiment,
        batch_size,
        seq_length,
        prompt_length,
        inputs,
        keyword_tokens=None,
        **kwargs
    ):
        if self.cur_disc_loss is not None:
            self.disc_loss.append(self.cur_disc_loss)
        self.cur_disc_loss = []
        self.prompt_length = prompt_length
        model.set_biases(
            batch_size=batch_size,
            seq_len=seq_length,
            prompt_length=prompt_length,
            attribute=sentiment,
            device=self.device,
            disc_weight=self.weight_val,
            use_scale_weights=self.use_scale_weights,
            use_bolt_weights=self.use_bolt_weights,
        )
        if keyword_tokens is not None:
            self.keyword_tokens = keyword_tokens.unsqueeze(dim=1).repeat(
                1, seq_length - prompt_length, 1
            )
        self.embed_map = model.get_input_embeddings()
        logit_dim = model.get_input_embeddings().weight.size(0)
        embed_dim = model.get_input_embeddings().weight.size(1)
        last_dim = logit_dim
        if self.initialization == "random_disc":
            sampled_ints = torch.randint(
                0, logit_dim, (batch_size, seq_length - prompt_length)
            ).to(self.device)
            if last_dim == embed_dim:
                initial_bias = self.embed_map(sampled_ints)
            else:
                initial_bias = self.compute_bias_l2_pen(sampled_ints)
        elif self.initialization == "random_cont":
            initial_bias = self.initialization_noise_rate * torch.randn(
                batch_size, seq_length - prompt_length, last_dim
            ).to(self.device)
        elif self.initialization == "zero":
            initial_bias = torch.zeros(
                batch_size, seq_length - prompt_length, last_dim
            ).to(self.device)
        elif self.initialization == "kw":
            sampled_kw_idx = torch.randint(
                0, keyword_tokens.size(-1), (batch_size, seq_length - prompt_length)
            ).to(self.device)
            sampled_kw = keyword_tokens[
                torch.arange(keyword_tokens.size(0))[:, None], sampled_kw_idx
            ]
            if last_dim == embed_dim:
                initial_bias = self.embed_map(sampled_kw)
            else:
                initial_bias = self.compute_bias_l2_pen(sampled_kw)
        self.weights = self.weight_val
        initial_bias = initial_bias.detach()
        initial_bias.requires_grad = True
        return inputs, initial_bias

    def calc_grad(self, loss, onehot):
        gx = torch.autograd.grad(loss, onehot, allow_unused=True)
        gx = gx[0].detach()[:, self.prompt_length :, :]
        return gx

    # computes the distribution over all the tokens in the model vocabulary
    def get_unfiltered_dist(self, gx, cur_token_ids, cur_bias=None):
        # print(gx.shape)
        token_dist = torch.ones_like(gx).to(self.device)
        token_dist[
            torch.arange(token_dist.size(0))[:, None, None],
            torch.arange(token_dist.size(1))[None, :, None],
            cur_token_ids[:, self.prompt_length :].unsqueeze(-1),
        ] = EPS
        unfiltered_dist = gx * token_dist
        return -1 * unfiltered_dist

    # selects the logits from unfiltered_dist for the tokens in the topk_ids
    def _apply_filter(self, unfiltered_dist, topk_ids):
        filtered_dist_logits = unfiltered_dist[
            torch.arange(unfiltered_dist.size(0))[:, None, None],
            torch.arange(unfiltered_dist.size(1))[None, :, None],
            topk_ids,
        ]
        return filtered_dist_logits

    # given the sampled topk indices, converts them to tokens from the vocab
    def _topk_to_tokens(self, topk_ids, sampled_indices):
        actual_ids = topk_ids[
            torch.arange(topk_ids.size(0))[:, None],
            torch.arange(topk_ids.size(1))[None, :],
            sampled_indices,
        ]
        return actual_ids

    # wrapper class for getting the dlp logits over the top k tokens
    # takes care of filtering the logits
    def get_dlp_dist(self, loss, onehot, cur_token_ids, logits):
        gx = self.calc_grad(loss, onehot)
        logits = logits[:, self.prompt_length :, :]
        unfiltered_dist = self.get_unfiltered_dist(gx, cur_token_ids)
        topk_ids = torch.topk(logits, self.k_val, dim=-1).indices
        return unfiltered_dist, topk_ids

    # given a kw_token, returns the
    # kw_top_k most similar tokens in terms of dot prod
    def compute_closest_embedding(self, kw_token, kw_top_k):
        kw_embeds = self.embed_map(kw_token)
        kw_top_k = torch.topk(
            torch.einsum("e, ve -> v", [kw_embeds, self.embed_map.weight]), k=kw_top_k
        )

    # performs the actual sampling of the bias tokens for soft constraints
    def compute_p_lm_soft(self, loss, output_ids, onehot, logits, senti_losses):
        unfiltered_dist, topk_ids = self.get_dlp_dist(loss, onehot, output_ids, logits)
        dist_logits = self._apply_filter(unfiltered_dist, topk_ids)
        proposal_dist = torch.distributions.Categorical(logits=dist_logits / self.temp)
        sampled_indices = proposal_dist.sample()
        sampled_tokens = self._topk_to_tokens(topk_ids, sampled_indices)
        return loss, output_ids, sampled_tokens, senti_losses.detach().cpu().numpy()

    # performs the actual sampling of the bias tokens for hard constraints
    def compute_p_lm_hard(self, loss, output_ids, onehot, logits, kw_losses):
        self.cur_disc_loss.append(kw_losses.mean().detach().cpu().numpy())
        unfiltered_dist, topk_ids = self.get_dlp_dist(loss, onehot, output_ids, logits)
        # ideally, this should capture the kw tokens + those that are semantically similar
        topk_ids = torch.concat([topk_ids, self.keyword_tokens], dim=-1)
        filtered_dist_logits = self._apply_filter(unfiltered_dist, topk_ids)
        proposal_dist = torch.distributions.Categorical(
            logits=filtered_dist_logits / self.temp
        )
        sampled_indices = proposal_dist.sample()
        sampled_tokens = self._topk_to_tokens(topk_ids, sampled_indices)
        return loss, output_ids, sampled_tokens, kw_losses.detach().cpu().numpy()

    # computes the l2 penalty bias, given the sampled embeddings
    def compute_bias_l2_pen(self, sampled_ids, kw_token=None):
        with torch.no_grad():
            # this is batch x seq_len x embed_dim
            cur_embeds = self.embed_map(sampled_ids)

            # compute ||embed - sampled_embed||^2 using foil
            t1 = torch.einsum("ve -> v", [self.embed_map.weight**2])[None, None, :]
            t2 = torch.einsum("bse, ve -> bsv", [cur_embeds, self.embed_map.weight])
            t3 = torch.einsum("bse -> bs", [cur_embeds**2]).unsqueeze(-1)
            bias = -1 * self.weight_val * (t1 - 2 * t2 + t3)
        return bias

    # wrapper for step -- just controls whether we
    # use step_soft or step_hard
    # determined by is_kw
    def step(self, x, energy_fn, **kwargs):
        loss, output_ids, onehot, logits, attr_losses = energy_fn(x)
        if self.is_kw:
            return self.step_hard(loss, output_ids, onehot, logits, attr_losses)
        else:
            return self.step_soft(loss, output_ids, onehot, logits, attr_losses)

    # Step function for soft constraints
    # x = the current bias embeddings (not the l2 penalty)
    # energy_fn = wrapper for the soft_forward function
    def step_soft(self, loss, output_ids, onehot, logits, senti_losses):
        loss, output_ids, sampled_ids, senti_losses = self.compute_p_lm_soft(
            loss, output_ids, onehot, logits, senti_losses
        )
        bias = self.compute_bias_l2_pen(sampled_ids)
        return bias, loss, output_ids, [senti_losses]

    # step function for hard constraints
    # x = the current bias embeddings (not the l2 penalty)
    # energy_fn = wrapper for the hard_forward function
    # kw_tokens = keyword tokens to be used for the hard constraint, dim  batch * num kw
    # cur_iter = current iteration of the sampler
    def step_hard(self, loss, output_ids, onehot, logits, kw_losses):
        loss, output_ids, sampled_ids, kw_losses = self.compute_p_lm_hard(
            loss, output_ids, onehot, logits, kw_losses
        )
        bias = self.compute_bias_l2_pen(sampled_ids)
        return bias, loss, output_ids, [kw_losses]

    # returns object for storing metrics for the sampling process
    def get_sampling_metrics(self):
        if self.cur_disc_loss is not None:
            self.disc_loss.append(self.cur_disc_loss)
        return {
            "bias_tokens": self.sampled_tokens,
            "max_unnorm": self.max_unnorm,
            "disc_loss": self.disc_loss,
        }
