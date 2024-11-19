import torch
import time
import numpy as np
import pdb

from torch.nn.modules import loss
from .dlp_embed import LangevinSampler
import random
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    RepetitionPenaltyLogitsProcessor,
    MaxLengthCriteria,
    LogitsProcessor,
)
import itertools

EPS = 1e-10
shuffle = random.shuffle


def cross_entropy_loss(x, y):
    """
    Compute the cross-entropy loss given true labels and logits.

    Parameters:
    x (torch.Tensor): True labels of shape (N,), where N is the number of samples.
    y (torch.Tensor): Logits of shape (N, C), where C is the number of classes.

    Returns:
    torch.Tensor: The mean cross-entropy loss.
    """
    # Number of samples
    N = x.shape[0]

    # Compute softmax probabilities with numerical stability
    shifted_logits = y - y.max(dim=1, keepdim=True)[0]
    exp_logits = torch.exp(shifted_logits)
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    # Compute the log probabilities of the correct classes
    log_probs = -torch.log(probs[torch.arange(N), x])

    # Return the mean cross-entropy loss
    return log_probs.mean()


class SpecificTokenPenaltyProcessor(LogitsProcessor):
    def __init__(self, token_ids, penalty=100.0):
        """
        Penalizes specific tokens by subtracting a penalty from their logits.

        Args:
            token_ids (list of int): List of token IDs to penalize.
            penalty (float): Amount to penalize. Higher values reduce the likelihood more.
        """
        self.token_ids = token_ids
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        for token_id in self.token_ids:
            scores[:, token_id] -= self.penalty
        return scores


# Identify the token ID(s) you want to penalize (e.g., the newline token)


class SelectiveMaskingDLP:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.use_softmax_target = self.use_softmax_target == "t"
        self.labels = 1
        self.logits_processor = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=1.2),
                SpecificTokenPenaltyProcessor(token_ids=[198, 628], penalty=10),
            ]
        )
        self.initialize_metric_tracking()
        self.stop_sampling = False

    def calc_grad(self, onehot, disc, return_losses=False):

        dis_embs = torch.matmul(onehot, disc.get_input_embeddings().weight)
        senti_logits = disc(
            inputs_embeds=dis_embs,
            labels=torch.tensor(self.labels, device=dis_embs.device).repeat(
                dis_embs.shape[0]
            ),
        ).logits
        senti_losses = -(senti_logits[:, 1] - senti_logits[:, 0])
        # senti_losses = -1 * (senti_logits[:, 1].exp() - senti_logits[:, 0].exp())
        gx = torch.autograd.grad(senti_losses.mean(), onehot, allow_unused=True)
        gx = gx[0].detach()
        self.gx = gx
        return gx, senti_losses.mean().item()

    def get_disc_loss(self, onehot, disc):
        dis_embs = torch.matmul(onehot, disc.get_input_embeddings().weight)
        senti_logits = disc(
            inputs_embeds=dis_embs,
            labels=torch.tensor(self.labels, device=dis_embs.device).repeat(
                dis_embs.shape[0]
            ),
        ).logits
        senti_losses = -(senti_logits[:, 1] - senti_logits[:, 0])
        return senti_losses

    def _apply_filter(self, unfiltered_dist, topk_ids):
        filtered_dist_logits = unfiltered_dist[
            torch.arange(unfiltered_dist.size(0))[:, None, None],
            torch.arange(unfiltered_dist.size(1))[None, :, None],
            topk_ids,
        ]
        return filtered_dist_logits

    def get_unfiltered_dist(self, gx, cur_token_ids):
        # print(gx.shape)
        token_dist = torch.ones_like(gx).to(self.device)
        token_dist[
            torch.arange(token_dist.size(0))[:, None, None],
            torch.arange(token_dist.size(1))[None, :, None],
            cur_token_ids.unsqueeze(-1),
        ] = EPS
        unfiltered_dist = gx * token_dist
        return -1 * unfiltered_dist

    def _get_dlp_constraint_dist(self, gen_tokens, logits, model):
        onehot = torch.nn.functional.one_hot(gen_tokens, num_classes=logits.size(-1))
        onehot = onehot.float()
        onehot.requires_grad = True
        grad, losses = self.calc_grad(onehot, model.discriminator)
        # grad = self.calc_grad(loss, onehot)
        self.grad = grad
        topk_ids = torch.topk(logits, self.k_val, dim=-1).indices
        unfiltered_dist = self.get_unfiltered_dist(grad, gen_tokens)[
            :, -logits.size(1) :, :
        ]
        return unfiltered_dist / self.temp, topk_ids, losses

    def get_dlp_sample(self, gen_tokens, logits, model, gpt_logits=None):
        unfiltered_dist, topk_ids, losses = self._get_dlp_constraint_dist(
            gen_tokens, logits, model
        )
        filtered_dist = self._apply_filter(unfiltered_dist, topk_ids)
        if gpt_logits is not None:
            filtered_dist = (
                torch.norm(filtered_dist, dim=-1, keepdim=True)
                * gpt_logits
                / torch.norm(gpt_logits, dim=-1, keepdim=True)
            )
        proposal_dist = torch.distributions.Categorical(logits=filtered_dist)
        sampled_indices = proposal_dist.sample()
        sampled_tokens = self._topk_to_tokens(topk_ids, sampled_indices)
        return sampled_tokens, sampled_indices, topk_ids, losses

    def step(self, prev_gen, model, step_idx):
        # generate the gpt tokens from the padded tokens\
        max_length = self.target_seq_len + self.prompt_length - len(prev_gen)
        if max_length == 0:
            return prev_gen
        generated_output = model.generate(
            input_ids=prev_gen,
            max_length=max_length,
            min_length=max_length,
            logits_processor=self.logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        self.embed_map = model.get_input_embeddings()
        logits = generated_output.scores
        logits = torch.stack(logits).transpose(1, 0)
        generated_ids = generated_output.sequences[:, -logits.shape[1] :]
        self.cur_prompt_gens.append(generated_output.sequences.tolist())
        constraint_dist, total_topk_ids, losses = self._get_dlp_constraint_dist(
            generated_ids, logits, model
        )
        constraint_dist = self._apply_filter(constraint_dist, total_topk_ids)
        gpt_dist = self._apply_filter(logits, total_topk_ids)
        self.compute_all_metrics(gpt_dist, constraint_dist)
        if self.acceptance_method == "random":
            accepted = self.compute_acceptance_naive(step_idx, generated_ids.size(1))
        elif self.acceptance_method == "spec":
            accepted = self.compute_acceptance_spec(gpt_dist, constraint_dist)
        elif self.acceptance_method == "gpt_entropy":
            accepted = self.compute_acceptance_gpt_entropy(gpt_dist, step_idx)
        elif self.acceptance_method == "constraint_entropy":
            accepted = self.compute_acceptance_constraint_entropy(
                constraint_dist, step_idx
            )
        elif self.acceptance_method == "both_entropy":
            accepted = self.compute_acceptance_both_entropy(
                constraint_dist, gpt_dist, step_idx
            )
        elif self.acceptance_method == "exp_est_change":
            accepted = self.compute_acceptance_expected_estimated_change(
                gpt_dist, total_topk_ids, generated_ids.shape[1], step_idx
            )
        if self.stop_sampling or step_idx >= self.max_attempts - 1:
            self.complete_inprogress_generations.append(self.cur_prompt_gens)
            return generated_ids
        resample_idx = accepted.argmin(dim=-1).long()
        # computing metadata for grad
        generated_ids = self.sample_new_tokens(
            generated_ids,
            logits,
            resample_idx,
            constraint_dist,
            model,
            losses,
            total_topk_ids,
        )
        # we only keep generating if we are still below the target length
        self.continue_sampling = (
            len(generated_ids) < self.target_seq_len + self.prompt_length
        )
        return generated_ids

    def compute_acceptance_constraint_entropy(self, constraint_dist, step_idx):
        dist_logits = (
            constraint_dist.log_softmax(dim=-1) * constraint_dist.softmax(dim=-1)
        ).sum(dim=-1)
        if step_idx == 0:
            self.mean_entropy_initial = dist_logits.mean(dim=-1)
        # bias the logits towards picking positions earlier on
        accepted = (dist_logits < self.mean_entropy_initial) * 1.0
        return accepted

    # main idea: sample based on the coordinates
    def compute_acceptance_expected_estimated_change(
        self, gpt_dist, topk_ids, generation_length, step_idx
    ):
        # expecting grad to be cached
        grad = self.grad
        filtered_grad = self._apply_filter(grad, topk_ids)
        est_change = filtered_grad - filtered_grad[:, :, 0].unsqueeze(dim=-1)
        pos_change = est_change * ((est_change > 0) * 1.0)
        accepted = torch.ones(size=(1, generation_length))
        expected_est_change = (est_change * (gpt_dist).softmax(dim=1)).sum(dim=-1)

        self.expected_estimated_change.append(expected_est_change.tolist())
        rejected = expected_est_change.argmin(dim=-1)
        accepted[0, rejected] = 0
        return accepted

    def compute_acceptance_both_entropy(self, constraint_dist, gpt_dist, step_idx):
        dist_logits_constraint = (
            constraint_dist.log_softmax(dim=-1) * constraint_dist.softmax(dim=-1)
        ).sum(dim=-1)
        dist_logits_gpt = (gpt_dist.log_softmax(dim=-1) * gpt_dist.softmax(dim=-1)).sum(
            dim=-1
        )
        if step_idx == 0:
            self.mean_entropy_initial_constraint = dist_logits_constraint.mean(dim=-1)
            self.mean_entropy_initial_gpt = dist_logits_gpt.mean(dim=-1)
        # first, only accepting tokens that lm is relatively confident in
        accepted_gpt = (dist_logits_gpt > self.mean_entropy_initial_gpt) * 1.0
        accepted_constraint = (
            dist_logits_constraint > self.mean_entropy_initial_constraint
        ) * 1.0
        # rejecting if it is either below the initial entropy mean or the initial constraint entropy
        accepted = torch.minimum(accepted_gpt, accepted_constraint)
        return accepted

    def compute_acceptance_gpt_entropy(self, gpt_dist, step_idx):
        dist_logits = (gpt_dist.log_softmax(dim=-1) * gpt_dist.softmax(dim=-1)).sum(
            dim=-1
        )
        if step_idx == 0:
            self.mean_entropy_initial = dist_logits.mean(dim=-1)
        # bias the logits towards picking positions earlier on
        accepted = (dist_logits > self.mean_entropy_initial) * 1.0
        return accepted

    def compute_acceptance_spec(self, gpt_dist, constraint_dist):

        to_accept_logits = (
            (constraint_dist).softmax(dim=-1) / (gpt_dist.softmax(dim=-1))
        ) ** self.bal_val
        to_accept_logits = torch.clamp(to_accept_logits, min=0, max=1)
        # we want to find the probability of accepting the autoregressive generations
        # that is just the to_accept logits of the first row, since the first row of the
        # top k correspends to greedy output
        to_accept_probs = to_accept_logits[:, :, 0]
        accepted = (
            torch.distributions.Bernoulli(probs=to_accept_probs).sample().squeeze(-1)
        )
        if torch.all(accepted == 1):
            self.stop_sampling = True
        return accepted

    def compute_acceptance_naive(self, step_idx, generation_length):

        accepted = torch.ones(size=(1, generation_length))
        if step_idx == 0:
            self.pos_to_update = self.random_to_alter
        if self.pos_to_update >= 0:
            to_change_idx = random.randint(0, accepted.size(1) - 1)
            accepted[0, to_change_idx] = 0
        return accepted

    def _topk_to_tokens(self, topk_ids, sampled_indices):
        actual_ids = topk_ids[
            torch.arange(topk_ids.size(0))[:, None],
            torch.arange(topk_ids.size(1))[None, :],
            sampled_indices,
        ]
        return actual_ids

    def sample_from_topk_dist(self, logits, total_topk_ids):
        sampled_topk_token = torch.distributions.Categorical(
            probs=logits.softmax(dim=-1)
        ).sample()
        sampled_token = self._topk_to_tokens(total_topk_ids, sampled_topk_token)
        return sampled_token

    def sample_new_tokens(
        self,
        generated_ids,
        logits,
        resample_idx,
        constraint_dist,
        model,
        cur_loss,
        topk_ids,
    ):
        orig_generated_ids = generated_ids.clone()
        num_steps = self.num_constraint_steps
        sampled_token = self.sample_from_topk_dist(constraint_dist, topk_ids)
        generated_ids[0, resample_idx] = sampled_token[0, resample_idx]
        losses = [cur_loss]
        tokens = [orig_generated_ids[0, resample_idx].item()]
        if num_steps >= 1:
            for _ in range(num_steps):
                new_sampled_token, _, _, new_loss = self.get_dlp_sample(
                    generated_ids, logits, model
                )
                losses.append(new_loss)
                tokens.append(generated_ids[0, resample_idx].item())
                generated_ids[0, resample_idx] = new_sampled_token[0, resample_idx]
            best_token_idx = np.argmin(losses)
            sampled_token = tokens[best_token_idx]
            generated_ids[0, resample_idx] = sampled_token

        generated_ids = generated_ids[:, : resample_idx + 1]
        return generated_ids

    # auxilary functions for computing metrics
    # TODO: define functions for each aspect I want to track
    def initialize_metric_tracking(self):
        self.complete_inprogress_generations = []
        self.cur_prompt_gens = []
        self.prev_gen_tokens = None
        self.altered_dist_zeros = []
        self.altered_dist_max = []
        self.altered_dist_min = []
        self.altered_dist_mean = []
        self.acceptance_mean = []
        self.acceptance_std = []
        self.constraint_entropy = []
        self.lm_entropy = []
        self.percentage_grad_neg = []
        self.percentage_grad_neg_seq_mean = []
        self.grad_max_above_mean = []
        self.grad_min_below_mean = []
        self.pos_neg_grad_dot = []
        self.lm_entropy_total = []
        self.constraint_entropy_total = []
        self.expected_estimated_change = []
        return

    def compute_all_metrics(self, gpt_dist, constraint_dist):
        self.compute_gpt_entropy(gpt_dist)
        self.compute_constraint_entropy(constraint_dist)
        return

    def compute_gpt_entropy(self, gpt_dist):
        dist_logits = (gpt_dist.log_softmax(dim=-1) * gpt_dist.softmax(dim=-1)).sum(
            dim=-1
        )
        self.lm_entropy.append(dist_logits.tolist())
        return

    def compute_constraint_entropy(self, constraint_dist):
        dist_logits = (
            constraint_dist.log_softmax(dim=-1) * constraint_dist.softmax(dim=-1)
        ).sum(dim=-1)
        self.constraint_entropy.append(dist_logits.tolist())
        return

    def get_sampling_metrics(self):
        return {
            "all_gens": self.cur_prompt_gens,
            "constraint_entropy": self.constraint_entropy,
            "lm_entropy": self.lm_entropy,
            "exp_est_change": self.expected_estimated_change,
        }
