import os
from pathlib import Path

from transformers import GPT2LMHeadModel, GPTNeoForCausalLM
import torch
import torch.nn as nn
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
from .bleuloss import batch_log_bleulosscnn_ae, my_bleulosscnn_ae, my_bleuloss_STG


class GPTPromptTuningWithBiasesModelMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        use_full_prompt: bool = False,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = True

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            pass

        return model

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path

        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def set_biases(
        self,
        batch_size,
        seq_len,
        use_scale_weights,
        use_bolt_weights,
        device="cpu",
        disc_weight=0.9,
        use_soft_disc=False,
        **kwargs,
    ):
        self.use_soft_disc = use_soft_disc
        self.seq_len = seq_len

        self.trainable_weights = None

        self.labels = torch.LongTensor([1]).to(device)
        self.logits_processor = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=1.2),
            ]
        )
        self.len_logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(
                    seq_len, eos_token_id=self.config.eos_token_id
                ),
            ]
        )
        self.disc_weight = disc_weight
        self.use_scale_weights = use_scale_weights
        self.use_bolt_weights = use_bolt_weights
        self.stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=seq_len)]
        )

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def init_discriminator(self, discriminator: nn.Module):
        self.discriminator = discriminator
        self.discriminator.eval()
        self.sim_count = None

    def init_language_model(self, languagemodel: nn.Module, tokenizer):
        self.language_model = languagemodel
        self.language_model.eval()
        self.tokenizer = tokenizer

        ##### ending is '.' !!!!
        self.ending_target = torch.LongTensor([13]).to(self.device)

    def prompt_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Drop most of the args for now
        return self.forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )

    def soft_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
        use_full_prompt=False,
        keywords=None,
        biases=None,
        keywords_idx=None,
        keywords_token=None,
        use_cnn_batchloss=True,
        bias_rep_space="logit",
        weight=1,
        **kwargs,
    ):
        if bias_rep_space == "logit":
            forward_func = self.soft_greedy_search_with_biases
        else:
            forward_func = self.soft_greedy_search_with_biases_embed

        if not inference:
            if use_full_prompt:
                output_ids, onehot_generates, last_score, soft_generates, logits, _ = (
                    forward_func(
                        inputs_embeds,
                        input_ids,
                        logits_processor=self.logits_processor,
                        len_logits_processor=self.len_logits_processor,
                        stopping_criteria=self.stopping_criteria,
                        pad_token_id=self.config.eos_token_id,
                        return_last_score=True,
                        full_prompt=self.full_prompts,
                        sent_labels=None,
                        biases=biases,
                        use_hidden_states_biases=False,
                        return_logit=True,
                        trainable_weights=self.trainable_weights,
                        seq_len=self.seq_len,
                        is_dlp=True,
                        weight=weight,
                        use_bolt_weights=self.use_bolt_weights,
                        use_scale_weights=self.use_scale_weights,
                    )
                )
            else:
                output_ids, onehot_generates, last_score, soft_generates, logits, _ = (
                    forward_func(
                        inputs_embeds,
                        input_ids,
                        logits_processor=self.logits_processor,
                        len_logits_processor=self.len_logits_processor,
                        stopping_criteria=self.stopping_criteria,
                        pad_token_id=self.config.eos_token_id,
                        return_last_score=True,
                        sent_labels=None,
                        biases=biases,
                        use_hidden_states_biases=False,
                        return_logit=True,
                        trainable_weights=self.trainable_weights,
                        seq_len=self.seq_len,
                        is_dlp=True,
                        weight=weight,
                        use_bolt_weights=self.use_bolt_weights,
                        use_scale_weights=self.use_scale_weights,
                    )
                )
        else:
            if use_full_prompt:
                output_ids, onehot_generates, last_score, soft_generates, logits, _ = (
                    forward_func(
                        inputs_embeds,
                        input_ids,
                        logits_processor=self.logits_processor,
                        len_logits_processor=self.len_logits_processor,
                        stopping_criteria=self.stopping_criteria,
                        pad_token_id=self.config.eos_token_id,
                        inference=True,
                        return_last_score=True,
                        full_prompt=self.full_prompts,
                        sent_labels=None,
                        biases=biases,
                        use_hidden_states_biases=False,
                        return_logit=True,
                        trainable_weights=self.trainable_weights,
                        seq_len=self.seq_len,
                        is_dlp=True,
                        weight=weight,
                        use_bolt_weights=self.use_bolt_weights,
                        use_scale_weights=self.use_scale_weights,
                    )
                )
            else:
                output_ids, onehot_generates, last_score, soft_generates, logits, _ = (
                    forward_func(
                        inputs_embeds,
                        input_ids,
                        logits_processor=self.logits_processor,
                        len_logits_processor=self.len_logits_processor,
                        stopping_criteria=self.stopping_criteria,
                        pad_token_id=self.config.eos_token_id,
                        inference=True,
                        return_last_score=True,
                        sent_labels=None,
                        biases=biases,
                        use_hidden_states_biases=False,
                        return_logit=True,
                        trainable_weights=self.trainable_weights,
                        seq_len=self.seq_len,
                        is_dlp=True,
                        weight=weight,
                        use_bolt_weights=self.use_bolt_weights,
                        use_scale_weights=self.use_scale_weights,
                    )
                )

        # ste trick to make sure they have the same gradients
        logits = logits + onehot_generates - onehot_generates.detach()
        keywords_losses = batch_log_bleulosscnn_ae(logits, keywords, 1)
        keywords_loss = torch.mean(keywords_losses)
        
        loss = keywords_loss
        return loss, output_ids, onehot_generates, logits, keywords_losses


class FullPrompt(nn.Module):
    def __init__(self, n_tokens: int = 20, random_range: float = 0.5, config=None):
        super().__init__()
        self.full_prompts_matrix = torch.zeros(
            config.num_hidden_layers,
            2,
            config.n_head,
            n_tokens,
            config.n_embd // config.n_head,
        ).to("cuda")
        self.full_prompts_matrix.requires_grad = True
        self.full_prompts_matrix = nn.parameter.Parameter(self.full_prompts_matrix)

    def forward(self):
        return self.full_prompts_matrix


class GPTPromptTuningWithBiasModelLM(
    GPTPromptTuningWithBiasesModelMixin, GPT2LMHeadModel
):
    def __init__(self, config):
        super().__init__(config)
