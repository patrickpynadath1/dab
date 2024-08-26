from transformers import (
    GPT2TokenizerFast,
    AutoModelForSequenceClassification,
)
from .model_with_biases import GPTPromptTuningWithBiasModelLM as GPT_BOLT_LM_senti
from .model_with_biases_mod import GPTPromptTuningWithBiasModelLM as GPT_DLP_LM_senti
from .keywords_model_with_biases import (
    GPTPromptTuningWithBiasModelLM as GPT_BOLT_LM_keywords,
)
from .keywords_model_with_biases_mod import (
    GPTPromptTuningWithBiasModelLM as GPT_DLP_LM_keywords,
)
from .reasoning_model_with_biases import (
    GPTPromptTuningWithBiasModelLM as GPT_BOLT_LM_reasoning,
)
from .reasoning_model_with_biases_mod import (
    GPTPromptTuningWithBiasModelLM as GPT_DLP_LM_reasoning,
)


def load_tokenizer(model="gpt2-large"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model)
    return tokenizer


def load_sentiment_discriminator():
    discriminator = AutoModelForSequenceClassification.from_pretrained(
        "./checkpoints/BOLT_models/replaced_vocab_roberta_for_yelp_polarity"
    )
    return discriminator


def load_toxicity_discriminator():
    discriminator = AutoModelForSequenceClassification.from_pretrained(
        "./checkpoints/BOLT_models/replaced_vocab_roberta_for_jigsaw"
    )
    return discriminator


def load_keyword_discriminator():
    return


def load_base_model(sampler, mode="senti", **kwargs):
    if mode == "senti":
        if sampler == "bolt":
            base_lm = GPT_BOLT_LM_senti.from_pretrained(**kwargs)
        else:
            base_lm = GPT_DLP_LM_senti.from_pretrained(**kwargs)
    elif mode == "keywords":
        if sampler == "bolt":
            base_lm = GPT_BOLT_LM_keywords.from_pretrained(**kwargs)
        else:
            base_lm = GPT_DLP_LM_keywords.from_pretrained(**kwargs)
    elif mode == "reasoning":
        if sampler == "bolt":
            base_lm = GPT_BOLT_LM_reasoning.from_pretrained(**kwargs)
        else:
            base_lm = GPT_DLP_LM_reasoning.from_pretrained(**kwargs)
    return base_lm
