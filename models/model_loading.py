from transformers import (
    GPT2TokenizerFast,
    AutoModelForSequenceClassification,

)
from .model_with_biases import GPTPromptTuningWithBiasModelLM as GPTBiasLM_senti
from .model_with_biases_mod import GPTPromptTuningWithBiasModelLM as GPTDiffMaskLM_senti
from .keywords_model_with_biases import GPTPromptTuningWithBiasModelLM as GPTBiasLM_keywords
from .keywords_model_with_diff_mask import GPTPromptTuningWithMaskModelLM as GPTMaskLM_keywords

def load_tokenizer(model="gpt2-large"): 
    tokenizer = GPT2TokenizerFast.from_pretrained(model)
    return tokenizer


def load_sentiment_discriminator():
    discriminator = AutoModelForSequenceClassification.from_pretrained(
            "./checkpoints/replaced_vocab_roberta_for_yelp_polarity"
        ) 
    return discriminator


def load_toxicity_discriminator():
    discriminator = AutoModelForSequenceClassification.from_pretrained(
            "./checkpoints/replaced_vocab_roberta_for_jigsaw"
        )  
    return discriminator


def load_keyword_discriminator(): 
    return 


def load_base_model(sampler, use_senti=True, **kwargs):
    if use_senti: 
        if sampler == "bolt": 
            base_lm = GPTBiasLM_senti.from_pretrained(
                                    **kwargs
                                )
        else: 
            base_lm = GPTDiffMaskLM_senti.from_pretrained(
                                    **kwargs
                                )
    else: 
        if sampler == "bolt": 
            base_lm = GPTBiasLM_keywords.from_pretrained(
                                    **kwargs
                                )
        else: 
            base_lm = GPTMaskLM_keywords.from_pretrained(
                                    **kwargs
                                )
    return base_lm 


