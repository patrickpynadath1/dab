from transformers import (
    GPT2TokenizerFast,
    AutoModelForSequenceClassification,

)
from .model_with_biases import GPTPromptTuningWithBiasModelLM as GPTBiasLM
from .model_with_diff_mask import GPTPromptTuningWithMaskModelLM as GPTDiffMaskLM

def load_tokenizer(model="gpt2"): 
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


def load_base_model(sampler, **kwargs):
    if sampler == "bolt": 
        base_lm = GPTBiasLM.from_pretrained(
                                **kwargs
                            )
    else: 
        base_lm = GPTDiffMaskLM.from_pretrained(
                                **kwargs
                            )
    return base_lm 


