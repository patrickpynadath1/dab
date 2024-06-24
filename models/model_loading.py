from transformers import (
    GPT2TokenizerFast,
    AutoModelForSequenceClassification,

)
from .model_with_biases import GPTPromptTuningWithBiasModelLM as GPT_BOLT_LM_senti
from .model_with_biases_mod import GPTPromptTuningWithBiasModelLM as GPT_DLP_LM_senti
from .keywords_model_with_biases import GPTPromptTuningWithBiasModelLM as GPT_BOLT_LM_keywords
from .keywords_model_with_biases_mod import GPTPromptTuningWithBiasModelLM as GPT_DLP_LM_keywords

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
            base_lm = GPT_BOLT_LM_senti.from_pretrained(
                                    **kwargs
                                )
        else: 
            base_lm = GPT_DLP_LM_senti.from_pretrained(
                                    **kwargs
                                )
    else: 
        if sampler == "bolt": 
            base_lm = GPT_BOLT_LM_keywords.from_pretrained(
                                    **kwargs
                                )
        else: 
            base_lm = GPT_DLP_LM_keywords.from_pretrained(
                                    **kwargs
                                )
    return base_lm 


