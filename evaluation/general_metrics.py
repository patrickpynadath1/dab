import torch
import numpy
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # for perplexity
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)  # for cola
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from typing import List
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk import word_tokenize
from typing import List
from nltk.util import bigrams, trigrams, everygrams
from evaluate import load
from tqdm import tqdm
from .perspective_api import PerspectiveAPI
import json
from torch.nn import CrossEntropyLoss

perplexity = load("perplexity", module_type="metric")


def compute_perplexity_explicit(
    predictions, batch_size: int = 16, add_start_token: bool = True, device='cuda', max_length=None
):


    model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    model = model.to(device)
    print(model.num_parameters())
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-xl')

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

def load_cola_model(cola_model="textattack/roberta-base-CoLA"):
    print(cola_model)
    tokenizer = RobertaTokenizerFast.from_pretrained(cola_model)
    model = RobertaForSequenceClassification.from_pretrained(cola_model)
    model.eval()
    return tokenizer, model


def load_internal_toxic(toxic_model="./checkpoints/BOLT_models/replaced_vocab_roberta_for_jigsaw"):
    tokenizer = RobertaTokenizerFast.from_pretrained(toxic_model)
    model = RobertaForSequenceClassification.from_pretrained(toxic_model)
    model.eval()
    return tokenizer, model


def load_external_sentiment(
    sentiment_model="VictorSanh/roberta-base-finetuned-yelp-polarity",
    saved_model="checkpoints/RobertaExtClsf",
):
    tokenizer = RobertaTokenizerFast.from_pretrained(sentiment_model, download=True)
    model = RobertaForSequenceClassification.from_pretrained(sentiment_model)
    model.eval()
    return tokenizer, model


def calc_cola(sentence_batch, cola_tokenizer, cola_model):
    cola_pred = []
    cola_logits = []
    with torch.no_grad():
        for gen_sent_text in sentence_batch:
            for cur_sent in gen_sent_text:
                inputs = cola_tokenizer(cur_sent, return_tensors="pt", padding=True).to(
                    cola_model.device
                )
                outputs = cola_model(**inputs)
                pred = outputs.logits.argmax(dim=1)
                logits = outputs.logits.softmax(dim=1)
                cola_logits.append(logits.cpu().numpy())
                cola_pred.append(pred.cpu().numpy())

                # just in case
                del inputs
                del outputs
    return cola_pred, cola_logits


def calc_hops(new_tokens, old_tokens):
    batch_hops_all = ((new_tokens != old_tokens) * 1.0).sum(dim=-1)
    return batch_hops_all


def calc_self_bleu(sentence_batch: List[str]) -> List[float]:
    scores = []
    for j in range(len(sentence_batch)):
        candidate = word_tokenize(
            sentence_batch[j].replace("<s> ", "").replace("</s>", "")
        )
        reference = [
            word_tokenize(sentence_batch[k].replace("<s> ", "").replace("</s>", ""))
            for k in range(len(sentence_batch))
            if k != j
        ]
        score = sentence_bleu(reference, candidate)
        scores.append(score)
    return scores


def get_unique_ngram(sentence_batch, n):
    total_n_grams = 0
    unique_n_grams = []
    for i in range(len(sentence_batch)):
        cur_group = sentence_batch[i]
        for j in range(len(cur_group)):
            candidate = word_tokenize(
                cur_group[j].replace("<s> ", "").replace("</s>", "")
            )
            n_grams = list(everygrams(candidate, min_len=n, max_len=n))
            for gram in n_grams:
                if gram not in unique_n_grams:
                    unique_n_grams.append(gram)
            total_n_grams += len(list(n_grams))
    return len(unique_n_grams) / total_n_grams


def compute_perplexity(sentence_batch):
    results = perplexity.compute(
        predictions=sentence_batch,
        model_id="gpt2-xl",
    )
    return results["perplexities"]


def compute_classifier_attribute(sentence_batch, ext_tokenizer, ext_clf):
    with torch.no_grad():
        inputs = ext_tokenizer(sentence_batch, return_tensors="pt", padding=True).to(
            ext_clf.device
        )
        outputs = ext_clf(**inputs)
        logits = outputs.logits.softmax(dim=-1).cpu().numpy()
        del outputs
    return logits


def compute_perspective_scores(sentences, save_dir, start_idx, rate_limit):
    api = PerspectiveAPI(rate_limit=rate_limit)
    print(len(sentences))
    api.request_bulk(
        sentences, output_file=f"{save_dir}/toxicity_scores_{start_idx}.json"
    )
    return


def clean_for_eval(sentences):
    cleaned = []
    for sent in sentences:
        if sent != "\n":
            cleaned.append(sent.replace("\n", ""))
    return cleaned
