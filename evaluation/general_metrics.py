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
import json 
import pandas as pd
perplexity = load("perplexity", module_type="metric")

def convert_to_json(sentences, prompts, outfile): 
    res = []
    for p_idx, prompt in enumerate(prompts):
        prompt_to_use = prompt.replace("\n", "")
        cur_prompt_dct = {'prompt': {"text": prompt_to_use}, 'generations': []}
        for s_idx, sentence in enumerate(sentences):
            if prompt_to_use in sentence: 
                sentence_to_use = sentence.replace("\n", "").replace(prompt_to_use, "")
                cur_prompt_dct['generations'].append({"text": sentence_to_use})
        res.append(cur_prompt_dct)

    with open(outfile, 'w') as f: 
        for d in res: 
            f.write(json.dumps(d) + "\n")
    print("Reformatted text to json")
            


def conditional_perplexity(loc_to_read, 
                           model, 
                           tokenizer, 
                           device='cuda'):
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    generations_df = pd.read_json(open(loc_to_read, 'r'), lines=True)

    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating PPL'):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(row.prompt['text'], return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = [gen['text'] for gen in row['generations']]
        # for gen_ids in generations:
        for gen in generations:

            # full_input_ids = torch.LongTensor([row.prompt['tokens'] + gen_ids]).to(device)
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            # print(f'{prompt}{gen}')
            # print(full_input_ids)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])

            ppl = np.exp(loss.item())
            # print(ppl)
            # input()
            if ppl < 100:   # for sanity
                goodperplexities.append(ppl)
                g += 1

            if ppl < 1e4:
                perplexities.append(ppl)
            total_nll += (full_loss - prompt_loss).item()
            total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
    return perplexities

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
        sentences, output_file=f"{save_dir}/toxicity_scores_{start_idx}.jsonl"
    )
    return


def clean_for_eval(sentences):
    cleaned = []
    for sent in sentences:
        if sent != "\n":
            cleaned.append(sent.replace("\n", ""))
    return cleaned
