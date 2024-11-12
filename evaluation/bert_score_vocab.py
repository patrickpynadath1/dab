# Code for comparing ref text from GPT 4 to generations using BERT score
# idea: the bert embed vectors shopuld capture context and meaning of text
# what we are actually trying to do is get the model to talk about the same thing as the ref text
# if the bert score is high, then the model is talking about the same thing as the ref text
import torch
import numpy as np
from evaluate import load

bertscore = load("bertscore")

topic_kw_dct = {
    "computer": ["router", "Linux", "keyboard", "server"],
    "legal": ["plea", "subpoena", "transcript", "bankrupt"],
    "military": ["torpedo", "headquarters", "infantry", "battlefield"],
    "politics": ["court", "culture", "communism", "capitalism"],
    "religion": ["Bible", "church", "priest", "saint"],
    "science": ["microscope", "mass", "mineral", "scientist"],
    "space": ["meteor", "planet", "satellite", "astronaut"],
}


def compute_bertscore_prompt(prompt_gens, ref_gens, mode="mean", metric="f1"):
    num_refs = len(ref_gens)
    res = {"precision": [], "recall": [], "f1": []}
    for cur_prompt in prompt_gens:
        single_prompt = [cur_prompt] * num_refs
        results = bertscore.compute(
            predictions=single_prompt, references=ref_gens, lang="en"
        )
        score_idx = np.argmax(results[metric])
        for k in res.keys():
            if mode == "maximal":
                res[k].append(np.mean(results[k]))
            elif mode == "mean":
                res[k].append(results[k][score_idx])
    return res


def load_ref_texts(base_dir, cur_prompt_idx, cur_prompt_text, topic):
    refs = []
    for kw in topic_kw_dct[topic]:
        ref_gens = open(
            f"{base_dir}/{topic}/{kw}_{cur_prompt_idx}.txt", "r"
        ).readlines()
        for r in ref_gens:
            refs.append(r)
    return refs


def group_gens_by_prompt(gens, prompts):
    grouped_gens = []
    for p in prompts:
        cur_group = []
        for g in gens:
            if p in g:
                cur_group.append(g)
        grouped_gens.append(cur_group)
    return grouped_gens


def bertscore_loop(gens, prompts, base_dir="keywords_ref_text", topic="computer"):
    grouped_gens = group_gens_by_prompt(gens, prompts)
    total_res = {"precision": [], "recall": [], "f1": []}
    for p_idx, prompt in enumerate(prompts):
        cur_gens = grouped_gens[p_idx]
        ref_gens = load_ref_texts(base_dir, p_idx, prompt, topic=topic)
        res = compute_bertscore_prompt(cur_gens, [[ref_gens]])
        for k in total_res.keys():
            total_res[k] += res[k]
    return total_res
