import pdb
import numpy as np
from utils import (
    load_sampler_conf,
    load_exp_conf,
    initialize_metric_storing,
    initialize_best_loss,
    updating_best_loss,
)
from samplers import BoltSampler, LangevinSampler, get_sampler
import argparse
from models import load_base_model, load_sentiment_discriminator, load_tokenizer
import pickle
import torch
import math
import time
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
from transformers import logging

logging.set_verbosity_error()


def selective_biasing_loop(
    total_conf, dump_sampling_metrics=True, return_sampling_metrics=False
):
    ### LOADING MODELS
    model = load_base_model(
        total_conf["sampler"], mode="senti", **total_conf["base_model_args"]
    ).to(total_conf["device"])
    discriminator = load_sentiment_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)
    # initialize the directory for storing data
    save_dir = total_conf.get("prev_run_dir", None)
    if save_dir is None:
        save_dir = f"{total_conf['save_dir']}/selective_biasing_{total_conf['sentiment']}/{total_conf['sampler']}"
        save_dir = initialize_metric_storing(total_conf, save_dir)
    total_conf["prev_run_dir"] = save_dir
    print(save_dir)
    ### initializing samplers
    Sampler = get_sampler(total_conf["sampler"])
    print(total_conf["use_softmax_target"])
    sampler = Sampler(target_seq_len=20, max_attempts=20, **total_conf)
    times = []
    prompts = [
        line.strip() for line in open(total_conf["selective_biasing_prompts"], "r")
    ]
    output_file = open(f"{save_dir}/output.txt", "w")
    total_sentence_ids = []

    batch_size = 1
    pg_bar = tqdm(total=len(prompts) * batch_size * 20, desc="Sampling Steps remaining")
    # keep track of the total sentences
    total_sentence_ids = []
    total_zero_counts = []
    altered_means = []
    altered_max = []
    acceptance_mean = []
    acceptance_std = []
    lm_entropy = []
    constraint_entropy = []
    for prompt in prompts[: total_conf["num_prompts"]]:
        sampler.prompt_length = len(prompt)

        model.set_biases(
            batch_size=1,
            seq_len=total_conf["seq_len"],
            prompt_length=len(prompt),
            attribute="pos",
            device=total_conf["device"],
            disc_weight=1,
            use_scale_weights=True,
            use_bolt_weights=True,
        )
        model.eval()
        # first step is always just auto-reg loss
        sentences_to_write = []
        for _ in range(batch_size):
            prev_gen = tokenizer([prompt], return_tensors="pt")
            prev_gen = prev_gen.to(total_conf["device"]).input_ids
            num_attempts = 0
            cur_gen_sents = []
            sampler.stop_sampling = False
            while not sampler.stop_sampling and num_attempts < 20:
                new_gen = sampler.step(prev_gen, model, num_attempts)
                prev_gen = torch.concatenate([prev_gen, new_gen], dim=-1)
                cur_gen_sents.append(prev_gen.tolist())
                pg_bar.update(1)
                num_attempts += 1
            sentences = tokenizer.batch_decode(prev_gen, skip_special_tokens=True)

            sentences_to_write += sentences
        ### Freeing CUDA space
        output_file.write("\n".join(sentences_to_write) + "\n\n")
        output_file.flush()
    del model
    del discriminator
    pickle.dump(times, open(f"{save_dir}/times.pkl", "wb"))
    if dump_sampling_metrics:
        with open(f"{save_dir}/sampling_metrics.pkl", "wb") as f:
            sampling_metrics = sampler.get_sampling_metrics()
            pickle.dump(sampling_metrics, f)
    if return_sampling_metrics:
        return total_conf, total_sentence_ids, sampler.get_sampling_metrics()
    return total_conf, sentences_to_write
