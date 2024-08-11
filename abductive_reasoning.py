from utils import (
    load_sampler_conf,
    load_exp_conf,
    initialize_metric_storing,
    initialize_best_loss,
    updating_best_loss,
)
import torch
from samplers import BoltSampler, LangevinSampler
import argparse
from models import load_base_model, load_toxicity_discriminator, load_tokenizer
import pickle
import json
import pandas as pd


def abductive_reasoning_loop(total_conf):
    ### LOADING CONFIGS
    ### LOADING MODELS
    model = load_base_model(
        total_conf["sampler"], mode="reasoning", **total_conf["base_model_args"]
    ).to(total_conf["device"])
    discriminator = load_toxicity_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)

    ### ADDITIONAL CONFIGS
    # total_conf['keyword'] = args.keyword
    # total_conf['sampler'] = args.sampler
    # total_conf['init_noise_rate'] = .7

    # initialize the directory for storing data
    save_dir = f"{total_conf['save_dir']}/abductive_reasoning/{total_conf['sampler']}"
    total_conf["prev_run_dir"] = save_dir
    save_dir = initialize_metric_storing(total_conf, save_dir)

    ### INITIALIZING SAMPLERS
    if total_conf["sampler"] == "bolt":
        sampler = BoltSampler(**total_conf)
    elif total_conf["sampler"] == "dlp":
        sampler = LangevinSampler(**total_conf, is_kw=False)

    ### INITIALIZE METADATA COLLECTION
    total_sentences = []

    observations = pd.read_json(total_conf["abductive_reasoning_file"], lines=True)

    output_file = open(f"{save_dir}/output.txt", "w")

    def energy_fn_wrapper(x, inputs, ending_tokens):
        prompt_bias = torch.zeros(x.size(0), inputs.input_ids.shape[1], 50257).to(
            total_conf["device"]
        )
        x_full = torch.concat([prompt_bias, x], dim=1)
        loss, output_ids, onehot_generates, gpt_logit, ending_losses = (
            model.soft_forward(
                **inputs,
                labels=inputs.input_ids,
                use_full_prompt=False,
                biases=x_full,
                ending_tokens=ending_tokens,
            )
        )
        return loss, output_ids, onehot_generates, gpt_logit, ending_losses

    for row_idx, row in observations.iterrows():
        prefix = row["obs1"]
        # adding period to help with finishing the sentence
        ending = ". " + row["obs2"]
        prefixs = [prefix] * total_conf["batch_size"]
        inputs = tokenizer(prefixs, return_tensors="pt")
        ending_tokens = tokenizer(
            [ending] * total_conf["batch_size"], return_tensors="pt"
        )["input_ids"].to(total_conf["device"])
        inputs = inputs.to(total_conf["device"])
        inputs, cur_batch = sampler.initialize_batch(
            model=model,
            seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
            batch_size=total_conf["batch_size"],
            prompt_length=inputs.input_ids.shape[1],
            inputs=inputs,
            sentiment=None,
        )
        energy_fn = lambda x: energy_fn_wrapper(x, inputs, ending_tokens)
        model.eval()
        minimum_loss, stored_sentence = initialize_best_loss(total_conf["batch_size"])
        for i in range(total_conf["num_steps"]):
            cur_batch, loss, output_ids, otheroutputs = sampler.step(
                x=cur_batch,
                model=model,
                energy_fn=energy_fn,
                inputs=inputs,
                cur_iter=i,
                ending_tokens=ending_tokens,
            )
            losses_to_eval = otheroutputs[-1]
            sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            updating_best_loss(
                total_conf["batch_size"],
                losses_to_eval,
                sentences,
                minimum_loss,
                stored_sentence,
            )
        print(sentences)

        ### Freeing CUDA space
        del inputs
        del cur_batch
        del output_ids
        total_sentences.extend(stored_sentence)
        output_file.write("\n".join(stored_sentence) + "\n\n")
        output_file.flush()
    with open(f"{save_dir}/sampler_metrics.pkl", "wb") as f:
        pickle.dump(sampler.get_metrics_to_store(), f)
    return total_conf, total_sentences
