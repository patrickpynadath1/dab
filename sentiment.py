from utils import (
    load_sampler_conf,
    load_exp_conf,
    initialize_metric_storing,
    initialize_best_loss,
    updating_best_loss,
)
from samplers import BoltSampler, LangevinSampler
import argparse
from models import load_base_model, load_sentiment_discriminator, load_tokenizer
import pickle
import torch
import time
from tqdm import tqdm


def sentiment_exp_loop(
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
        save_dir = f"{total_conf['save_dir']}/sentiment_{total_conf['sentiment']}/{total_conf['sampler']}"
        save_dir = initialize_metric_storing(total_conf, save_dir)
    total_conf["prev_run_dir"] = save_dir
    ### initializing samplers
    # print(model.get_input_embeddings().weight.size())
    if total_conf["sampler"] == "bolt":
        sampler = BoltSampler(**total_conf)
        bias_dim = model.get_input_embeddings().weight.shape[0]
    elif total_conf["sampler"] == "dlp":
        sampler = LangevinSampler(**total_conf)
        bias_dim = model.get_input_embeddings().weight.shape[0]
    times = []
    prompts = [line.strip() for line in open(total_conf["sentiment_prompts"], "r")]
    output_file = open(f"{save_dir}/output.txt", "w")
    total_sentences = []

    def energy_fn_wrapper(x, inputs):
        prompt_bias = torch.zeros(x.size(0), inputs.input_ids.shape[1], bias_dim).to(
            total_conf["device"]
        )
        x_full = torch.concat([prompt_bias, x], dim=1)
        loss, output_ids, onehot_generates, gpt_logit, senti_losses = (
            model.soft_forward(
                **inputs,
                labels=inputs,
                use_full_prompt=False,
                biases=x_full,
                bias_rep_space="logit",
                weight=total_conf["weight_val"],
            )
        )
        return loss, output_ids, onehot_generates, gpt_logit, senti_losses

    all_output_ids = []
    for prompt in prompts[: total_conf["num_prompts"]]:
        if total_conf["seq_len"] > 20:
            batch_size = 10
            repeats = 2 
        else:
            batch_size = 20
            repeats = 1
        sentences_to_store = []
        for r in range(repeats):
            prefixs = [prompt] * batch_size 
            inputs = tokenizer(prefixs, return_tensors="pt")
            inputs = inputs.to(total_conf["device"])
            start = time.time()
            inputs, cur_batch = sampler.initialize_batch(
                model=model,
                seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
                sentiment=total_conf["sentiment"],
                batch_size=batch_size,
                prompt_length=inputs.input_ids.shape[1],
                inputs=inputs,
            )
            energy_fn = lambda x: energy_fn_wrapper(x, inputs)
            model.eval()
            minimum_loss, stored_sentence = initialize_best_loss(batch_size)
            cur_prompt_output_ids = []
            for i in range(total_conf["num_steps"]):
                cur_batch, loss, new_output_ids, otheroutputs = sampler.step(
                    x=cur_batch, model=model, energy_fn=energy_fn, inputs=inputs
                )
                losses_to_eval = otheroutputs[-1]
                output_ids = new_output_ids
                cur_prompt_output_ids.append(output_ids.tolist())
                sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                updating_best_loss(
                    batch_size,
                    losses_to_eval,
                    sentences,
                    minimum_loss,
                    stored_sentence,
                    iter=i
                )
            all_output_ids.append(cur_prompt_output_ids)
            end = time.time()
            # print(end - start)
            times.append(f"time: {end - start}")

            ### Freeing CUDA space
            del inputs
            del cur_batch
            del output_ids
            sentences_to_store += stored_sentence
        output_file.write("\n".join(sentences_to_store) + "\n\n")
        output_file.flush()
        total_sentences.extend(sentences_to_store)
    del model
    del discriminator
    print(save_dir)
    pickle.dump(times, open(f"{save_dir}/times.pkl", "wb"))
    if dump_sampling_metrics:
        with open(f"{save_dir}/sampling_metrics.pkl", "wb") as f:
            sampling_metrics = sampler.get_sampling_metrics()
            sampling_metrics["all_output_ids"] = all_output_ids
            pickle.dump(sampling_metrics, f)
    if return_sampling_metrics:
        return total_conf, total_sentences, sampler.get_sampling_metrics()
    return total_conf, total_sentences
