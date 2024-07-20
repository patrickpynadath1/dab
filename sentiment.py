from utils import (
    load_sampler_conf,
    load_exp_conf,
    initialize_metric_storing,
    initialize_best_loss,
    updating_best_loss,
)
from samplers import BoltSampler, LangevinSampler
import argparse
from models import (
    load_base_model, 
    load_sentiment_discriminator, 
    load_tokenizer
)
import pickle 
import torch
import time 

def sentiment_exp_loop(total_conf):


    ### LOADING MODELS
    model = load_base_model(total_conf['sampler'], mode='senti', **total_conf["base_model_args"]).to(total_conf["device"])
    discriminator = load_sentiment_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)
    # initialize the directory for storing data
    save_dir = f"{total_conf['save_dir']}/sentiment_{total_conf['sentiment']}/{total_conf['sampler']}"
    save_dir = initialize_metric_storing(total_conf, save_dir)
    total_conf['prev_run_dir'] = save_dir
    ### initializing samplers
    if total_conf['sampler'] == "bolt":
        sampler = BoltSampler(**total_conf)
    elif total_conf['sampler'] == "dlp":
        sampler = LangevinSampler(**total_conf)
    times = []
    prompts = [line.strip() for line in open(total_conf["sentiment_prompts"], "r")]
    output_file = open(f"{save_dir}/output.txt", "w")
    total_sentences = []
    def energy_fn_wrapper(x, inputs):
        prompt_bias = torch.zeros(x.size(0), inputs.input_ids.shape[1], 50257).to(total_conf["device"])
        x_full = torch.concat([prompt_bias, x], dim=1)
        loss, output_ids, onehot_generates, gpt_logit, senti_losses = model.soft_forward(
            **inputs, labels=inputs, use_full_prompt=False, biases=x_full
        )
        return loss, output_ids, onehot_generates, gpt_logit, senti_losses
    
    for prompt in prompts:
        prefixs = [prompt] * total_conf["batch_size"]
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to(total_conf["device"])
        start = time.time()
        inputs, cur_batch = sampler.initialize_batch(
            model=model,
            seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
            sentiment=total_conf["sentiment"],
            batch_size=total_conf["batch_size"], 
            prompt_length=inputs.input_ids.shape[1],
            inputs=inputs
        )
        energy_fn = lambda x : energy_fn_wrapper(x, inputs)
        model.eval()
        minimum_loss, stored_sentence = initialize_best_loss(total_conf["batch_size"])
        for i in range(total_conf["num_steps"]):
            cur_batch, loss, output_ids, otheroutputs = sampler.step(
                x=cur_batch, model=model, energy_fn=energy_fn, inputs=inputs
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
        end = time.time()
        times.append(end - start)

        ### Freeing CUDA space
        del inputs 
        del cur_batch
        del output_ids
        output_file.write("\n".join(stored_sentence) + "\n\n")
        output_file.flush()
        total_sentences.extend(stored_sentence)
    del model 
    del discriminator
    pickle.dump(times, open(f"{save_dir}/times.pkl", "wb"))
    return total_conf, total_sentences