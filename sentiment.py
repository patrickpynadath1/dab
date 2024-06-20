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
import torch

def sentiment_exp_loop(total_conf):


    ### LOADING MODELS
    model = load_base_model(total_conf['sampler'], **total_conf["base_model_args"]).to(total_conf["device"])
    discriminator = load_sentiment_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)
    # initialize the directory for storing data
    if total_conf['prev_run_dir'] is None: 
        save_dir = f"{total_conf['save_dir']}/sentiment_{total_conf['sentiment']}/{total_conf['sampler']}"
        total_conf['prev_run_dir'] = save_dir
    else: 
        save_dir = total_conf['prev_run_dir']
    save_dir = initialize_metric_storing(total_conf, save_dir)

    ### initializing samplers
    if total_conf['sampler'] == "bolt":
        sampler = BoltSampler(**total_conf)
    elif total_conf['sampler'] == "dlp":
        sampler = LangevinSampler(**total_conf)

    prompts = [line.strip() for line in open(total_conf["sentiment_prompts"], "r")]
    output_file = open(f"{save_dir}/output.txt", "w")
    total_sentences = []
    def energy_fn_wrapper(x, inputs):
        prompt_bias = torch.zeros(x.size(0), inputs.input_ids.shape[1], 50257)
        x_full = torch.concat([prompt_bias, x], dim=1)
        loss, output_ids, onehot_generates, gpt_logit, senti_losses = model.soft_forward(
            **inputs, labels=inputs.input_ids, use_full_prompt=False, biases=x_full
        )
        return loss, output_ids, onehot_generates, gpt_logit, senti_losses
    for prompt in prompts:
        prefixs = [prompt] * total_conf["batch_size"]
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to(total_conf["device"])
        energy_fn = lambda x : energy_fn_wrapper(x, inputs)
        cur_batch = sampler.initialize_batch(
            model=model,
            seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
            sentiment=total_conf["sentiment"],
            batch_size=total_conf["batch_size"], 
            prompt_length=inputs.input_ids.shape[1],
        )
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

        ### Freeing CUDA space
        del inputs 
        del cur_batch
        del output_ids
        output_file.write("\n".join(stored_sentence) + "\n\n")
        output_file.flush()
        total_sentences.append(stored_sentence)
    return total_conf, total_sentences