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


def detoxify_loop(total_conf):

    ### LOADING MODELS
    model = load_base_model(
        total_conf["sampler"], mode="senti", **total_conf["base_model_args"]
    ).to(total_conf["device"])
    discriminator = load_toxicity_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator, is_detoxic=True)

    # ### ADDITIONAL CONFIGS
    # total_conf['sentiment'] = "non_toxic"
    # total_conf['sampler'] = args.sampler
    # total_conf['init_noise_rate'] = .7

    save_dir = f"{total_conf['save_dir']}/detoxify/{total_conf['sampler']}"
    save_dir = initialize_metric_storing(total_conf, save_dir)
    ### INITIALIZING SAMPLERS
    if total_conf["sampler"] == "bolt":
        sampler = BoltSampler(**total_conf)
        bias_dim = model.get_input_embeddings().weight.shape[0]
    elif total_conf["sampler"] == "dlp":
        sampler = LangevinSampler(**total_conf)
        bias_dim = model.get_input_embeddings().weight.shape[0]

    ### INITIALIZE METADATA COLLECTION
    # TODO: do the above

    prompts = [line.strip() for line in open(total_conf["detoxic_prompts"], "r")]
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
                labels=inputs.input_ids,
                use_full_prompt=False,
                biases=x_full,
                bias_rep_space="logit",
                weight=total_conf["weight_val"],
            )
        )
        return loss, output_ids, onehot_generates, gpt_logit, senti_losses

    for prompt in prompts:
        prefixs = [prompt] * total_conf["batch_size"]
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to(total_conf["device"])
        inputs, cur_batch = sampler.initialize_batch(
            model=model,
            seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
            sentiment=total_conf["sentiment"],
            batch_size=total_conf["batch_size"],
            prompt_length=inputs.input_ids.shape[1],
            inputs=inputs,
        )
        energy_fn = lambda x: energy_fn_wrapper(x, inputs)
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
        del loss
        torch.cuda.empty_cache()
        total_sentences.extend(stored_sentence)
        output_file.write("\n".join(stored_sentence))
        output_file.write("\n\n")
        output_file.flush()
    return total_conf, total_sentences
