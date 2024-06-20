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
    load_toxicity_discriminator, 
    load_tokenizer
)


def detoxify_loop(total_conf):


    ### LOADING MODELS
    model = load_base_model(total_conf['sampler'], **total_conf["base_model_args"]).to(total_conf["device"])
    discriminator = load_toxicity_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)

    # ### ADDITIONAL CONFIGS 
    # total_conf['sentiment'] = "non_toxic"
    # total_conf['sampler'] = args.sampler
    # total_conf['init_noise_rate'] = .7 

    if total_conf['prev_run_dir'] is None: 
        save_dir = f"{total_conf['save_dir']}/{total_conf['sampler']}"
        total_conf['prev_run_dir'] = save_dir
    else: 
        save_dir = total_conf['prev_run_dir']
    save_dir = initialize_metric_storing(total_conf, save_dir)

    ### INITIALIZING SAMPLERS
    if total_conf['sampler'] == "bolt":
        sampler = BoltSampler(**total_conf)
    elif total_conf['sampler'] == "dlp":
        sampler = LangevinSampler(**total_conf)

    ### INITIALIZE METADATA COLLECTION
    # TODO: do the above

    prompts = [line.strip() for line in open(total_conf["detoxic_prompts"], "r")]
    output_file = open(f"{save_dir}/output.txt", "w")

    def energy_fn(x):
        loss, output_ids, gpt_logit, senti_losses = model.soft_forward(
            **inputs, labels=inputs.input_ids, use_full_prompt=False, diff_mask=x
        )
        return loss, output_ids, gpt_logit, senti_losses

    for prompt in prompts:
        prefixs = [prompt] * total_conf["batch_size"]
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to("cuda")
        cur_batch = sampler.initialize_batch(
            model=model,
            seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
            sentiment=total_conf["sentiment"],
            batch_size=total_conf["batch_size"]
        )
        model.eval()
        minimum_loss, stored_sentence = initialize_best_loss(total_conf["batch_size"])
        for i in range(total_conf["num_steps"]):
            if all([loss < 0.0003 for loss in minimum_loss]):
                break   
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
