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


def main(args):
    ### LOADING CONFIGS
    sampler_conf = load_sampler_conf(args)
    exp_conf = load_exp_conf(args)

    ### LOADING MODELS
    model = load_base_model(args.sampler, **exp_conf["base_model_args"]).cuda()
    discriminator = load_sentiment_discriminator().cuda()
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)

    ### COMBINING ALL CONF FOR SAVING
    total_conf = {**sampler_conf, **exp_conf}
    total_conf['sentiment'] = args.sentiment
    total_conf['sampler'] = args.sampler 

    # initialize the directory for storing data
    save_dir = f"{args.save_dir}/{args.sampler}"
    save_dir = initialize_metric_storing(total_conf, save_dir)

    ### initializing samplers
    if args.sampler == "bolt":
        sampler = BoltSampler(**total_conf)
    elif args.sampler == "langevin":
        sampler = LangevinSampler(model, **total_conf)

    ### INITIALIZE METADATA COLLECTION
    # TODO: do the above

    prompts = [line.strip() for line in open(total_conf["sentiment_prompts"], "r")]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", type=str, default="bolt")
    parser.add_argument("--sentiment", type=str, default="pos")
    parser.add_argument("--save_dir", type=str, default="results/sentiment")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--sampler_setup", type=str, default="default")
    args = parser.parse_args()

    main(args)
