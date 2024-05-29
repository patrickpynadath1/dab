from utils import (
    load_sampler_conf,
    load_exp_conf,
    initialize_metric_storing,
    initialize_best_loss,
    updating_best_keywords
)
from samplers import BoltSampler, LangevinSampler
import argparse
from models import (
    load_base_model, 
    load_toxicity_discriminator, 
    load_tokenizer
)


def main(args):
    ### LOADING CONFIGS
    sampler_conf = load_sampler_conf(args)
    exp_conf = load_exp_conf(args)

    ### LOADING MODELS
    model = load_base_model(args.sampler, 
                            use_senti=False, 
                            **exp_conf["base_model_args"]).cuda()
    discriminator = load_toxicity_discriminator().cuda()
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)

    ### COMBINING ALL CONF FOR SAVING
    total_conf = {**sampler_conf, **exp_conf}

    ### ADDITIONAL CONFIGS 
    total_conf['keyword'] = args.keyword
    total_conf['sampler'] = args.sampler
    total_conf['init_noise_rate'] = .7 

    # initialize the directory for storing data
    save_dir = f"{args.save_dir}/{args.sampler}"
    save_dir = initialize_metric_storing(total_conf, save_dir)

    ### INITIALIZING SAMPLERS
    if args.sampler == "bolt":
        sampler = BoltSampler(**total_conf)
    elif args.sampler == "dlp":
        sampler = LangevinSampler(**total_conf)

    ### INITIALIZE METADATA COLLECTION
    # TODO: do the above

    prompts = [line.strip() for line in open(total_conf["keyword_prompts"], "r")]
    output_file = open(f"{save_dir}/output.txt", "w")
    keywords_list = total_conf["keywords_dict"][args.keyword]
    keywords_string = " ".join(keywords_list)
    keywords_token = tokenizer([keywords_string] * total_conf['batch_size'], return_tensors="pt")['input_ids'].cuda()

    def energy_fn(x):
        loss, output_ids = model.soft_forward(
                **inputs, 
                labels=inputs.input_ids, 
                use_full_prompt=False, 
                diff_mask=x, 
                keywords=keywords_token
            ) 
        return loss, output_ids
    
    for prompt in prompts:
        prefixs = [prompt] * total_conf["batch_size"]
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to("cuda")
        cur_batch = sampler.initialize_batch(
            model=model,
            seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
            batch_size=total_conf["batch_size"]
        )
        model.eval()
        success_idx, stored_sentence = initialize_best_loss(total_conf["batch_size"], use_senti=False)
        for i in range(total_conf["num_steps"]):
            cur_batch, loss, output_ids, otheroutputs = sampler.step(
                x=cur_batch, 
                model=model, 
                energy_fn=energy_fn, 
                inputs=inputs, 
                keywords=keywords_token
            )
            sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            updating_best_keywords(cur_iter=i,
                                   batch_size=total_conf["batch_size"],
                                   sentences=sentences,
                                   success_idx=success_idx,
                                   keywords_word=keywords_list,
                                   stored_sentence_list=stored_sentence)
            if all([idx != -1 for idx in success_idx]):
                print("success")
                break

        ### Freeing CUDA space
        del inputs 
        del cur_batch
        del output_ids
        output_file.write("\n".join(stored_sentence) + "\n\n")
        output_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", type=str, default="bolt")
    parser.add_argument("--keyword", type=str, default="computer")
    parser.add_argument("--save_dir", type=str, default="results/keywords")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--sampler_setup", type=str, default="default")
    args = parser.parse_args()

    main(args)
