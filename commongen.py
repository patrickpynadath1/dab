from utils import (
    load_sampler_conf,
    load_exp_conf,
    initialize_metric_storing,
    initialize_best_loss,
    updating_best_keywords
)
import torch
from samplers import BoltSampler, LangevinSampler
import argparse
from models import (
    load_base_model, 
    load_toxicity_discriminator, 
    load_tokenizer
)
import pandas as pd
import pickle


def keyword_loss(logits, target_kw_idx, kw_token): 
    return - logits.softmax(dim=-1)[torch.arange(logits.size(0)), target_kw_idx, kw_token].sum()



def commongen_loop(total_conf):
    ### LOADING CONFIGS
    ### LOADING MODELS
    model = load_base_model(total_conf['sampler'], 
                            use_senti=False, 
                            **total_conf["base_model_args"]).to(total_conf["device"])
    discriminator = load_toxicity_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)

    ### ADDITIONAL CONFIGS 
    # total_conf['keyword'] = args.keyword
    # total_conf['sampler'] = args.sampler
    # total_conf['init_noise_rate'] = .7 

    # initialize the directory for storing data
    save_dir = f"{total_conf['save_dir']}/commongen/{total_conf['sampler']}"
    total_conf['prev_run_dir'] = save_dir
    save_dir = initialize_metric_storing(total_conf, save_dir)


    ### INITIALIZING SAMPLERS
    if total_conf['sampler'] == "bolt":
        sampler = BoltSampler(**total_conf)
    elif total_conf['sampler'] == "dlp":
        sampler = LangevinSampler(**total_conf, is_kw=True)

    ### INITIALIZE METADATA COLLECTION
    # TODO: do the above
    total_sentences = []

    concept_sets = pd.read_json(total_conf["commongen_file"], lines=True)
    print(concept_sets)
    output_file = open(f"{save_dir}/output.txt", "w")
    # keywords_list = total_conf["keywords_dict"][total_conf['keyword']]
    # keywords_string = " ".join(keywords_list)
    # keywords_token = tokenizer([keywords_string] * total_conf['batch_size'], return_tensors="pt")['input_ids'].to(total_conf['device'])
     
    def energy_fn_wrapper(x, inputs, keywords_token):
        prompt_bias = torch.zeros(x.size(0), inputs.input_ids.shape[1], 50257).to(total_conf["device"])
        x_full = torch.concat([prompt_bias, x], dim=1)
        loss, output_ids, onehot_generates, gpt_logit = model.soft_forward(
            **inputs, 
            labels=inputs.input_ids, 
            use_full_prompt=False, 
            biases=x_full,
            keywords=keywords_token, 
            use_cnn_batchloss=total_conf['use_cnn_batchloss']
        )
        return loss, output_ids, onehot_generates, gpt_logit
    

    for concept_idx, concept in concept_sets.iterrows():
        keywords_list = concept["concept_set"].split("#")
        keywords_token = tokenizer(keywords_list, return_tensors="pt")['input_ids'].to(total_conf['device'])
        prompt = keywords_list.join(" ") + " = "
        prefixs = [prompt] * total_conf["batch_size"]
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to(total_conf["device"])
        inputs, cur_batch = sampler.initialize_batch(
            model=model,
            seq_length=total_conf["seq_len"] + inputs.input_ids.shape[1],
            batch_size=total_conf["batch_size"], 
            prompt_length=inputs.input_ids.shape[1], 
            inputs=inputs,
            sentiment=None,
            keyword_tokens=keywords_token
        )
        energy_fn = lambda x : energy_fn_wrapper(x, inputs, keywords_token)
        model.eval()
        success_idx, stored_sentence = initialize_best_loss(total_conf["batch_size"], use_senti=False)
        for i in range(total_conf["num_steps"]):
            cur_batch, loss, output_ids, otheroutputs = sampler.step(
                x=cur_batch, 
                model=model, 
                energy_fn=energy_fn, 
                inputs=inputs, 
                kw_tokens=keywords_token, 
                keywords=keywords_token,
                cur_iter=i
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



