from utils import (
    load_sampler_conf,
    load_exp_conf,
    initialize_metric_storing,
    initialize_best_loss,
    updating_best_keywords, 
    updating_best_loss
)
import torch
from samplers import BoltSampler, LangevinSampler
import argparse
from models import (
    load_base_model, 
    load_toxicity_discriminator, 
    load_tokenizer
)
import pickle


def keyword_loss(logits, target_kw_idx, kw_token): 
    return - logits.softmax(dim=-1)[torch.arange(logits.size(0)), target_kw_idx, kw_token].sum()



def keywords_loop(total_conf, dump_sampling_metrics=True, return_sampling_metrics=False):
    ### LOADING CONFIGS
    ### LOADING MODELS
    if total_conf['device'] == 'cuda':
        torch.cuda.set_device(total_conf["cuda_id"])
    model = load_base_model(total_conf['sampler'], 
                            mode='keywords', 
                            **total_conf["base_model_args"]).to(total_conf["device"])
    discriminator = load_toxicity_discriminator().to(total_conf["device"])
    tokenizer = load_tokenizer()
    model.init_discriminator(discriminator)

    ### ADDITIONAL CONFIGS 
    # total_conf['keyword'] = args.keyword
    # total_conf['sampler'] = args.sampler
    # total_conf['init_noise_rate'] = .7 

    # initialize the directory for storing data
    save_dir = f"{total_conf['save_dir']}/keywords/{total_conf['sampler']}"
    total_conf['prev_run_dir'] = save_dir
    save_dir = initialize_metric_storing(total_conf, save_dir)


    ### INITIALIZING SAMPLERS
    if total_conf['sampler'] == "bolt":
        sampler = BoltSampler(**total_conf, is_kw = True)
        bias_dim = model.get_input_embeddings().weight.shape[0]

    elif total_conf['sampler'] == "dlp":
        sampler = LangevinSampler(**total_conf, is_kw=True)
        if total_conf['bias_rep_space'] == "logit":
            bias_dim = model.get_input_embeddings().weight.shape[0]
        else:
            bias_dim = model.get_input_embeddings().weight.shape[1]
    ### INITIALIZE METADATA COLLECTION
    # TODO: do the above
    total_sentences = []

    prompts = [line.strip() for line in open(total_conf["keyword_prompts"], "r")]
    output_file = open(f"{save_dir}/output.txt", "w")
    keywords_list = total_conf["keywords_dict"][total_conf['keyword']]
    keywords_string = " ".join(keywords_list)
    keywords_token = tokenizer([keywords_string] * total_conf['batch_size'], return_tensors="pt")['input_ids'].to(total_conf['device'])
     
    def energy_fn_wrapper(x, inputs):
        prompt_bias = torch.zeros(x.size(0), inputs.input_ids.shape[1], bias_dim).to(total_conf["device"])
        x_full = torch.concat([prompt_bias, x], dim=1)
        loss, output_ids, onehot_generates, gpt_logit, kw_losses = model.soft_forward(
            **inputs, 
            labels=inputs.input_ids, 
            use_full_prompt=False, 
            biases=x_full,
            keywords=keywords_token, 
            use_cnn_batchloss=total_conf['use_cnn_batchloss'],
            bias_rep_space = total_conf['bias_rep_space'], 
            weight=total_conf['weight_val']
        )
        return loss, output_ids, onehot_generates, gpt_logit, kw_losses
    

    for prompt in prompts:
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
        energy_fn = lambda x : energy_fn_wrapper(x, inputs)
        model.eval()
        if total_conf['sampler'] == 'bolt':  
            stored_res, stored_sentence = initialize_best_loss(total_conf["batch_size"], use_senti=False)
        else: 
            stored_res, stored_sentence = initialize_best_loss(total_conf["batch_size"], use_senti=True)
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
            if total_conf['sampler'] == "bolt":
                updating_best_keywords(cur_iter=i,
                                    batch_size=total_conf["batch_size"],
                                    sentences=sentences,
                                    success_idx=stored_res,
                                    keywords_word=keywords_list,
                                    stored_sentence_list=stored_sentence)
            else:
                print(otheroutputs[-1]) 
                updating_best_loss(batch_size=total_conf["batch_size"],
                                loss=otheroutputs[-1],
                                sentences=sentences,
                                min_loss_list=stored_res,
                                stored_sentence_list=stored_sentence)
            if all([idx != -1 for idx in stored_res]) and total_conf["early_stop"]:
                # print("success")
                break
        print(sentences)
        if stored_sentence == [""] * total_conf["batch_size"]:
            stored_sentence = sentences
        ### Freeing CUDA space
        del inputs 
        del cur_batch
        del output_ids
        total_sentences.extend(stored_sentence)
        output_file.write("\n".join(stored_sentence) + "\n\n")
        output_file.flush()
    if dump_sampling_metrics:
        with open(f"{save_dir}/sampling_metrics.pkl", "wb") as f: 
            pickle.dump(sampler.get_sampling_metrics(), f)
    if return_sampling_metrics:
        return total_conf, total_sentences, sampler.get_sampling_metrics()
    return total_conf, total_sentences 

