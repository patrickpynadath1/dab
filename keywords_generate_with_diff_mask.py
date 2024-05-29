from transformers import (
    GPT2TokenizerFast,
    AdamW,
    get_scheduler,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    AutoModelForCausalLM,
    BeamSearchScorer,
)
from transformers import AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import sys
import time
import pickle

from dlp import LangevinSampler
from keywords_model_with_diff_mask import GPTPromptTuningWithbiasesModelLM

prompt_file = "./keywords/prompts_15.txt"

keywords_dict = {
    "computer" : ["router", "Linux", "keyboard", "server"],
    "legal" : ["plea", "subpoena", "transcript", "bankrupt"],
    "military" : ["torpedo", "headquarters", "infantry", "battlefield"],
    "politics" : ["court", "culture", "communism", "capitalism"],
    "religion" : ["Bible", "church", "priest", "saint"],
    "science" : ["microscope", "mass", "mineral", "scientist"],
    "space" : ["meteor", "planet", "satellite", "astronaut"],
}

seq_len = int(sys.argv[1])
topic = sys.argv[2]
save_dir = "keyword/mask_generations/"
output_file = f"{save_dir}" + "generations.txt.len" + str(seq_len)

class Config:
    num_train_epochs = 50
    weight_decay = 0.01
    learning_rate = 0.4
    lr_scheduler_type = "linear"
    num_warmup_steps = 5
    max_train_steps = num_train_epochs
    
    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 10
    init_from_vocab = True
args = Config()

batch_size = 20
sampler = LangevinSampler(.1, mh=False)
with open(prompt_file, "r") as f, open(output_file, "w") as g:
    prompts_list = [line.strip() for line in f] 
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Initialize GPT2LM with soft prompt
    model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
        "gpt2-large",
        n_tokens=args.n_prompt_tokens,
        initialize_from_vocab=args.init_from_vocab,
        use_full_prompt=False,
    )
    model.cuda()
    total_data = []
    total_successes = []
    for prompt in tqdm(prompts_list):
        keywords_word = [' '.join(keywords_dict[topic])] * batch_size
        prefixs = [prompt] * batch_size
        inputs = tokenizer(prefixs, return_tensors="pt")
        keywords = tokenizer([w for w in keywords_word], return_tensors="pt")['input_ids']
        inputs = inputs.to("cuda")
        keywords = keywords.to("cuda")
        model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1])
        model.eval()
        stored_sentence = [""] * batch_size
        success_idx = [-1] * batch_size
        start_time = time.time()
        probs = torch.ones(batch_size, seq_len + inputs.input_ids.shape[1] + 5, 50257).cuda() * .5
        diff_mask = torch.bernoulli(probs)
        def energy_fn(x):
            loss, output_ids = model.soft_forward(
                    **inputs, labels=inputs.input_ids, use_full_prompt=False, diff_mask=x, keywords=keywords
                ) 
            return loss, output_ids
        diff_mask.requires_grad_()
        prompt_data = {
            'loss_total': [],
        }
        for i in range(100):
            print("#################")
            diff_mask = diff_mask.detach()
            diff_mask, loss, output_ids, otheroutputs = sampler.step(diff_mask, energy_fn)
            print(keywords_word)
            sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print(sentences)
            prompt_data['loss_total'].append(loss.item())
            for idx in range(batch_size):
                if success_idx[idx] == -1:
                    if any([keyword in sentences[idx] for keyword in keywords_word[0].split(' ')]):
                        success_idx[idx] = i
                        stored_sentence[idx] = sentences[idx]
            if all([idx != -1 for idx in success_idx]):
                print("success")
                break
        del inputs 
        del diff_mask 
        del output_ids 
        end_time = time.time()
        print("success_idx: ", success_idx)
        print("stored_sentence: ", stored_sentence)
        print("time: ", end_time - start_time)
        g.write('\n'.join(stored_sentence) + "\n\n")
        g.flush()
        total_successes.append(success_idx)

    with open(f"{save_dir}data_{topic}.pkl", "wb") as h:
        pickle.dump(total_data, h)
    with open(f"{save_dir}best_losses_{topic}.pkl", "wb") as h: 
        pickle.dump(total_successes, h)
