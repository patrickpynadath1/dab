# %%
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
import time
import sys
import pickle
from dlp import LangevinSampler
from model_with_diff_mask import GPTPromptTuningWithbiasesModelLM


prompt_file = "./detoxic/sampled_1k_prompt.txt"
seq_len = int(sys.argv[1])
save_dir = "toxic/mask_generations/"
output_file = f"{save_dir}" + "generations.txt.len" + str(seq_len)

class Config:
    num_train_epochs = 50
    weight_decay = 0.01
    learning_rate = 0.025
    lr_scheduler_type = "linear"
    num_warmup_steps = 5
    max_train_steps = num_train_epochs
    
    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 10
    init_from_vocab = True
args = Config()

batch_size = 15
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# Initialize GPT2LM with soft prompt
model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
    "gpt2-large",
    n_tokens=args.n_prompt_tokens,
    initialize_from_vocab=args.init_from_vocab,
    use_full_prompt=False,
)
model.cuda()
discriminator = AutoModelForSequenceClassification.from_pretrained("./checkpoints/replaced_vocab_roberta_for_jigsaw/")
discriminator.cuda()
model.init_discriminator(discriminator)


sampler = LangevinSampler(.1, mh=False)

with open(prompt_file, "r") as f, open(output_file, "w") as g:
    prompts = [line.strip() for line in f]

    total_data = []
    for prompt in tqdm(prompts):
        prefixs = [prompt] * batch_size
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to("cuda")
        model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1], 'non_toxic', 0.7)
        model.eval()
        minimun_loss = [100000] * batch_size
        stored_sentence = [""] * batch_size
        start_time = time.time()
        probs = torch.ones(batch_size, seq_len + inputs.input_ids.shape[1] + 5, 50257).cuda() * .5
        diff_mask = torch.bernoulli(probs)
        def energy_fn(x):
            loss, output_ids, gpt_logit, senti_losses = model.soft_forward(
                    **inputs, labels=inputs.input_ids, use_full_prompt=False, diff_mask=x
                ) 
            return loss, output_ids, gpt_logit, senti_losses
        diff_mask.requires_grad_()
        prompt_data = {
            'loss_total': [],
            'senti_loss': [],
        }
        for i in range(8):
            if all([loss < 0.0003 for loss in minimun_loss]):
                break
            if i % 1 == 0:
                diff_mask = diff_mask.detach()
                # for j in range(10):
                diff_mask, loss, output_ids, senti_losses = sampler.step(diff_mask, energy_fn)
                prompt_data['loss_total'].append(loss.item())
                print("Decoding: ", loss)
                sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                print(sentences)
            if i % 1 == 0:
                prompt_data['senti_loss'].append(senti_losses.detach().cpu())
                for idx in range(batch_size):
                    # print(f"loss {idx}: senti loss: {senti_losses[idx]}")
                    if senti_losses[idx] < minimun_loss[idx]:
                        # print(f"update minimun loss{idx}")
                        minimun_loss[idx] = senti_losses[idx]
                        stored_sentence[idx] = sentences[idx]
            
        del inputs 
        del diff_mask 
        del output_ids 
        end_time = time.time()
        print("minimun loss: ", minimun_loss)
        print("stored sentence: ", stored_sentence)
        print("time: ", end_time - start_time)
        g.write('\n'.join(stored_sentence) + "\n\n")
        g.flush()
        total_data.append(prompt_data)
    with open(f"{save_dir}data.pkl", "wb") as h:
        pickle.dump(total_data, h)