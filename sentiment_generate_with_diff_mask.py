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
from transformers import (
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
)
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import sys
from dlp import LangevinSampler
from model_with_diff_mask import GPTPromptTuningWithbiasesModelLM

prompt_file = "./sentiment/prompts_15.txt"
seq_len = int(sys.argv[1])
sentiment = sys.argv[2]  # pos or neg
save_dir = "sentiment/mask_generations/"
output_file = f"{save_dir}" + sys.argv[2] + ".txt.len" + str(seq_len)


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

batch_size = 20

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
    "gpt2-large",
    n_tokens=args.n_prompt_tokens,
    initialize_from_vocab=args.init_from_vocab,
    use_full_prompt=False,
)
model.cuda()
discriminator = AutoModelForSequenceClassification.from_pretrained(
    "./checkpoints/replaced_vocab_roberta_for_yelp_polarity"
)
discriminator.cuda()
model.init_discriminator(discriminator)


sampler = LangevinSampler(.2, mh=False)


with open(prompt_file, "r") as f, open(output_file, "w") as g:
    prompts = [line.strip() for line in f]
    total_data = []
    for prompt in tqdm(prompts):
        prefixs = [prompt] * batch_size
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to("cuda")
        model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1], sentiment)
        model.eval()
        minimum_loss = [100000] * batch_size
        stored_sentence = [""] * batch_size
        start_time = time.time()

        diff_mask = torch.ones(batch_size, seq_len + inputs.input_ids.shape[1] + 5, 50257).cuda()
        # diff_mask = nn.ParameterList(
        #     [
        #         nn.Parameter(torch.ones(batch_size, 50257))
        #         for i in range(seq_len + inputs.input_ids.shape[1] + 5)
        #     ]
        # ).cuda()

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
            if i % 1 == 0:
                # do the step here
                diff_mask = diff_mask.detach()
                diff_mask, loss, output_ids, senti_losses = sampler.step(diff_mask, energy_fn)
                prompt_data['loss_total'].append(loss.item())
                print("Decoding: ", loss)
                sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                print(sentences)
                print(time.time() - start_time)

            if i % 1 == 0:
                # print(f"loss: {loss}")
                prompt_data['senti_loss'].append(senti_losses.detach().cpu())
                for idx in range(batch_size):
                    # print(f"loss {idx}: senti loss: {senti_losses[idx]}")
                    if senti_losses[idx] < minimum_loss[idx]:
                        # print(f"update minimum loss{idx}")
                        minimum_loss[idx] = senti_losses[idx]
                        stored_sentence[idx] = sentences[idx]
        del inputs 
        del diff_mask 
        del output_ids 
        end_time = time.time()
        print("minimum loss: ", minimum_loss)
        print("stored sentence: ", stored_sentence)
        print("time: ", end_time - start_time)
        g.write("\n".join(stored_sentence) + "\n\n")
        g.flush()
        total_data.append(prompt_data)
    with open(f"{save_dir}data_{sentiment}.pkl", "wb") as h:
        pickle.dump(total_data, h)