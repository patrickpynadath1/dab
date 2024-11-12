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
import torch
from torch.optim import SGD
from tqdm import tqdm
import time
import sys
import pandas as pd
import pickle
from model_with_biases import GPTPromptTuningWithbiasesModelLM

prompt_file = "./sentiment/prompts_15.txt"
seq_len = int(sys.argv[1])
sentiment = sys.argv[2]  # pos or neg
output_file = "./sentiment/" + sys.argv[2] + ".txt.len" + str(seq_len)


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

if seq_len > 20:
    batch_size = 10
    repeats = 2
else:
    batch_size = 20
    repeats = 1
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")

model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
    "gpt2-large",
    n_tokens=args.n_prompt_tokens,
    initialize_from_vocab=args.init_from_vocab,
    use_full_prompt=False,
)
model.cuda()
model.get_input_embeddings().weight.register_hook(lambda grad: print("base embeds"))
discriminator = AutoModelForSequenceClassification.from_pretrained(
    "../../checkpoints/BOLT_models/replaced_vocab_roberta_for_yelp_polarity"
)
discriminator.cuda()
model.init_discriminator(discriminator)


def profiler_to_dataframe(profiler_stats):
    records = []
    for evt in profiler_stats:
        record = {
            "name": evt.key,
            "cpu_time_total": evt.cpu_time_total,
            "cuda_time_total": evt.cuda_time_total,
            "cpu_memory_usage": evt.cpu_memory_usage,
            "self_cuda_time_total": evt.self_cuda_time_total,
            "self_cpu_time_total": evt.self_cpu_time_total,
            "count": evt.count,
            # Add more fields as needed
        }
        records.append(record)
    return pd.DataFrame(records)


with open(prompt_file, "r") as f, open(output_file, "w") as g:
    prompts = [line.strip() for line in f]

    all_output_ids = []
    true_delta_losses = []
    estimated_delta_losses = []
    for prompt in tqdm(prompts):
        prompt_true_delta_loss = []
        prompt_estimated_delta_loss = []
        sentence_to_store = []

        for i in range(repeats):
            prefixs = [prompt] * batch_size
            inputs = tokenizer(prefixs, return_tensors="pt")
            inputs = inputs.to("cuda")
            model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1], sentiment)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "biases" in n or "trainable_weights" in n
                    ],
                    "weight_decay": args.weight_decay,
                }
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
            model.eval()
            minimum_loss = [100000] * batch_size
            stored_sentence = [""] * batch_size
            start_time = time.time()
            cur_prompt_output_ids = []

            prompt_est_delta = []
            prompt_actual_delta = []
            for iter in range(50):
                # making a copy of biases to perform interp experiments
                old_biases = [
                    model.biases[i].clone().detach() for i in range(len(model.biases))
                ]
                loss, new_output_ids, gpt_logit, senti_losses, one_hot = (
                    model.soft_forward(
                        **inputs, labels=inputs.input_ids, use_full_prompt=False
                    )
                )

                # print(one_hot.grad_fn)
                # print("Decoding: ", loss)
                output_ids = new_output_ids
                cur_prompt_output_ids.append(output_ids.tolist())
                sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                # print(sentences)
                # print(time.time()-start_time)

                start = time.time()

                loss.backward()
                optimizer.step()
                print("Time for one step: ", time.time() - start)
                noise = [
                    torch.normal(
                        mean=0.01,
                        std=0.01,
                        size=model.biases[0].shape,
                        device="cuda",
                        requires_grad=False,
                    )
                    for _ in range(len(model.biases))
                ]

                # computing the raw gradient
                for i in range(len(model.biases)):
                    model.biases[i].data = model.biases[i].data + noise[i]
                print(iter)
                # computing the est loss for each interpolation
                if iter % 1 == 0:
                    # print(f"iter: {iter}; loss: {loss}")
                    for idx in range(batch_size):
                        # print(f"loss {idx}: senti loss: {senti_losses[idx]}")
                        if senti_losses[idx] < minimum_loss[idx]:
                            # print(f"update minimum loss{idx}")
                            minimum_loss[idx] = senti_losses[idx]
                            stored_sentence[idx] = sentences[idx]
                optimizer.zero_grad()
        true_delta_losses.append(prompt_actual_delta)
        estimated_delta_losses.append(prompt_est_delta)
        all_output_ids.append(cur_prompt_output_ids)
        sentence_to_store += stored_sentence
        # print("minimum loss: ", minimum_loss)
        # print("stored sentence: ", stored_sentence)
        g.write("\n".join(sentence_to_store) + "\n\n")
        g.flush()
