from transformers import (
    GPT2TokenizerFast,
    AdamW,
)
from transformers import AutoModelForSequenceClassification 
from collections import deque
from model_with_biases import GPTPromptTuningWithbiasesModelLM
import torch 

seq_len = 20
sentiment = 'pos'


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
discriminator = AutoModelForSequenceClassification.from_pretrained("../../checkpoints/BOLT_models/replaced_vocab_roberta_for_yelp_polarity")
discriminator.cuda()
model.init_discriminator(discriminator)
prompt_file = "./sentiment/prompts_15.txt"

with open(prompt_file, "r") as f:
    prompts = [line.strip() for line in f]

prompt = prompts[0]
prefixs = [prompt] * batch_size
inputs = tokenizer(prefixs, return_tensors="pt")
inputs = inputs.to("cuda")
model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1], sentiment)
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if "biases" in n or "trainable_weights" in n],
        "weight_decay": args.weight_decay,
    }
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
model.train()
traced_model = torch.jit.trace(model.soft_forward, (inputs, labels=inputs.input_ids, use_full_prompt=False))
# Analyze the computational cost of computing gradients w.r.t. biases and trainable weights
print("runnig the analysis")
# torch.autograd.grad(loss, model.biases, retain_graph=True)  




