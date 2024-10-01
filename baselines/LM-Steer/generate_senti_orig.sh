#!/bin/bash
TRIAL=polarity-gpt2-large

source=positive
control=5
neg_weight=1

PYTHONPATH=. python experiments/training/generate_from_text_file.py \
    --eval_file prompts/sentiment_prompts_15.txt \
    --output_file logs/$TRIAL/predictions-${source}_${control}.txt \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values ${control} ${neg_weight} --top_p 0.9
