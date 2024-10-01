#!/bin/bash 

TRIAL=lm-steer-polarity-neut-to-neg
mkdir -p logs/$TRIAL

source=negative
control=-5
# PYTHONPATH=. python experiments/training/train.py \
#     --dataset_name sentiment-sst5 \
#     --ckpt_name logs/$TRIAL/checkpoint.pt \
#     --model gpt2-large --cuda \
#     --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
#     --batch_size 32 --max_length 256 \
#     --n_steps 1000 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3;


PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/sentiment_prompts-10k/neutral_prompts.jsonl \
    --output_file logs/$TRIAL/output.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values ${control} 1 --top_p 0.9


python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/output.jsonl \
    --metrics sentiment \
    --output_file result_stats_${source}_${control}.txt

python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/output.jsonl \
    --metrics ppl-big \
    --output_file result_stats_${source}_${control}.txt.ppl

TRIAL=lm-steer-polarity-pos-to-neg
mkdir -p logs/$TRIAL

source=negative
control=-5


PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/sentiment_prompts-10k/neutral_prompts.jsonl \
    --output_file logs/$TRIAL/output.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values ${control} 1 --top_p 0.9


python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/output.jsonl \
    --metrics sentiment \
    --output_file result_stats_${source}_${control}.txt

python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/output.jsonl \
    --metrics ppl-big \
    --output_file result_stats_${source}_${control}.txt.ppl

echo "Sentiment control results:"


TRIAL=lm-steer-polarity-open-ended-neg
mkdir -p logs/$TRIAL

source=negative
control=-5


PYTHONPATH=. python experiments/training/generate.py \
    --eval_file prompts/sentiment_prompts_15.jsonl \
    --output_file logs/$TRIAL/output.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values ${control} 1 --top_p 0.9


python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/output.jsonl \
    --metrics sentiment \
    --output_file result_stats_${source}_${control}.txt

python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/output.jsonl \
    --metrics ppl-big \
    --output_file result_stats_${source}_${control}.txt.ppl

echo "Sentiment control results:"

