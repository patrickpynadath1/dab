#!/bin/bash 

python experiments/evaluation/evaluate.py \
    --generations_file logs/dab-polarity-neg-to-pos/output_w=2.0.jsonl \
    --metrics sentiment \
    --output_file senti_res.jsonl 

python experiments/evaluation/evaluate.py \
    --generations_file logs/dab-polarity-pos-to-neg/output_w=2.0.jsonl \
    --metrics ppl-big \
    --output_file result_stats_dab_ppl.txt

echo "Sentiment control results:"