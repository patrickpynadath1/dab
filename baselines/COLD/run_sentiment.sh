#!/bin/bash

## Counterfactual

python3 cold_decoding.py \
	--seed 12 \
	--mode sentiment \
	--pretrained_model gpt2-large \
	--init-temp 1 \
    --length 50 \
	--max-length 50 \
	--num-iters 400 \
	--min-iters 10 \
	--constraint-weight 0.7 \
    --counterfactual-max-ngram 3 \
	--stepsize 0.1 \
	--noise-iters 1 \
	--win-anneal-iters 1000 \
	--start 0 \
	--end 5 \
	--lr-nll-portion 0.9 \
    --topk 5 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--large-noise-iters 50,200,500 \
	--large_gs_std 0.5,0.1,0.05  \
	--input-file "./data/counterfactual/dev_data.json" \
	--output-dir "./data/counterfactual/" \
	--stepsize-ratio 1  \
    --batch-size 20 \
    --print-every 200
