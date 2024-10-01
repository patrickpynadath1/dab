#!/bin/bash 

TRIAL=polarity-gpt2-large
mkdir -p logs/$TRIAL

source=positive
control=-5
PYTHONPATH=. python experiments/training/train.py \
    --dataset_name polarity-yelp \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
    --batch_size 32 --max_length 256 \
    --n_steps 1000 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3;