#!/bin/bash 


TRIAL=detoxification-gpt2-large
# mkdir -p logs/$TRIAL

# # bash data/toxicity/toxicity_preprocess.sh \
# #     data/toxicity/jigsaw ;

# PYTHONPATH=. python experiments/training/train.py \
#     --dataset_name toxicity \
#     --data_dir data/toxicity/jigsaw \
#     --ckpt_name logs/$TRIAL/checkpoint.pt \
#     --model gpt2-large --cuda \
#     --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
#     --batch_size 32 --max_length 256 \
#     --n_steps 1000 --lr 1e-2 \
#     --cuda ;

PYTHONPATH=. python experiments/training/generate_from_text_file.py \
    --eval_file  prompts/detoxic_prompts_1k.txt\
    --output_file logs/$TRIAL/outputs.txt\
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values 5 1
