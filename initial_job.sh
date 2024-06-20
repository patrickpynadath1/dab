#!/bin/bash
# FILENAME:  initial_job.sh

module load anaconda/2020.11-py38

echo "Running initial job script"

# should be run from the project directory 
cd $SLURM_SUBMIT_DIR
pip install -r requirements.txt


# initial setup for transformers 
cd transformers
pip install -e .
cd ..

# first try 
python main.py --exp sentiment --device cuda --eval_on_fin bolt; 
python main.py --exp sentiment --device cuda --eval_on_fin dlp; 
