#!/bin/bash
#SBATCH  --output=log/%j.out 
#SBATCH  --gres=gpu:1 
#SBATCH  --mem=30G

source /scratch_net/ken/jiezcao/anaconda3/etc/profile.d/conda.sh
conda activate usrswin
python -u main_train_psnr.py --opt options/train_usrnet_ST.json 
