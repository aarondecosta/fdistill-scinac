#!/bin/sh

#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=0-01:00
#SBATCH --account=def-lilimou
#SBATCH --gres=gpu:1

nvidia-smi

export MODEL_NAME='sshleifer/distilbart-xsum-6-6'
echo $MODEL_NAME
export BEAM=6
export SAVE_PATH=runs/pred-$(date +%m-%d-%y--%T)-teacher-eval


python3 run_eval.py \
  --model_name $MODEL_NAME \
  --input_path xsum/test.source \
  --save_path $SAVE_PATH/res.out \
  --reference_path xsum/test.target \
  --score_path $SAVE_PATH/metrics.json \
  --num_beams $BEAM\
  --length_penalty 0.5\
  --device cuda
  "$@"