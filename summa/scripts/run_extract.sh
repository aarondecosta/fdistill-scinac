#!/bin/sh

#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --time=0-04:00
#SBATCH --account=def-lilimou
#SBATCH --gres=gpu:1

nvidia-smi

module load gcc arrow/17.0.0 python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index torch
pip install --no-index tqdm
pip install --no-index tensorboard
pip install --no-index scikit-learn
pip install --no-index psutil
pip install --no-index sacrebleu
pip install --no-index rouge-score
pip install --no-index tensorflow_datasets
pip install --no-index pytorch-lightning
pip install --no-index matplotlib
pip install --no-index git-python
pip install --no-index faiss-cpu
pip install --no-index streamlit   
pip install --no-index elasticsearch
pip install --no-index numpy
pip install --no-index transformers
pip install --no-index nltk
pip install --no-index pandas
pip install --no-index datasets
pip install --no-index fire
pip install --no-index pytest
pip install --no-index conllu
pip install --no-index protobuf

export MODEL_NAME=xsum_student_init_model/best_tfmr
echo $MODEL_NAME
export BEAM=6
export SAVE_PATH=runs/pred-$(date +%m-%d-%y--%T)-teacher-eval

python3 run_extract.py \
  --model_name $MODEL_NAME \
  --input_path xsum/test.source \
  --save_path $SAVE_PATH/ \
  --reference_path xsum/test.target \
  --score_path $SAVE_PATH/metrics.json \
  --num_beams $BEAM\
  --length_penalty 0.5\
  --device cuda
  "$@"