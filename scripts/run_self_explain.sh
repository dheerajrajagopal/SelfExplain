#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python model/run.py --dataset_basedir data/XLNet-SUBJ \
                         --lr 2e-5  --max_epochs 5 \
                         --gpus 1 \
                         --concept_store data/XLNet-SUBJ/concept_store.pt \
                         --accelerator ddp \
                         --gamma 0.1 \
                         --lamda 0.1 \
                         --topk 5

# for RoBERTa
# python model/run.py --dataset_basedir data/RoBERTa-SUBJ \
#                          --lr 2e-5  --max_epochs 5 \
#                          --gpus 1 \
#                          --concept_store data/RoBERTa-SUBJ/concept_store.pt \
#                          --accelerator ddp \
#                          --model_name roberta-base \
#                          --topk 5 \
#                          --gamma 0.1 \
#                          --lamda 0.1
