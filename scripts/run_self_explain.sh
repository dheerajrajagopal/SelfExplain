#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python model/run.py --dataset_basedir data/ \
                         --lr 2e-5  --max_epochs 5 \
                         --gpus 2 \
                         --concept_store data/concept_store.pt \
                         --accelerator ddp \
                         --default_root_dir  "seed_18_k=10_multihead" \
                         --num_train_samples