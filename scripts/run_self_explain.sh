#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python model/run.py --dataset_basedir data/SST-2-XLNet \
                         --lr 2e-5  --max_epochs 5 \
                         --gpus 2 \
                         --concept_store data/SST-2-XLNet/concept_store.pt \
                         --accelerator ddp