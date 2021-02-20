#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python model/run.py --dataset_basedir data/SST-2-XLNet-small/ \
                         --lr 2e-5  --max_epochs 20 \
                         --gpus 2 \
                         --accelerator ddp