#!/bin/bash
source /projects/tir5/users/rjoshi2/envs/torch16/bin/activate
export TOKENIZERS_PARALLELISM=false
# run normal
#python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --name full

# main full run 50 epochs
#python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 50 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 4 --num_gpus 2 --name full_50_2e5
#python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 100 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 4 --num_gpus 2 --name full_100_2e5
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 100 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 4 --num_gpus 2 --name full_100_2e5_weight --use_weight

# only lil layer full
#python model/run.py --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --batch_size 4 --num_gpus 2 --name only_lil_full_5_2e5 --lamda 1.0
# no lil
#python model/run.py --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --batch_size 4 --num_gpus 2 --name no_lil_full_5_2e5 --lamda 0.0
# half lil
python model/run.py --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --batch_size 4 --num_gpus 2 --name lil_05_full_5ep_2e5lr --lamda 0.5

# run baseline - keywords no lil
#python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --nokeyword --nolil --name lightning_logs/topics_no_lil/

# run baseline - topics no lil
#python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --notopic --nolil --name lightning_logs/keywords_no_lil/

# # run baseline - topics and keywords but no lil
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --nolil	--name lightning_logs/topics_keywords_no_lil/

# # run baseline - keywords 
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --nokeyword --name lightning_logs/topics_only/

# # run baseline - topics 
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --notopic --name lightning_logs/keywords_only/
