#!/bin/bash
source /projects/tir5/users/rjoshi2/envs/torch16/bin/activate
export TOKENIZERS_PARALLELISM=false

# run normal
#python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2

# # run baseline - keywords no lil
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --nokeyword --nolil --name lightning_logs/topics_no_lil/

# # run baseline - topics no lil
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --notopic --nolil --name lightning_logs/keywords_no_lil/

# # run baseline - topics and keywords but no lil
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --nolil	--name lightning_logs/topics_keywords_no_lil/

# # run baseline - keywords 
# python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --nokeyword --name lightning_logs/topics_only/
python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --nokeyword --name lightning_logs/topics_only/

# run baseline - topics 
#python model/run.py --dataset_basedir ../../data/ --lr 2e-5 --max_epochs 5 --gpus 2 --accelerator ddp --concept_store data/original_combined_data/concept_store.pt --batch_size 2 --num_gpus 2 --notopic --name lightning_logs/keywords_only/
