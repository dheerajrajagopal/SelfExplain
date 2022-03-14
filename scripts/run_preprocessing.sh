export DATA_FOLDER='data/XLNet-SUBJ'
export TOKENIZER_NAME='xlnet-base-cased'
export MAX_LENGTH=5

# Creates jsonl files for train and dev

python preprocessing/store_parse_trees.py \
      --data_dir $DATA_FOLDER  \
      --tokenizer_name $TOKENIZER_NAME

# Create concept store for SST-2 dataset
# Since SST-2 already provides parsed output, easier to do it this way, for other datasets, need to adapt

python preprocessing/build_concept_store.py \
       -i $DATA_FOLDER/train_with_parse.json \
       -o $DATA_FOLDER \
       -m $TOKENIZER_NAME \
       -l $MAX_LENGTH