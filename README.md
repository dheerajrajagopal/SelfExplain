# SelfExplain Framework

The code for the SelfExplain framework (https://arxiv.org/abs/2103.12279) 

Currently, this repo supports SelfExplain-XLNet and SelfExplain-RoBERTa version for SST-2 dataset, SST-5 dataset, 
and SUBJ dataset. We have also tested it with CoLA, which only RoBERTa provide reasonable performance because
sentences in the CoLA are too short for XLNet.

## Preprocessing

Data for preprocessing available in `data/` folder

On a python shell, do the following for installing the parser

```python
>>> import benepar
>>> benepar.download('benepar_en3')
```

```shell
sh scripts/run_preprocessing.sh
```

For preprocessing, we want to point out that we will need to adjust the hyperparameters on the top.
We have created two separate folders in data folder: RoBERTa-SST-2 and XLNet-SST-2. We expect users
follow this practice because concept store are unique for each Transformer-based classifier and 
each dataset.

Please comfirm DATA_FOLDER is the correct path.
Please comfirm TOKENIZER_NAME is the correct tokenizer you would like to use. (roberta-base or 
xlnet-base-cased).
Please comfirm MAX_LENGTH because this will affect the number of concepts. If MAX_LENGTH is  
small and average length for dataset is long, you may end up in training errors.

Example:
``` run_preprocessing.sh

export DATA_FOLDER='data/SST-2-XLNet'
export TOKENIZER_NAME='xlnet-base-cased'
export MAX_LENGTH=5

```

Note if you wish to parse test.tsv please edit process_trec_dataset.py at line 57.
Note we have provided data for SST-2 and SUBJ.

## Training

For training, please edit data path and control other parameters.

```shell
sh scripts/run_self_explain.sh
```

Example:

```run_self_explain.sh
python model/run.py --dataset_basedir data/RoBERTa-SST-2 \
                         --lr 2e-5  --max_epochs 5 \
                         --gpus 1 \
                         --model_name roberta-base \
                         --concept_store data/RoBERTa-SST-2/concept_store.pt \
                         --topk 5 \
                         --gamma 0.1 \
                         --lamda 0.1
```

Note the specified model_name should accord with the tokenizer used in the pre-processing stage.

## Generation (Inference)

The Original author claims this is in developing setting. We have utilized it and it works well.

```sh
 python model/infer_model.py
        --ckpt $PATH_TO_BEST_DEV_CHECKPOINT \
        --concept_map $DATA_FOLDER/concept_idx.json \ 
        --batch_size $BS \
        --paths_output_loc $PATH_TO_OUTPUT_PREDS \
        --dev_file $PATH_TO_DEV_FILE
 ```

Example:

```
 python model/infer_model.py 
      --ckpt lightning_logs/version_3/checkpoints/epoch=2-step=1499-val_acc_epoch=0.9570.ckpt \
      --concept_map data/RoBERTa-SST-2/concept_idx.json \
      --paths_output_loc result/result_roberta_7.csv \
      --dev_file data/RoBERTa-SST-2/dev_with_parse.json \
      --batch_size 16
```

## Citation 

```
@inproceedings{rajagopal-etal-2021-selfexplain,
    title = "{SELFEXPLAIN}: A Self-Explaining Architecture for Neural Text Classifiers",
    author = "Rajagopal, Dheeraj  and
      Balachandran, Vidhisha  and
      Hovy, Eduard H  and
      Tsvetkov, Yulia",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.64",
    doi = "10.18653/v1/2021.emnlp-main.64",
    pages = "836--850",
}
```
