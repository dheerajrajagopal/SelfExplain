# SelfExplain Framework

## Todo

- [x] GIL interpret output (Inference)
- [] LIL interpret output (Inference)

## Preprocess to get POS-tagged data

On a python shell, do the following for installing the parser

```python
>>> import benepar
>>> benepar.download('benepar_en3')
```

```shell
sh scripts/run_preprocessing.sh
```


## Training

```shell
sh scripts/run_self_explain.sh
```
## Generation (Inference)

(In Progress)

```sh
 python model/infer_model.py
        --ckpt $PATH_TO_BEST_DEV_CHECKPOINT \
        --concept_map $DATA_FOLDER/concept_idx.json \ 
        --batch_size $BS \
        --paths_output_loc $PATH_TO_OUTPUT_PREDS
 ```
