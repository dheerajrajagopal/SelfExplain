import json
from operator import itemgetter

import torch
import numpy as np
from pytorch_lightning import Trainer
from tqdm import tqdm
import pandas as pd
import resource
from argparse import ArgumentParser

from SE_XLNet import SEXLNet
from data import ClassificationData


def load_model(ckpt, batch_size):
    model = SEXLNet.load_from_checkpoint(ckpt)
    model.eval()
    trainer = Trainer(gpus=1)
    dm = ClassificationData(basedir=model.hparams.dataset_basedir, tokenizer_name=model.hparams.model_name,
                            batch_size=batch_size)
    return model, trainer, dm


def load_dev_examples(file_name):
    dev_samples = []
    with open(file_name, 'r') as open_file:
        for line in open_file:
            dev_samples.append(json.loads(line))
    return dev_samples


def eval(model, dataloader, concept_map, dev_file, paths_output_loc: str = None):
    dev_samples = load_dev_examples(dev_file)
    total_evaluated = 0.
    total_correct = 0.
    i = 0
    predicted_labels, true_labels, gil_overall, lil_overall = [], [], [], []
    accs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_tokens, token_type_ids, nt_idx_matrix, labels = batch
            logits, acc, interpret_dict_list = model(batch)
            gil_interpretations = gil_interpret(concept_map=concept_map,
                                                list_of_interpret_dict=interpret_dict_list)
            lil_interpretations = lil_interpret(logits=logits,
                                                list_of_interpret_dict=interpret_dict_list,
                                                dev_samples=dev_samples,
                                                current_idx=i)
            accs.append(acc)
            batch_predicted_labels = torch.argmax(logits, -1)
            predicted_labels.extend(batch_predicted_labels.tolist())

            true_labels.extend(labels.tolist())
            gil_overall.extend(gil_interpretations)
            lil_overall.extend(lil_interpretations)

            total_evaluated += len(batch)
            total_correct += (acc.item() * len(batch))
            print(
                f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}, Batch accuracy = {round(acc.item(), 2)}")
            i += input_tokens.size(0)
        print(f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}")
        print(f"Accuracy = {round(np.array(accs).mean(), 2)}")
    pd.DataFrame({"predicted_labels": predicted_labels, "true_labels": true_labels, "lil_interpretations": lil_overall,
                  "gil_interpretations": gil_overall}).to_csv(paths_output_loc, sep="\t", index=None)


def gil_interpret(concept_map, list_of_interpret_dict):
    batch_concepts = []
    for topk_concepts in list_of_interpret_dict["topk_indices"]:
        concepts = [concept_map[x] for x in topk_concepts.tolist()][:10]
        batch_concepts.append(concepts)
    return batch_concepts


def lil_interpret(logits, list_of_interpret_dict, dev_samples, current_idx):
    sf_logits = torch.softmax(logits, dim=1).tolist()
    lil_sf_logits = torch.softmax(list_of_interpret_dict["lil_logits"], dim=-1).tolist()

    lil_outputs = []
    for idx, (sf_item, lil_sf_item) in enumerate(zip(sf_logits, lil_sf_logits)):
        dev_sample = dev_samples[current_idx + idx]
        lil_dict = {}
        argmax_sf, _ = max(enumerate(sf_item), key=itemgetter(1))
        for phrase_idx, phrase in enumerate(dev_sample["parse_tree"]):
            phrase_logits = lil_sf_logits[idx][phrase_idx]
            relevance_score = phrase_logits[argmax_sf] - sf_item[argmax_sf]
            if phrase_idx != 0:
                lil_dict[phrase["phrase"]] = relevance_score
        lil_dict = sorted(lil_dict.items(), key=lambda item: item[1], reverse=True)[:5]
        lil_outputs.append(lil_dict)
    return lil_outputs


def load_concept_map(concept_map_path):
    concept_map = {}
    with open(concept_map_path, 'r') as open_file:
        concept_map_str = json.loads(open_file.read())
    for key, value in concept_map_str.items():
        concept_map[int(key)] = value
    return concept_map


if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--concept_map', type=str)
    parser.add_argument('--dev_file', type=str, default="")
    parser.add_argument('--paths_output_loc', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    model, trainer, dm = load_model(args.ckpt,
                                    batch_size=args.batch_size)
    concept_map = load_concept_map(args.concept_map)
    eval(model,
         dm.val_dataloader(),
         concept_map=concept_map,
         dev_file=args.dev_file,
         paths_output_loc=args.paths_output_loc)
