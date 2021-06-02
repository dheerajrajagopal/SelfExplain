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
from data import ClassificationData, ICLRData
import pdb

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def load_model(ckpt, batch_size):
    model = SEXLNet.load_from_checkpoint(ckpt)
    model.eval()
    trainer = Trainer(gpus=1)
    #dm = ClassificationData(basedir=model.hparams.dataset_basedir, tokenizer_name=model.hparams.model_name,
                            #batch_size=batch_size)
    dm = ICLRData(basedir=model.hparams.dataset_basedir, tokenizer_name=model.hparams.model_name, batch_size=args.batch_size)
    return model, trainer, dm


def load_dev_examples(file_name):
    dev_samples = []
    with open(file_name, 'r') as open_file:
        for line in open_file:
            dev_samples.append(json.loads(line))
    return dev_samples


def eval(model, dataloader, concept_map, dev_file, paths_output_loc: str = None):
    #dev_samples = load_dev_examples(dev_file)
    total_evaluated = 0.
    total_correct = 0.
    i = 0
    predicted_labels, true_labels, gil_overall, lil_overall = [], [], [], []
    predicted_keywords_labels, true_keywords_labels, predicted_topics_labels, true_topics_labels = [], [], [], []
    accs = []
    #pdb.set_trace()
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            #input_tokens, token_type_ids, nt_idx_matrix, labels = batch
            input_tokens, token_type_ids, nt_idx_matrix, keywordlabels, topiclabels, phrases = batch
            #logits, acc, interpret_dict_list = model(batch)
            try:
                keywords_logits, topics_logits, keywordsmetrics, topicsmetrics, interpret_dict_list = model(batch)
            except:
                batch = [b.to('cuda') for b in batch[:-1]] + [batch[-1]]
                keywords_logits, topics_logits, keywordsmetrics, topicsmetrics, interpret_dict_list = model(batch)
            predicted_keywords_labels.append(keywords_logits)
            predicted_topics_labels.append(topics_logits)
            true_keywords_labels.append(keywordlabels)
            true_topics_labels.append(topiclabels)
            logits = {'keywords': keywords_logits, 'topics': topics_logits}
            labels = {'keywords': keywordlabels, 'topics': topiclabels}
            # the dict has keywords_lil_logits and topics_lil_logits
            # gil_interpretations = gil_interpret(concept_map=concept_map,
            #                                     list_of_interpret_dict=interpret_dict_list)
            keywords_lil_interpretations = lil_interpret(logits=logits['keywords'],
                                                interpret_logits=interpret_dict_list['keywords_lil_logits'],
                                                #dev_samples=dev_samples,
                                                phrases=phrases,
                                                current_idx=i)
            topics_lil_interpretations = lil_interpret(logits=logits['topics'],
                                                interpret_logits=interpret_dict_list['topics_lil_logits'],
                                                phrases=phrases,
                                                current_idx=i)
            from collections import defaultdict as ddict
            topic2phrase = ddict(list)
            for pi in range(len(phrases[0])): # only for first item in batch
                for t in range(75):
                    topic2phrase[t].append((topics_lil_interpretations[0][pi][t], phrases[0][pi]))
            for t in range(75):
                topic2phrase[t] = sorted(topic2phrase[t], key = lambda x: x[0], reverse = True)
            pdb.set_trace()
            true_labels_0 = torch.where(topiclabels[0])
            templogits = topics_logits[0]
            pred = torch.sigmoid(templogits)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred_labels_0 = torch.where(pred)
            lil_logits = interpret_dict_list['topics_lil_logits'][0]
            rels = torch.sigmoid(lil_logits) - torch.sigmoid(templogits)
	    # scores for a particular topic - rels[:, 12]
	    # top phrases for a particular topic - topic2phrase[12][:20]



            #accs.append(acc)
            # all_keywordsmetrics.append(keywordsmetrics)
            # all_topicsmetrics.append(topicsmetrics)
            # batch_predicted_labels = torch.argmax(logits, -1)
            # predicted_labels.extend(batch_predicted_labels.tolist())
            batch_predicted_keywords_labels = torch.argmax(keywords_logits, -1)
            predicted_keywords_labels.extend(batch_predicted_keywords_labels.tolist())
            batch_predicted_topics_labels = torch.argmax(topics_logits, -1)
            predicted_topics_labels.extend(batch_predicted_topics_labels.tolist())

            # true_labels.extend(labels.tolist())
            #true_keywords_labels.extend(keywordlabels.tolist())
            #true_topics_labels.extend(topicslabels.tolist())
            #gil_overall.extend(gil_interpretations)
            #lil_overall.extend(lil_interpretations)

            #print(gil_interpretations)
            #print(lil_interpretations)
            #print(dev_samples[i]["sentence"])
            #pdb.set_trace()
            #exit(0)
            #total_evaluated += len(batch)
            #total_correct += (acc.item() * len(batch))
            #print(
                #f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}, Batch accuracy = {round(acc.item(), 2)}")
            #i += input_tokens.size(0)
        predicted_keywords_labels = torch.cat(predicted_keywords_labels)
        predicted_topics_labels = torch.cat(predicted_topics_labels)
        true_keywords_labels = torch.cat(true_keywords_labels)
        true_topics_labels = torch.cat(true_topics_labels)
        all_keywordsmetrics = get_metrics(predicted_keywords_labels, true_keywords_labels)
        all_topicsmetrics = get_metrics(predicted_topics_labels, true_topics_labels)
        pdb.set_trace()
        # print(f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}")
        # print(f"Accuracy = {round(np.array(accs).mean(), 2)}")
    pd.DataFrame({"predicted_labels": predicted_labels, "true_labels": true_labels, "lil_interpretations": lil_overall,
                  "gil_interpretations": gil_overall}).to_csv(paths_output_loc, sep="\t", index=None)

def get_metrics(logits, labels):
    logits = logits.cpu()
    labels = labels.cpu()
    preds = sigmoid(logits.detach())
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    metric = {}
    metric['report'] = classification_report(labels, preds)
    for averaging in ['macro', 'micro', 'weighted']:
        metric['f1_{}'.format(averaging)] = f1_score(labels, preds, average = averaging)
        #metric['roc_auc_{}'.format(averaging)] = roc_auc_score(labels, logits, average = averaging)
    return metric

def gil_interpret(concept_map, list_of_interpret_dict):
    batch_concepts = []
    for topk_concepts in list_of_interpret_dict["topk_indices"]:
        concepts = [concept_map[x] for x in topk_concepts.tolist()][:10]
        batch_concepts.append(concepts)
    return batch_concepts


#def lil_interpret(logits, interpret_logits, dev_samples, current_idx):
def lil_interpret(logits, interpret_logits, phrases, current_idx):
    sf_logits = torch.sigmoid(logits).tolist()                                                          # 4 x 64
    #lil_sf_logits = torch.sigmoid(list_of_interpret_dict[f'{labelname}_lil_logits']).tolist()
    lil_sf_logits = torch.sigmoid(interpret_logits).tolist()                                            # 4 x zzz x 64

    lil_outputs = []
    for idx, (sf_item, lil_sf_item) in enumerate(zip(sf_logits, lil_sf_logits)):                        # for each item in batch
        #dev_sample = dev_samples[current_idx + idx]
        lil_dict = []
        # argmax_sf, _ = max(enumerate(sf_item), key = itemgetter(1))                                     # just doing argmax
        preds = np.array(sf_item)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        for phrase_idx, phrase in enumerate(phrases[idx]):
            phrase_logits = lil_sf_item[phrase_idx]
            assert len(phrase_logits) == len(sf_item)
            # if phrase == 'Islam':
            #     print (phrase_logits, sf_item)
            #     pdb.set_trace()
            relevance_scores_per_topic = np.array(phrase_logits) - np.array(sf_item)
            #lil_dict[phrase] = relevance_scores_per_topic
            lil_dict.append(relevance_scores_per_topic)
        # for phrase_idx, phrase in enumerate(dev_sample['parse_tree']):
        #     phrase_logits = lil_sf_logits[idx][phrase_idx]
        #     relevance_score = phrase_logits[argmax_sf] - sf_item[argmax_sf]
        #     if phrase_idx != 0:
        #         lil_dict[phrase["phrase"]] = relevance_score
        lil_outputs.append(lil_dict)
    return lil_outputs

#def lil_interpret(logits, list_of_interpret_dict, dev_samples, current_idx):
def lil_interpretold(keywords_logits, topics_logits, list_of_interpret_dict, dev_samples, current_idx):
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
    res = trainer.test(model, test_dataloaders = dm.val_dataloader())
    pdb.set_trace()
    eval(model,
         dm.val_dataloader(),
         concept_map=concept_map,
         dev_file=args.dev_file,
         paths_output_loc=args.paths_output_loc)
