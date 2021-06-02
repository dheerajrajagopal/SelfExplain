"""Wrapper for a conditional generation dataset present in 2 tab-separated columns:
source[TAB]target
"""
import logging
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import pickle
import random
import pdb
import json
import multiprocessing

from data_utils import pad_nt_matrix_roberta, pad_nt_matrix_xlnet

class ICLRData(pl.LightningDataModule):
    def __init__(self, basedir: str, tokenizer_name: str, batch_size: int, num_workers: int = 16, use_weight: bool = False):
        super().__init__()
        self.basedir = basedir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        self.collator = MyCollator(tokenizer_name)
        #self.all_files = pickle.load(open(basedir + 'all_iclr_files.pkl', 'rb'))
        self.all_files = pickle.load(open(basedir + 'all_iclr_files_split.pkl', 'rb'))
        #random.Random(47).shuffle(self.all_files)                               # always same
        self.train_files, self.test_files, self.val_files = self.get_test_train_split()
        self.use_weight = use_weight

        self.train_dataset = ICLRDataset(tokenizer = self.tokenizer, data_path = f'{self.basedir}ICLR.cc.parse_roberta-base_selfexplaindata/',
                filenames = self.train_files)
        self.val_dataset = ICLRDataset(tokenizer = self.tokenizer, data_path = f'{self.basedir}ICLR.cc.parse_roberta-base_selfexplaindata/',
                filenames = self.val_files, isval = True)
        self.test_dataset = ICLRDataset(tokenizer = self.tokenizer, data_path = f'{self.basedir}ICLR.cc.parse_roberta-base_selfexplaindata/',
                filenames = self.test_files)

        self.label_weights = {'keywords': np.ones(64), 'topics': np.ones(75)}
        if self.use_weight:
            kls = np.array(self.train_dataset.keywordlabels)
            tls = np.array(self.train_dataset.topiclabels)
            total_egs = kls.shape[0]
            keyword_weight = np.sum(kls, axis = 0)
            topic_weight = np.sum(tls, axis = 0)
            for k in range(len(keyword_weight)):
                if keyword_weight[k] != 0:
                    self.label_weights['keywords'][k] = (total_egs - keyword_weight[k]) / keyword_weight[k] # neg / pos
            for t in range(len(topic_weight)):
                if topic_weight[t] != 0:
                    self.label_weights['topics'][t] = (total_egs - topic_weight[t]) / topic_weight[t] # neg / pos

    def get_test_train_split(self):
        return self.all_files['train'], self.all_files['dev'], self.all_files['test']
        # n = len(self.all_files)
        # train = self.all_files[:int(0.8*n)]
        # dev = self.all_files[int(0.8*n):int(0.9*n)]
        # test = self.all_files[int(0.9*n):]
        # return train, test, dev

    def train_dataloader(self):
        # dataset = ICLRDataset(tokenizer = self.tokenizer, data_path = f'{self.basedir}ICLR.cc.parse_roberta-base_selfexplaindata/',
        #         filenames = self.train_files)
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self):
        # dataset = ICLRDataset(tokenizer = self.tokenizer, data_path = f'{self.basedir}ICLR.cc.parse_roberta-base_selfexplaindata/',
        #         filenames = self.val_files, isval = True)
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)

    def test_dataloader(self):
        # dataset = ICLRDataset(tokenizer = self.tokenizer, data_path = f'{self.basedir}ICLR.cc.parse_roberta-base_selfexplaindata/',
        #         filenames = self.test_files)
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)



class ICLRDataset(Dataset):
    def __init__(self, tokenizer, data_path: str, filenames: list, isval = False) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.filenames = filenames
        self.files_w_keywords = 0
        self.files_w_topics = 0
        self.isval = isval
        self.labels = self.read_labels()
        self.read_dataset()

    def get_nt_idx_matrix(self, row):
        rows = 0
        cols = 0
        for mat in row['nt_idx_matrices']:
            rows += len(mat)
            cols += len(mat[0])
        fullmat = np.zeros((rows, cols))
        curr_row = 0
        curr_col = 0
        for mat in row['nt_idx_matrices']:
            fullmat[curr_row: curr_row+len(mat), curr_col:curr_col+len(mat[0])] = np.array(mat)
            curr_row += len(mat)
            curr_col += len(mat[0])
        return torch.tensor(fullmat).long()

    def read_labels(self):
        logging.info("Reading labels, keywords and topics")
        papers2keywordclusters64 = pickle.load(open(self.data_path + '../ICLR.cc/papers2keywordclusters64.pkl', 'rb'))
        papers2topiclabels05 = pickle.load(open(self.data_path + '../ICLR.cc/papers2topiclabels_final.pkl', 'rb'))
        logging.info(f"Papers with keywords {len(papers2keywordclusters64)}, Papers with Topics {len(papers2topiclabels05)}")
        return {'keywords': papers2keywordclusters64, 'topics': papers2topiclabels05}

    def read_file(self, fname):
        #pdb.set_trace()
        try:
            data = json.load(open(self.data_path + fname + '_processed_idx.json'))
        except:
            print ('File {} not present. Please process first'.format(fname))
            return None, None, None, None
        if 'text' not in data:
            data['text'] = [' '.join(x) for x in data['sentences']]
        if len(data['text']) == 0:
            print('File {} has no text'.format(fname))
            return None, None, None, None, None, None
            #pdb.set_trace()
        # data has keys labels, text, nt_idx_matrices, phrase_labels which are all list of paras
        skip = False
        if fname not in self.labels['keywords']:
            skip = True
        else:
            self.files_w_keywords += 1
        if fname not in self.labels['topics']:
            skip = True
        else:
            self.files_w_topics += 1
        if skip:
            return None, None, None, None, None, None
        keyword_cluster_labels = self.labels['keywords'][fname]
        topic_labels = self.labels['topics'][fname]
        keywords = np.zeros(64)
        topics = np.zeros(75)
        keywords[np.asarray(keyword_cluster_labels)] = 1 
        topics[np.asarray(topic_labels)] = 1
        text = data['text'][0] # take para 1 only
        phrases = data['phrase_text'][0]
        #nt_idx_matrix = data['nt_idx_matrices'][0]
        indexes = data['nt_idx_matrices_indices'][0]

        shape = data['shapes'][0]
        # phrase_label_idx = np.where(np.asarray(np.asarray(data['phrase_labels'][0]) == 'NP') == True)[0]
        # nt_idx_matrix = np.asarray(nt_idx_matrix)[phrase_label_idx]

        #return text, np.array(nt_idx_matrix), keywords, topics
        return text, indexes, shape, keywords, topics, phrases

    def read_dataset(self):
        filenames = self.filenames[:]
        logging.info("Reading {} data from {}".format(len(filenames), self.data_path))
        self.texts, self.keywordlabels, self.topiclabels, self.nt_idx_matrices = [], [], [], []
        self.indices, self.shapes = [], []
        self.phrases = []
        
        # multiprocessing.freeze_support()
        # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # d = dict(pool.map(self.read_file, filenames))
        # pdb.set_trace()
        for i, file in tqdm(enumerate(filenames), total=len(filenames), desc='Reading files'):
            #text, nt_idx_matrix, keywords, topics = self.read_file(file)
            # if file == 'BkxgbhCqtQ':
            #     continue # skip this file because text is wrongly processed
            text, indexes, shape, keywords, topics, phrases = self.read_file(file)
            if text == None:
                continue
            self.texts.append(text)
            if self.isval:
                self.phrases.append(phrases)
            # if len(text) > 10000:
            #     print(file)
            #     pdb.set_trace()
            #self.nt_idx_matrices.append(nt_idx_matrix)
            self.indices.append(indexes)
            self.shapes.append(shape)
            self.keywordlabels.append(keywords)
            self.topiclabels.append(topics)
        encoded_input = self.tokenizer(self.texts)
        logging.info('Read {} data points, selected {}. Files with keywords {}, Files with topics {}'.format(len(filenames),
            len(self.texts), self.files_w_keywords, self.files_w_topics))
        self.input_ids = encoded_input['input_ids']
        self.input_ids = [x[:600]+[x[-1]] for x in self.input_ids]          # remove "extra" text. OOM FIX!!
        if 'token_type_ids' in encoded_input:
            self.token_type_ids = encoded_input["token_type_ids"]
        else:
            self.token_type_ids = [[0] * len(s) for s in encoded_input["input_ids"]]


    def read_dataset2(self):
        logging.info("Reading {} data from {}".format(len(filenames), self.data_path))
        data = pd.read_json(self.data_path, orient="records", lines=True)
        self.texts, self.all_sentences, self.answer_labels, self.nt_idx_matrices = [], [], [], []
        self.nt_idx_matrix = []
        logging.info(f"Reading dataset file from {self.data_path}")
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Reading dataset samples"):
            self.answer_labels.append(int(row["label"]))
            self.texts.append(row['text'])
            self.nt_idx_matrix.append(self.get_nt_idx_matrix(row))
            #self.nt_idx_matrices.append([torch.tensor(x).long() for x in row['nt_idx_matrices']])
        encoded_input = self.tokenizer(self.texts)
        self.input_ids = encoded_input["input_ids"]
        if "token_type_ids" in encoded_input:
            self.token_type_ids = encoded_input["token_type_ids"]
        else:
            self.token_type_ids = [[0] * len(s) for s in encoded_input["input_ids"]]

    def __len__(self) -> int:
            return len(self.texts)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        #return (self.input_ids[i], self.token_type_ids[i], self.nt_idx_matrix[i], self.answer_labels[i])
        nt_idx_matrices = np.zeros(self.shapes[i])
        nt_idx_matrices[self.indices[i][0], self.indices[i][1]] = 1
        #return (self.input_ids[i], self.token_type_ids[i], self.nt_idx_matrices[i], self.keywordlabels[i], self.topiclabels[i])
        if not self.isval:
            return (self.input_ids[i], self.token_type_ids[i], nt_idx_matrices, self.keywordlabels[i], self.topiclabels[i], None)
        else:
            return (self.input_ids[i], self.token_type_ids[i], nt_idx_matrices, self.keywordlabels[i], self.topiclabels[i], self.phrases[i])

class ClassificationData(pl.LightningDataModule):
    def __init__(self, basedir: str, tokenizer_name: str, batch_size: int, num_workers: int = 16):
        super().__init__()
        self.basedir = basedir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        self.collator = MyCollator(tokenizer_name)

    def train_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/train_with_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/dev_with_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)

    def test_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/test_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)



class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.read_dataset()

    def get_nt_idx_matrix(self, row):
        rows = 0
        cols = 0
        for mat in row['nt_idx_matrices']:
            rows += len(mat)
            cols += len(mat[0])
        fullmat = np.zeros((rows, cols))
        curr_row = 0
        curr_col = 0
        for mat in row['nt_idx_matrices']:
            fullmat[curr_row: curr_row+len(mat), curr_col:curr_col+len(mat[0])] = np.array(mat)
            curr_row += len(mat)
            curr_col += len(mat[0])
        return torch.tensor(fullmat).long()

    def read_dataset(self):
        logging.info("Reading data from {}".format(self.data_path))
        data = pd.read_json(self.data_path, orient="records", lines=True)
        self.texts, self.all_sentences, self.answer_labels, self.nt_idx_matrices = [], [], [], []
        self.nt_idx_matrix = []
        logging.info(f"Reading dataset file from {self.data_path}")
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Reading dataset samples"):
            self.answer_labels.append(int(row["label"]))
            self.texts.append(row['text'])
            self.nt_idx_matrix.append(self.get_nt_idx_matrix(row))
            #self.nt_idx_matrices.append([torch.tensor(x).long() for x in row['nt_idx_matrices']])
        encoded_input = self.tokenizer(self.texts)
        self.input_ids = encoded_input["input_ids"]
        if "token_type_ids" in encoded_input:
            self.token_type_ids = encoded_input["token_type_ids"]
        else:
            self.token_type_ids = [[0] * len(s) for s in encoded_input["input_ids"]]

    def __len__(self) -> int:
            return len(self.texts)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return (self.input_ids[i], self.token_type_ids[i], self.nt_idx_matrix[i], self.answer_labels[i])


class ClassificationDatasetOld(Dataset):
    def __init__(self, tokenizer, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.read_dataset()


    def read_dataset(self):
        logging.info("Reading data from {}".format(self.data_path))
        data = pd.read_json(self.data_path, orient="records", lines=True)
        self.sentences, self.answer_labels, self.nt_idx_matrix = [], [], []
        logging.info(f"Reading dataset file from {self.data_path}")
        # print(data, len(data))
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Reading dataset samples"):
            self.answer_labels.append(int(row["label"]))
            self.sentences.append(row["sentence"])
            self.nt_idx_matrix.append(torch.tensor(row["nt_idx_matrix"]).long())

        encoded_input = self.tokenizer(self.sentences)
        self.input_ids = encoded_input["input_ids"]
        if "token_type_ids" in encoded_input:
            self.token_type_ids = encoded_input["token_type_ids"]
        else:
            self.token_type_ids = [[0] * len(s) for s in encoded_input["input_ids"]]


    def __len__(self) -> int:
            return len(self.sentences)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return (self.input_ids[i], self.token_type_ids[i], self.nt_idx_matrix[i], self.answer_labels[i])


class MyCollator(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if "xlnet" in model_name:
            self.pad_fn = pad_nt_matrix_xlnet
        elif "roberta" in model_name:
            self.pad_fn = pad_nt_matrix_roberta
        else:
            raise NotImplementedError

    def __call__(self, batch):
        max_token_len = 0
        max_phrase_len = 0
        num_elems = len(batch)
        for i in range(num_elems):
            tokens, _, idx_m, _, _, _ = batch[i]
            #max_token_len = max(max_token_len, len(tokens))
            max_token_len = max(max_token_len, np.array(idx_m).shape[1])
            max_token_len = max(max_token_len, len(tokens))
            max_phrase_len = max(max_phrase_len, np.array(idx_m).shape[0])

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        #labels = torch.zeros(num_elems).long()
        keywords_labels = torch.zeros((num_elems, 64)).long()
        topics_labels = torch.zeros((num_elems, 75)).long()
        nt_idx_matrix = []
        phrases = []

        for i in range(num_elems):
            #toks, _, idx_matrix, label = batch[i]
            toks, _, idx_matrix, keyword_label, topic_label, phrase = batch[i]
            # idx_matrix = torch.tensor(idx_matrix).long()
            idx_matrix = self.pad_fn(nt_idx_matrix=torch.tensor(idx_matrix),
                                     max_nt_len=max_phrase_len,
                                     max_length=max_token_len)
            length = len(toks)
            try:
                tokens[i, :length] = torch.LongTensor(toks)
            except:
                print(tokens.shape)
                print(length)
                print(max_token_len)
                print(tokens)
                pdb.set_trace()
            tokens_mask[i, :length] = 1
            nt_idx_matrix.append(idx_matrix)
            #labels[i] = label
            keywords_labels[i] = torch.tensor(keyword_label)
            topics_labels[i] = torch.tensor(topic_label)
            phrases.append(phrase)

        padded_ndx_tensor = torch.stack(nt_idx_matrix, dim=0)
        #return [tokens, tokens_mask, padded_ndx_tensor, labels]
        return [tokens, tokens_mask, padded_ndx_tensor, keywords_labels, topics_labels, phrases]


class MyCollatorOld(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if "xlnet" in model_name:
            self.pad_fn = pad_nt_matrix_xlnet
        elif "roberta" in model_name:
            self.pad_fn = pad_nt_matrix_roberta
        else:
            raise NotImplementedError

    def __call__(self, batch):
        max_token_len = 0
        max_phrase_len = 0
        num_elems = len(batch)
        for i in range(num_elems):
            tokens, _, idx_m, _ = batch[i]
            max_token_len = max(max_token_len, len(tokens))
            max_phrase_len = max(max_phrase_len, idx_m.size(0))

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        labels = torch.zeros(num_elems).long()
        nt_idx_matrix = []

        for i in range(num_elems):
            toks, _, idx_matrix, label = batch[i]
            # idx_matrix = torch.tensor(idx_matrix).long()
            idx_matrix = self.pad_fn(nt_idx_matrix=idx_matrix,
                                     max_nt_len=max_phrase_len,
                                     max_length=max_token_len)
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            nt_idx_matrix.append(idx_matrix)
            labels[i] = label

        padded_ndx_tensor = torch.stack(nt_idx_matrix, dim=0)
        return [tokens, tokens_mask, padded_ndx_tensor, labels]



if __name__ == "__main__":
    import sys
    dm = ClassificationData(
        basedir=sys.argv[1], model_name=sys.argv[2], batch_size=32)
    for (tokens, tokens_mask, nt_idx_matrix, labels) in dm.train_dataloader():
        print(torch.tensor(tokens_mask[0].tokens).shape)
