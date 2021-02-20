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


class ClassificationData(pl.LightningDataModule):
    def __init__(self, basedir: str, tokenizer_name: str, batch_size: int, num_workers: int = 16):
        super().__init__()
        self.basedir = basedir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

    def train_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/train_with_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=ClassificationDataset.collate_pad)

    def val_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/dev_with_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=ClassificationDataset.collate_pad)

    def test_dataloader(self):
        dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                        data_path=f"{self.basedir}/test_parse.json")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=ClassificationDataset.collate_pad)





class ClassificationDataset(Dataset):
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
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Reading dataset samples"):
            self.answer_labels.append(int(row["label"]))
            self.sentences.append(row["sentence"])
            self.nt_idx_matrix.append(row["nt_idx_matrix"])

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
        return (self.input_ids[i], self.nt_idx_matrix[i], self.answer_labels[i])

    @staticmethod
    def collate_pad(batch):
        max_token_len = 0
        num_elems = len(batch)
        for i in range(num_elems):
            tokens, _, _ = batch[i]
            max_token_len = max(max_token_len, len(tokens))

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        labels = torch.zeros(num_elems).long()
        nt_idx_matrix = []

        for i in range(num_elems):
            toks, idx_matrix , label = batch[i]
            idx_matrix = torch.tensor(idx_matrix)
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            nt_idx_matrix.append(idx_matrix)

        padded_ndx_tensor = ClassificationDataset.pad_phrase_tensor(nt_idx_matrix)
        return [tokens, tokens_mask, padded_ndx_tensor, labels]

    @staticmethod
    def pad_phrase_tensor(list_nt_idx_matrix):
        max_sent_len = max([x.size(1) for x in list_nt_idx_matrix])
        max_num_phrases = max([x.size(0) for x in list_nt_idx_matrix])
        batch_size = len(list_nt_idx_matrix)

        tokens = torch.zeros(batch_size, max_num_phrases, max_sent_len).long()

        for i in range(batch_size):
            row_len, col_len = list_nt_idx_matrix[i].size()
            tokens[i, :row_len, :col_len] = list_nt_idx_matrix[i]

        return tokens


if __name__ == "__main__":
    import sys
    dm = ClassificationData(
        basedir=sys.argv[1], model_name=sys.argv[2], batch_size=32)
    for (tokens, tokens_mask, nt_idx_matrix, labels) in dm.train_dataloader():
        print(torch.tensor(tokens_mask[0].tokens).shape)
