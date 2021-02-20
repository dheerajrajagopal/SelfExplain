from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel, XLNetConfig
from transformers.modeling_utils import SequenceSummary


class SEXLNet(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        config = XLNetConfig()
        self.model = AutoModel.from_pretrained(self.hparams.model_name)
        self.pooler = SequenceSummary(config)

        self.logits_proj = nn.Linear(config.d_model, self.hparams.num_labels)
        self.proto_proj = nn.Linear(config.d_model, self.hparams.num_labels)
        self.proto_label_proj = nn.Linear(config.num_labels, self.hparams.num_labels)

        self.large_neg_val = -99999.0
        self.eps = 1e-06
        self.activation = nn.ReLU()

        self.lamda = self.hparams.lamda
        self.gamma = self.hparams.gamma

        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--min_lr", default=0, type=float,
                            help="Minimum learning rate.")
        parser.add_argument("--h_dim", type=int,
                            help="Size of the hidden dimension.", default=768)
        parser.add_argument("--n_heads", type=int,
                            help="Number of attention heads.", default=1)
        parser.add_argument("--kqv_dim", type=int,
                            help="Dimensionality of the each attention head.", default=256)
        parser.add_argument("--num_labels", type=float,
                            help="Number of classes.", default=2)
        parser.add_argument("--lr", default=5e-4, type=float,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay rate.")
        parser.add_argument("--warmup_prop", default=0., type=float,
                            help="Warmup proportion.")
        parser.add_argument("--lamda", default=0.2, type=float,
                            help="Lamda Parameter")
        parser.add_argument("--gamma", default=0.2, type=float,
                            help="Gamma parameter")
        parser.add_argument(
            "--model_name", default='xlnet-base-cased',  help="Model to use.")
        parser.add_argument(
            "--concept_store", required=True, help="Concept Store for GIL")
        return parser

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99),
                     eps=1e-8)
    
    def forward(self, batch):
        question_tokens, question_type_ids, question_masks, labels = batch

        # step 1: encode the question/paragraph
        question_cls_embeddeding = self.forward_bert(
            input_ids=question_tokens, token_type_ids=question_type_ids, attention_mask=question_masks)


        logits = self.classifier(question_cls_embeddeding)
        predicted_labels = torch.argmax(logits, -1)
        acc = torch.true_divide(
            (predicted_labels == labels).sum(), labels.shape[0])
        return logits, acc

    def forward_bert(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        """Returns the pooled token from BERT
        """
        outputs = self.model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs["hidden_states"]
        cls_embeddeding = self.dropout(self.pooler(hidden_states[-1])) 
        return cls_embeddeding

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc = self(batch)
        loss = self.loss(logits, batch[-1])
        self.log('train_acc', acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])

        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])
        return {"loss": loss}

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("val_loss_step", None)
        tqdm_dict.pop("val_acc_step", None)
        return tqdm_dict


if __name__ == "__main__":
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']
