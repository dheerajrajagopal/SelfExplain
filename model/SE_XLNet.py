from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, RobertaConfig, RobertaModel
from transformers.modeling_utils import SequenceSummary

from model_utils import TimeDistributed


class SEXLNet(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        # initialize config
        config = None

        print(self.hparams.model_name)
        if self.hparams.model_name == "xlnet-base-cased":

            config = AutoConfig.from_pretrained(self.hparams.model_name)
            self.model = AutoModel.from_pretrained(self.hparams.model_name)

        else:
            config = RobertaConfig()

            self.model = RobertaModel.from_pretrained(self.hparams.model_name)
            config = self.model.config
            config.d_model = config.hidden_size
            config.dropout = 0.2

        self.pooler = SequenceSummary(config)

        self.classifier = nn.Linear(config.d_model, self.hparams.num_classes)

        self.concept_store = torch.load(self.hparams.concept_store)

        self.phrase_logits = TimeDistributed(nn.Linear(config.d_model,
                                                        self.hparams.num_classes))
        self.sequence_summary = SequenceSummary(config)

        self.topk =  self.hparams.topk
        # self.topk_gil_mlp = TimeDistributed(nn.Linear(config.d_model,
        #                                               self.hparams.num_classes))

        self.topk_gil_mlp = nn.Linear(config.d_model,self.hparams.num_classes)

        self.multihead_attention = torch.nn.MultiheadAttention(config.d_model,
                                                               dropout=0.2,
                                                               num_heads=8)

        self.activation = nn.ReLU()

        self.lamda = self.hparams.lamda
        self.gamma = self.hparams.gamma

        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()

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
        parser.add_argument("--num_classes", type=float,
                            help="Number of classes.", default=2)
        parser.add_argument("--lr", default=2e-5, type=float,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay rate.")
        parser.add_argument("--warmup_prop", default=0.01, type=float,
                            help="Warmup proportion.")
        return parser

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99),
                     eps=1e-8)
    
    def forward(self, batch):
        self.concept_store = self.concept_store.to(self.model.device)
        # print(self.concept_store.size(), self.hparams.concept_store)
        tokens, tokens_mask, padded_ndx_tensor, labels = batch

        # step 1: encode the sentence
        if self.hparams.model_name == "xlnet-base-cased":
            sentence_cls, hidden_state = self.forward_classifier(input_ids=tokens,
                                                                token_type_ids=tokens_mask,
                                                                attention_mask=tokens_mask)
        else:
            sentence_cls, hidden_state = self.forward_classifier(input_ids=tokens,
                                                             attention_mask=tokens_mask)

        logits = self.classifier(sentence_cls)

        lil_logits = self.lil(hidden_state=hidden_state,
                              nt_idx_matrix=padded_ndx_tensor)
        lil_logits_mean = torch.mean(lil_logits, dim=1)
        gil_logits, topk_indices = self.gil(pooled_input=sentence_cls)

        logits = logits + self.lamda * lil_logits_mean + self.gamma * gil_logits
        predicted_labels = torch.argmax(logits, -1)
        if labels is not None:
            acc = torch.true_divide(
                (predicted_labels == labels).sum(), labels.shape[0])
        else:
            acc = None

        return logits, acc, {"topk_indices": topk_indices,
                             "lil_logits": lil_logits}

    def gil(self, pooled_input):
        batch_size = pooled_input.size(0)
        inner_products = torch.mm(pooled_input, self.concept_store.T)
        _, topk_indices = torch.topk(inner_products, k=self.topk)
        topk_concepts = torch.index_select(self.concept_store, 0, topk_indices.view(-1))
        topk_concepts = topk_concepts.view(batch_size, self.topk, -1).contiguous()

        concat_pooled_concepts = torch.cat([pooled_input.unsqueeze(1), topk_concepts], dim=1)
        attended_concepts, _ = self.multihead_attention(query=concat_pooled_concepts,
                                                     key=concat_pooled_concepts,
                                                     value=concat_pooled_concepts)

        gil_topk_logits = self.topk_gil_mlp(attended_concepts[:,0,:])
        # print(gil_topk_logits.size())
        # gil_logits = torch.mean(gil_topk_logits, dim=1)
        return gil_topk_logits, topk_indices

    def lil(self, hidden_state, nt_idx_matrix):
        phrase_level_hidden = torch.bmm(nt_idx_matrix, hidden_state)
        phrase_level_activations = self.activation(phrase_level_hidden)
        pooled_seq_rep = self.sequence_summary(hidden_state).unsqueeze(1)
        phrase_level_activations = phrase_level_activations - pooled_seq_rep
        phrase_level_logits = self.phrase_logits(phrase_level_activations)
        return phrase_level_logits


    def forward_classifier(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        """Returns the pooled token
        """
        outputs = self.model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True)
        hidden_states = outputs["hidden_states"]
        cls_hidden_state = self.dropout(self.pooler(hidden_states[-1]))
        return cls_hidden_state, hidden_states[-1]

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, _ = self(batch)
        loss = self.loss(logits, batch[-1])
        self.log('train_acc', acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, _ = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])

        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, _ = self(batch)

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
