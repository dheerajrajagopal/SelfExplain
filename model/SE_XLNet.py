from argparse import ArgumentParser

import torch
import pdb
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary

from model_utils import TimeDistributed
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SEXLNet(LightningModule):
    def __init__(self, hparams, label_weights = None):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        config = AutoConfig.from_pretrained(self.hparams.model_name)
        self.model = AutoModel.from_pretrained(self.hparams.model_name)
        self.pooler = SequenceSummary(config)

        self.classifier = nn.Linear(config.d_model, self.hparams.num_classes)
        self.keywords_classifier = nn.Linear(config.d_model, 64)
        self.topics_classifier = nn.Linear(config.d_model, 75)

        self.concept_store = torch.load(self.hparams.concept_store)

        self.phrase_logits = TimeDistributed(nn.Linear(config.d_model, self.hparams.num_classes))
        self.keywords_phrase_logits = TimeDistributed(nn.Linear(config.d_model, 64))
        self.topics_phrase_logits = TimeDistributed(nn.Linear(config.d_model, 75))

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
        self.gamma = 0 # make global layer 0

        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()
        self.label_weights = label_weights
        if self.label_weights == None:                  # backward compatible
            self.keyword_loss = nn.BCEWithLogitsLoss()
            self.topic_loss   = nn.BCEWithLogitsLoss()
        else:
            self.keyword_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(self.label_weights['keywords']))
            self.topic_loss   = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(self.label_weights['topics']))
        if 'nokeyword' in self.hparams:
            self.nokeyword = self.hparams.nokeyword
            self.notopic = self.hparams.notopic
            self.nolil = self.hparams.nolil
        else:
            self.nokeyword = False
            self.notopic = False
            self.nolil = False

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
        tokens, tokens_mask, padded_ndx_tensor, keywordlabels, topiclabels, _ = batch

        # step 1: encode the sentence
        sentence_cls, hidden_state = self.forward_classifier(input_ids=tokens,
                                                             token_type_ids=tokens_mask,
                                                             attention_mask=tokens_mask)
        #logits = self.classifier(sentence_cls)
        keywords_logits = self.keywords_classifier(sentence_cls)
        topics_logits = self.topics_classifier(sentence_cls)

        #lil_logits = self.lil(hidden_state=hidden_state, nt_idx_matrix=padded_ndx_tensor)
        keywords_lil_logits, topics_lil_logits = self.lil(hidden_state=hidden_state, nt_idx_matrix=padded_ndx_tensor)
        #lil_logits_mean = torch.mean(lil_logits, dim=1)
        keywords_lil_logits_mean = torch.mean(keywords_lil_logits, dim=1)
        topics_lil_logits_mean = torch.mean(topics_lil_logits, dim=1)
        #gil_logits, topk_indices = self.gil(pooled_input=sentence_cls)
        topk_indices = None

        ##logits = logits + self.lamda * lil_logits_mean + self.gamma * gil_logits
        #logits = logits + self.lamda * lil_logits_mean 
        if not self.nolil:
            keywords_logits = (1 - self.lamda) * keywords_logits + self.lamda * keywords_lil_logits_mean 
            topics_logits = (1 - self.lamda) * topics_logits + self.lamda * topics_lil_logits_mean 
        #predicted_labels = torch.argmax(logits, -1)
        labels = None
        if labels is not None:
            acc = torch.true_divide(
                (predicted_labels == labels).sum(), labels.shape[0])
        else:
            acc = None

        keywordsmetrics = self.get_metrics(keywords_logits, keywordlabels)
        topicsmetrics = self.get_metrics(topics_logits, topiclabels)
#        return keywords_logits, topics_logits, acc, {"topk_indices": topk_indices,
			#"keywords_lil_logits": keywords_lil_logits, 'topics_lil_logits': topics_lil_logits}
        return keywords_logits, topics_logits, keywordsmetrics, topicsmetrics, {"topk_indices": topk_indices,
                "keywords_lil_logits": keywords_lil_logits, 'topics_lil_logits': topics_lil_logits}

    def get_metrics(self, logits, labels):
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
        phrase_level_activations = phrase_level_activations - self.activation(hidden_state[:,0,:].unsqueeze(1))
        #phrase_level_logits = self.phrase_logits(phrase_level_activations)
        keywords_phrase_level_logits = self.keywords_phrase_logits(phrase_level_activations)
        topics_phrase_level_logits = self.topics_phrase_logits(phrase_level_activations)
        return keywords_phrase_level_logits, topics_phrase_level_logits


    def forward_classifier(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        """Returns the pooled token
        """
        #print(input_ids.shape, attention_mask.shape)
        outputs = self.model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True)
        hidden_states = outputs["hidden_states"]
        #hidden_states = outputs[2]
        cls_hidden_state = self.dropout(self.pooler(hidden_states[-1]))
        return cls_hidden_state, hidden_states[-1]

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        #logits, acc, _ = self(batch)
        keywords_logits, topics_logits, keywordsmetrics, topicsmetrics, _ = self(batch)
        #pdb.set_trace()
        keywords_loss = self.keyword_loss(keywords_logits, batch[-3].float())
        topics_loss = self.topic_loss(topics_logits, batch[-2].float())
        self.log('train_keywords_loss', keywords_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_topics_loss', topics_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #loss = self.loss(logits, batch[-1])
        # self.log('train_acc', acc, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        # return {"loss": loss}
        self.log('train_keywords_f1', keywordsmetrics['f1_weighted'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_topics_f1', topicsmetrics['f1_weighted'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #return {"keywords_loss": keywords_loss, "topics_loss": topics_loss}
        if self.nokeyword:
            loss = topics_loss
        elif self.notopic:
            loss = keywords_loss
        else:
            loss = keywords_loss + topics_loss
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        #logits, acc, _ = self(batch)
        keywords_logits, topics_logits, keywordsmetrics, topicsmetrics, _ = self(batch)

        # loss_f = nn.CrossEntropyLoss()
        # loss = loss_f(logits, batch[-1])
        # print(keywords_logits.shape)
        # print(topics_logits.shape)
        keywords_loss = self.keyword_loss(keywords_logits, batch[-3].float())
        topics_loss = self.topic_loss(topics_logits, batch[-2].float())

        #self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_keywords_loss', keywords_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_topics_loss', topics_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.log('val_keywords_f1', keywordsmetrics['f1_weighted'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_topics_f1', topicsmetrics['f1_weighted'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.nokeyword:
            loss = topics_loss
        elif self.notopic:
            loss = keywords_loss
        else:
            loss = keywords_loss + topics_loss
        return {"loss": loss}
        #return {"keywords_loss": keywords_loss, 'topics_loss': topics_loss}

    def test_step(self, batch, batch_idx):
        # # Load the data into variables
        # logits, acc, _ = self(batch)

        # loss_f = nn.CrossEntropyLoss()
        # loss = loss_f(logits, batch[-1])
        # return {"loss": loss}
        keywords_logits, topics_logits, keywordsmetrics, topicsmetrics, _ = self(batch)
        keywords_labels = batch[-3]
        topics_labels = batch[-2]

        # loss_f = nn.CrossEntropyLoss()
        # loss = loss_f(logits, batch[-1])
        # print(keywords_logits.shape)
        # print(topics_logits.shape)
        keywords_loss = self.keyword_loss(keywords_logits, batch[-3].float())
        topics_loss = self.topic_loss(topics_logits, batch[-2].float())

        #self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_keywords_loss', keywords_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_topics_loss', topics_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.log('test_keywords_f1', keywordsmetrics['f1_weighted'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_topics_f1', topicsmetrics['f1_weighted'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.nokeyword:
            loss = topics_loss
        elif self.notopic:
            loss = keywords_loss
        else:
            loss = keywords_loss + topics_loss
        #return {"loss": loss}
        return {"loss": loss, 'keywords_logits' : keywords_logits, 'topics_logits': topics_logits, 
                'keywords_labels': keywords_labels, 'topics_labels': topics_labels}
        #return {"keywords_loss": keywords_loss, 'topics_loss': topics_loss}

    def test_epoch_end(self, outputs):
        total_loss = torch.mean(torch.tensor([x['loss'] for x in outputs]))
        keywords_logits = torch.cat([x['keywords_logits'] for x in outputs])
        topics_logits = torch.cat([x['topics_logits'] for x in outputs])
        keywords_labels = torch.cat([x['keywords_labels'] for x in outputs])
        topics_labels = torch.cat([x['topics_labels'] for x in outputs])
        all_keywordsmetrics = self.get_metrics(keywords_logits, keywords_labels)
        all_topicsmetrics = self.get_metrics(topics_logits, topics_labels)
        return {'loss': total_loss, 'keywords_micro_f1': all_keywordsmetrics['f1_micro'], 
                'keywords_macro_f1': all_keywordsmetrics['f1_macro'], 'keywords_weighted_f1': all_keywordsmetrics['f1_weighted'],
                'topics_micro_f1': all_topicsmetrics['f1_micro'], 'topics_macro_f1': all_topicsmetrics['f1_macro'], 
                'topics_weighted_f1': all_topicsmetrics['f1_weighted'],
                'topics_report': all_topicsmetrics['report'], 'keywords_report': all_keywordsmetrics['report']}

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
