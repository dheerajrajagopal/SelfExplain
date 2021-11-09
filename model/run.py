import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import random
import numpy as np
import pytorch_lightning as pl
import logging
from argparse import ArgumentParser
import resource
from data import ClassificationData, ICLRData
from SE_XLNet import SEXLNet
import pdb

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_train_steps(dm):
  total_devices = args.num_gpus * args.num_nodes
  train_batches = len(dm.train_dataloader()) // total_devices
  #return (args.max_epochs * train_batches) // args.accumulate_grad_batches
  return (args.max_epochs * train_batches)




rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# init: important to make sure every node initializes the same weights
SEED = 18
np.random.seed(SEED)
random.seed(SEED)
pl.utilities.seed.seed_everything(SEED)
pytorch_lightning.seed_everything(SEED)


# argparser
parser = ArgumentParser()
parser.add_argument('--num_gpus', type=int)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--clip_grad', type=float, default=1.0)
parser.add_argument("--dataset_basedir", help="Base directory where the dataset is located.", type=str, default='../../data/')
parser.add_argument("--concept_store", help="Concept store file", type=str, default='data/original_combined_data/concept_store.pt')
parser.add_argument("--model_name", default='xlnet-base-cased', help="Model to use.")
parser.add_argument("--gamma", default=0.01, type=float, help="Gamma parameter")
parser.add_argument("--lamda", default=0.01, type=float, help="Lamda Parameter")
parser.add_argument("--topk", default=100, type=int,help="Topk GIL concepts")
parser.add_argument("--nokeyword", action = 'store_true',help="To not use keywords")
parser.add_argument("--notopic", action='store_true', help="To not use topics")
parser.add_argument("--nolil", action='store_true', help='To not use LIL layer')
parser.add_argument('--name', default='temp', help='Model default root dir')
parser.add_argument('--use_weight', action='store_true', help='To use weight')

parser = pl.Trainer.add_argparse_args(parser)
parser = SEXLNet.add_model_specific_args(parser)

args = parser.parse_args()
args.num_gpus = len(str(args.gpus).split(","))
if args.notopic and args.nokeyword:
    print('Cant call both nokeyword and notopic. Using topics!')
    args.notopic = False
if args.nolil:
    args.lamda = 0.0


logging.basicConfig(level=logging.INFO)

# Step 1: Init Data
logging.info("Loading the data module")
dm = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name, batch_size=args.batch_size)
#dm = ICLRData(basedir=args.dataset_basedir, tokenizer_name=args.model_name, batch_size=args.batch_size, use_weight = args.use_weight)

# Step 2: Init Model
logging.info("Initializing the model")
model = SEXLNet(hparams=args)
#model = SEXLNet(hparams=args, label_weights=dm.label_weights)
train_steps = get_train_steps(dm)
model.hparams.warmup_steps = int(train_steps * model.hparams.warmup_prop)
lr_monitor = LearningRateMonitor(logging_interval='step')

# Step 3: Start
logging.info("Starting the training")
monitor_metric = 'val_topics_f1_epoch'
if args.notopic:
    monitor_metric = 'val_keywords_f1_epoch'

checkpoint_callback = ModelCheckpoint(
    #filename='{epoch}-{step}-{val_acc_epoch:.4f}',
    filename='{epoch}-{step}-{val_keywords_f1_epoch:.4f}',
    save_top_k=3,
    verbose=True,
    monitor=monitor_metric,
    mode='max'
)

tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/', name = args.name)
trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], val_check_interval=0.5, gradient_clip_val=args.clip_grad, track_grad_norm=2, logger=tb_logger)
trainer.fit(model, dm)
#trainer.test()
print ('Now val results')
res = trainer.test(model, test_dataloaders = dm.val_dataloader())
print ('Now test results')
res = trainer.test(model, test_dataloaders = dm.test_dataloader())
