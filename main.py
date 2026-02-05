import sys, os
import logging
from torch.utils import data
from torch.utils.data import dataset
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from federated.configuration import get_configs
from data.polyp.generate_data import prepare_fed_polyp
from federated.executer import FederatedExecuter
from federated.model_trainer_segmentation import ModelTrainerSegmentation
from collections.abc import Iterable

def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     

def custom_model_trainer(args):
     from model_zoo.models import UNet
     model = UNet()
     model_trainer = ModelTrainerSegmentation(model, args)
     return model_trainer

def prepare_dataset(args):
     datasets,train_len = prepare_fed_polyp(args)
     return datasets,train_len

def federated_executer(args, model_trainer, datasets,train_len=None):
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     executer = FederatedExecuter(datasets, device, args, model_trainer,train_len=train_len)
     return executer

def federated_executer_TTA(args, model_trainer, datasets,adapt_lrs=None,train_len=None):
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     executer = FederatedExecuter(datasets, device, args, model_trainer,adapt_lrs,train_len=train_len)
     return executer

if __name__ == "__main__":
     args = get_configs()
     deterministic(args.seed)
     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
     log_path = args.save_path.replace('checkpoint', 'log')
     if not os.path.exists(log_path): os.makedirs(log_path)
     log_path = log_path+'/log.txt'
     logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
     logging.info(str(args))
     datasets,train_len = prepare_dataset(args)

     if args.ood_test:
          ckpt1=torch.load(args.wei_path)
          ckpt2=torch.load(args.lr_path)
          model_trainer = custom_model_trainer(args)
          model_trainer.set_model_params(ckpt1)
          model_trainer.change_tta_bn(mode='grad') 
          model_trainer.get_model_params()
          federated_manager = federated_executer_TTA(args, model_trainer, datasets,ckpt2)
          federated_manager.ood_test(args)
          
          
     elif args.test:
          ckpt1=torch.load(args.wei_path)
          model_trainer = custom_model_trainer(args) 
          model_trainer.set_model_params(ckpt1)
          federated_manager = federated_executer_TTA(args, model_trainer, datasets,train_len=train_len)
          federated_manager.ood_test(args)

     elif args.train:
          model_trainer = custom_model_trainer(args)
          ckpt=torch.load(args.pt_path)
          model_trainer.set_model_params(ckpt) 
          model_trainer.change_bn(mode='grad') 
          federated_manager = federated_executer(args, model_trainer, datasets,train_len=train_len)
          federated_manager.run(args)
     
     elif args.pretrain:
          """
          this part is for pretraining stage
          """
          model_trainer = custom_model_trainer(args)
          federated_manager = federated_executer(args, model_trainer, datasets,train_len=train_len)
          federated_manager.train()
          

     

