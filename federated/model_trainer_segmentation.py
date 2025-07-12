import logging

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .model_trainer import ModelTrainer
from utils.loss import DiceLoss
import copy
import numpy as np
import random
import tensorflow as tf
from .MyBatchNorm2d import MyBatchNorm2d

from .loss_contrast_memv2 import  MemContrastLoss
from .loss_contrast_batch import MiniBContrastLoss
from .loss_contrast_server_mem import ServerPixelContrastLoss
from .create_loss import turn_prediction
from time import time
    
def deterministic(seed):
     cudnn.benchmark = False  
     cudnn.deterministic = True  
     np.random.seed(seed) 
     torch.manual_seed(seed) 
     torch.cuda.manual_seed_all(seed)  
     random.seed(seed)  

class ModelTrainerSegmentation(ModelTrainer,nn.Module): 
    def get_trainable_parmas(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name) 
        return [p for p in self.parameters() if p.requires_grad]

    def get_model_params(self):
        return self.model.state_dict()
    
    def calculate_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)  
    
    def get_bn_var(self):
        names=[name for i,(name,params)in enumerate(self.model.named_parameters())]
        run_var_idx=list()
        for idx,name in enumerate(names):
            if 'running_mean' in name or 'running_var' in name:
                run_var_idx.append(idx) 
        return run_var_idx
        
    def set_running_stat_grads(self):
        for m in self.model.modules():
            if isinstance(m,MyBatchNorm2d):
                m.set_running_stat_grads()


    def change_bn(self,mode='grad',prior=0):
        model=self.model
        if mode =='grad':
            model.encoder1[1]=MyBatchNorm2d(model.encoder1[1])
            model.encoder1[4]=MyBatchNorm2d(model.encoder1[4])
            model.encoder2[1]=MyBatchNorm2d(model.encoder2[1])
            model.encoder2[4]=MyBatchNorm2d(model.encoder2[4])
            model.encoder3[1]=MyBatchNorm2d(model.encoder3[1])
            model.encoder3[4]=MyBatchNorm2d(model.encoder3[4])
            model.encoder4[1]=MyBatchNorm2d(model.encoder4[1])
            model.encoder4[4]=MyBatchNorm2d(model.encoder4[4])
            model.decoder1[1]=MyBatchNorm2d(model.decoder1[1])
            model.decoder1[4]=MyBatchNorm2d(model.decoder1[4])
            model.decoder2[1]=MyBatchNorm2d(model.decoder2[1])
            model.decoder2[4]=MyBatchNorm2d(model.decoder2[4])
            model.decoder3[1]=MyBatchNorm2d(model.decoder3[1])
            model.decoder3[4]=MyBatchNorm2d(model.decoder3[4])
            model.decoder4[1]=MyBatchNorm2d(model.decoder4[1])
            model.decoder4[4]=MyBatchNorm2d(model.decoder4[4])
            model.bottleneck[1]=MyBatchNorm2d(model.bottleneck[1]) 
            model.bottleneck[4]=MyBatchNorm2d(model.bottleneck[4])
        names=[name for i,(name,params)in enumerate(model.named_parameters())]
        print(names)


    def update_memory(self,model,args,mode,x,labels,queue):
        if mode=='train':
            with torch.no_grad():
                x=x.cuda()
                _,features=model(x,'pixel_contrast')
                features=features.permute(0,2,3,1)
                batch_size=features.shape[0]
                feat_dim=features.shape[-1]
                features=features.view(batch_size,-1,feat_dim)
                features=F.normalize(features,dim=-1)
                labels=labels.view(batch_size,-1)
                self.queue.push(features,labels)
        elif args.pretrain:
            with torch.no_grad():
                x=x.cuda()
                _,features=model(x,'pixel_contrast')
                features=features.permute(0,2,3,1)
                batch_size=features.shape[0]
                feat_dim=features.shape[-1]
                features=features.view(batch_size,-1,feat_dim)
                features=F.normalize(features,dim=-1)
                labels=labels.view(batch_size,-1)
                queue.push(features,labels)

        
    def train(self, train_data, device, args,queue,Squeue=None,rnd=None):
        """
        use three levels contrast learning to apply federated learning
        
        """
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = DiceLoss().to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, amsgrad=True) 

        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.local_epoch): 
            batch_loss = []
            batch_acc = []
            for batch_idx, (idx, samples) in enumerate(train_data):
                model.zero_grad() 
                x=samples['image']
                labels=samples['label']
                x, labels = x.to(device), labels.to(device)
                log_probs,feats= model(x,"pixel_contrast")
                preds=nn.functional.softmax(log_probs,dim=1)
                preds_v2=turn_prediction(2,preds) 
                ###Dice loss
                loss_dice = criterion(log_probs, labels)

                Contrast_MemBank=MemContrastLoss(self.args).to(device)
                loss_contrast_lmem=Contrast_MemBank(feats,predict=preds_v2,queue=queue,labels=labels)

                ###server memory loss
                Contrast_SMemBank=ServerPixelContrastLoss(self.args)
                loss_contrast_smem=Contrast_SMemBank(feats,predict=preds_v2,queue=Squeue,labels=labels)

                ###minibatch memory loss
                Contrast_MemBatch=MiniBContrastLoss(self.args)
                loss_contrast_bmem=Contrast_MemBatch(feats,labels=labels,predict=preds_v2)

                acc = DiceLoss().dice_coef(log_probs, labels).item() 

                ###total_loss
                coefficient=float(2*rnd)/float(args.comm_round)
                loss=loss_dice+0.01*loss_contrast_lmem*coefficient+0.01*loss_contrast_bmem*coefficient+0.02*loss_contrast_smem*coefficient
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_acc.append(acc)

                    ###update memory queue
                self.update_memory(self.model,args,None,x=x,labels=labels,queue=queue)
                
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            logging.info('Client Index = {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                self.id, epoch, sum(epoch_acc) / len(epoch_acc),sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args, ood=False):
        model = copy.deepcopy(self.model)

        model.to(device)
        if ood:
            model.train()
        else:
            model.eval()

        metrics = {
            'test_acc': 0,
            'test_loss': 0,
        }

        criterion = DiceLoss().to(device)
        feas=[]
        cnt=0
        times=[]
        with torch.no_grad():  
            for batch_idx, (idx, samples) in enumerate(test_data):
                x=samples['image']
                target=samples['label']
                x = x.to(device)
                target = target.to(device)
                cnt+=x.shape[0]
                pred = model(x)
                loss = criterion(pred, target)
                acc = DiceLoss().dice_coef(pred, target).item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_acc'] += acc
        metrics["test_loss"] = metrics["test_loss"] / len(test_data)
        metrics["test_acc"] = metrics["test_acc"] / len(test_data)
        return metrics
    
