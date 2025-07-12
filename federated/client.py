from email.mime import base
import logging
import torch
import os
import copy
from copy import deepcopy
from .create_loss import turn_prediction

from .create_loss import DiceLoss
from data.polyp.polyp_transform import Transform_polyp
from .create_loss import unspv_consistency_loss_fea
import torch.nn.functional as F
import torch.nn as nn
from .create_loss import turn_gt
from  .Queue import Memory
from .create_loss import turn_prediction
from .loss_contrast_memv2 import MemContrastLoss
from .loss_contrast_batch import MiniBContrastLoss
import torch.nn.functional as F
import numpy as np
from .loss_contrast_server_mem import ServerPixelContrastLoss

class Client:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device,
                 model_trainer,train_len=None):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        
        if args.ood_test!=True:
            if client_idx !=-1: 
                self.train_len=train_len
                assert self.train_len!=None,"train_len is {}".format(self.train_len) 
                self.queue=Memory(num_sample=self.train_len[client_idx],args=args) 
        else:
            self.queue=Memory(num_sample=args.num_cluster*5,args=args) 
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.prev_weight = None
        self.lr=args.lm_lr  
        self.transform=Transform_polyp(args) 



    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx) 
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data 
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def _fill_memory(self,model,args,mode):
        if mode=='train' or 'vis_feature':
            with torch.no_grad():
                for batch_idx,(idx,samples) in enumerate(self.local_training_data):
                    x=samples['image']
                    labels=samples['label']
                    x=x.to(self.device)
                    _,features=model.model(x,'pixel_contrast')
                    features=features.permute(0,2,3,1)
                    batch_size=features.shape[0]
                    feat_dim=features.shape[-1]
                    features=features.view(batch_size,-1,feat_dim)
                    features=F.normalize(features,dim=-1)
                    labels=labels.view(batch_size,-1)
                    self.queue.push(features,labels)


    def update_memory(self,model,args,mode,x,labels):
        if mode=='train':
            with torch.no_grad():
                x=x.to(self.device)
                _,features=model.model(x,'pixel_contrast')
                features=features.permute(0,2,3,1)
                batch_size=features.shape[0]
                feat_dim=features.shape[-1]
                features=features.view(batch_size,-1,feat_dim)
                features=F.normalize(features,dim=-1)
                labels=labels.view(batch_size,-1)
                self.queue.push(features,labels)
        elif args.pretrain:
            with torch.no_grad():
                x=x.to(self.device)
                _,features=model.model(x,'pixel_contrast')
                features=features.permute(0,2,3,1)
                batch_size=features.shape[0]
                feat_dim=features.shape[-1]
                features=features.view(batch_size,-1,feat_dim)
                features=F.normalize(features,dim=-1)
                labels=labels.view(batch_size,-1)
                self.queue.push(features,labels)             


    def local_train(self,model,adapt_lrs,args,mode='train',Squeue=None,rnd=None,pre_batch_acc=None):
        batch_acc,batch_loss=[],[]
        state=deepcopy(model.model.state_dict())  
        model.to(self.device)
        model.model.load_state_dict(state)

        if self.queue.get_size()==0:
            if mode=='train' :
                self._fill_memory(model,args,mode) 

            elif mode=='test' :
                self._fill_memory_tta(args) 

            

        if mode=='train':
            num_data=len(self.local_training_data) 
            for batch_idx,(idx,samples) in enumerate(self.local_training_data):
                x=samples['image']
                labels=samples['label']
                x,labels=x.to(self.device),labels.to(self.device)

                ##### 1.unsupervised adaptation
                unspv_grad=self.adapt_one_step(model.model,adapt_lrs,x,labels,args,Squeue=Squeue)
                self.update_memory(model,args,mode,x,labels) 
                
                ##### 2.supervised adaptation
                model.model.eval()
                pred=model.model(x,'main')
                criterion = DiceLoss().to(self.device)
                spv_loss=criterion(pred,labels)
                spv_grad=torch.autograd.grad(spv_loss,model.model.trainable_params(),allow_unused=True)

                ##### 3.update the adaptation rate
                with torch.no_grad():
                    g = torch.zeros_like(adapt_lrs).to(self.device)
                    l = torch.zeros_like(adapt_lrs).to(self.device)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()
                        l[i] += g1.numel()

                    g /= torch.sqrt(l)
                    adapt_lrs += self.lr* g
                with torch.no_grad():
                    batch_loss.append(spv_loss.item())
                    batch_acc.append(DiceLoss().dice_coef(pred,labels).item())
            avg_loss=sum(batch_loss)/len(batch_loss)
            avg_acc=sum(batch_acc)/len(batch_acc)
            return avg_loss,avg_acc,num_data 

        if mode=='test':
            """
            finetune on the test image without any annotations
            """
            num_data=self.local_sample_number
            count=0
            for batch_idx,(x,labels) in enumerate(self.local_test_data):
                x,labels=x.to(self.device),labels.to(self.device)
                ##### 1.unsupervised adaptation
                unspv_grad=self.tta_adapt_one_step(model.model,adapt_lrs,x,labels,args,rnd=rnd)
                count+=1

                model.model.eval()
                pred=model.model(x,'main')
                criterion = DiceLoss().to(self.device)
                spv_loss=criterion(pred,labels)
                b_acc=DiceLoss().dice_coef(pred,labels).item()
                    
                with torch.no_grad():
                    batch_loss.append(spv_loss.item())
                    batch_acc.append(b_acc)
            avg_loss=sum(batch_loss)/len(batch_loss)
            avg_acc=sum(batch_acc)/len(batch_acc)
            return avg_loss,avg_acc,num_data,batch_acc 

    def adapt_one_step(self,model,adapt_lrs,x,labels,args,Squeue=None):
        model.eval()
        x_s,flip_mask_s,rot_mask_s=self.transform(x,"strong")
        x_w,flip_mask,rot_mask=self.transform(x,"weak")
        x=x.cpu()
        Contrast_memory=MemContrastLoss(self.args).to(self.device)
        Contrast_anchor=MiniBContrastLoss(self.args).to(self.device)
        Contrast_Smemory=ServerPixelContrastLoss(self.args).to(self.device)
        logits_s,feat_s=model(x_s,'pixel_contrast')
        logits_w,feat_w=model(x_w,'pixel_contrast') 
        preds_w=nn.functional.softmax(logits_w,dim=1) 
        preds_w_v2=turn_prediction(2,preds_w)
        ###local memory loss
        loss_contrast_mem=Contrast_memory(feat_w,predict=preds_w_v2,queue=self.queue)
        
        ###current minibatch loss
        loss_contrast_anc=Contrast_anchor(feat_w,predict=preds_w_v2)
        
        ###server memory loss
        loss_contrast_smem=Contrast_Smemory(feat_w,predict=preds_w_v2,queue=Squeue)

        ### consistency(bottleneck)

        feat_w=self.transform.transforms_back_spatial(feat_w,rot_mask,flip_mask)
        feat_s=self.transform.transforms_back_spatial(feat_s,rot_mask_s,flip_mask_s)
        unspv_cons_loss=unspv_consistency_loss_fea(feat_w,feat_s)


        ### CE
        ce=nn.CrossEntropyLoss().to(self.device) 
        #weak_ce=ce(preds_w,refined_label)
        weak_ce=ce(preds_w,preds_w)
    
        unspv_loss=self.args.coef_cons*torch.mean(unspv_cons_loss)+weak_ce+self.args.coef_contra*loss_contrast_anc+self.args.coef_contra*loss_contrast_smem+self.args.coef_contra*loss_contrast_mem
        unspv_loss.backward()
        model.set_running_stat_grads()
        unspv_grad=[p.grad.clone() for p in model.trainable_params()] 

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.trainable_params(), unspv_grad)):
                p -= adapt_lrs[i] * g
        model.zero_grad()
        model.clip_bn_running_vars()
        return unspv_grad


    def train(self, w_global,Squeue=None,rnd=None):
        self.model_trainer.set_model_params(w_global) 
        self.model_trainer.to(self.device)
        if self.queue.get_size()==0:
            self._fill_memory(self.model_trainer,self.args,'train') 
        
        self.model_trainer.train(self.local_training_data, self.device, self.args,self.queue,Squeue=Squeue,rnd=rnd) 
        weights = self.model_trainer.get_model_params()         
        return weights

    def local_validate(self, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        
        metrics = self.model_trainer.test(self.local_val_data, self.device, self.args)
        return metrics

    def local_test(self, b_use_test_dataset, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)

        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
    