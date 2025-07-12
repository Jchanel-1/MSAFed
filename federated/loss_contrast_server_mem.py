from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

class ServerPixelContrastLoss(nn.Module, ABC):
    def __init__(self, args):
        super(ServerPixelContrastLoss, self).__init__()
        self.args=args
        self.temperature=self.args.temperature_contrast
        self.base_temperature=self.args.base_temperature

        self.ignore_label = -1
        self.max_samples=self.args.max_samples
        self.max_views=self.args.max_views
        self.mode=self.args.ood_test

    def _anchor_sampling(self,X,y_hat):
        """
        random sample anchor based on prediction for SSF task
        """      
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii] 
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views] 
            classes.append(this_classes)
            total_classes += len(this_classes) 

        if total_classes == 0:
            return None, None
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)       
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()                     
        X_ptr = 0 
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]  
            this_classes = classes[ii]  

            for cls_id in this_classes: 
                indices=(this_y_hat == cls_id).nonzero()
                num_indices=indices.shape[0]
                if num_indices>n_view: 
                    num_indices_keep=n_view
                else:
                    num_indices_keep=num_indices             
                perm=torch.randperm(num_indices)
                sampled_indices=indices[perm[:num_indices_keep]]
                X_[X_ptr, :, :] = X[ii, sampled_indices, :].squeeze(1)
                X_[X_ptr,]
                y_[X_ptr] = cls_id 
                X_ptr += 1
        X_=F.normalize(X_,dim=-1)
        return X_, y_

    def _hard_anchor_sampling(self, X, y_hat, y): ##sample anchor
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y) 
            this_classes = [x for x in this_classes if x != self.ignore_label] 
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views] 

            classes.append(this_classes)
            total_classes += len(this_classes) 

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes 
        n_view = min(n_view, self.max_views)       

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda() 
        y_v2=torch.zeros((total_classes,n_view),dtype=torch.float).cuda()
        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]  
            this_y = y[ii]          
            this_classes = classes[ii] 
            for cls_id in this_classes: 
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero() 

                num_hard = hard_indices.shape[0] 
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view/2  and num_easy >= n_view/2 :
                    num_hard_keep = n_view //2
                    num_easy_keep = n_view -num_hard_keep
                elif num_hard >= n_view /2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view-num_easy_keep
                elif num_easy >= n_view/2 :
                    num_hard_keep = num_hard
                    num_easy_keep = n_view-num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard) 
                hard_indices = hard_indices[perm[:num_hard_keep]] 
                perm = torch.randperm(num_easy) 
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                indices=indices.squeeze(1)
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                X_[X_ptr,]
                y_v2[X_ptr,:]=this_y[indices]
                X_ptr += 1

        X_=F.normalize(X_,dim=1)
        return X_, y_v2

    def _sample_negative(self, Q):
        class_num=self.args.num_class
        cache_size=self.args.num_cluster*len(self.args.source) 
        feat_size=Q.memory[0][0].shape[0]
        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            this_q =Q.get_class_elements(ii)
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1] 

        y_anchor = y_anchor.contiguous().view(-1, 1) 
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None and queue.get_size()!=0:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        elif queue.get_size()==0:
            return 0
        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()
    
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature) 
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits=anchor_dot_contrast-logits_max.detach()
        if not self.args.pretrain: 
            mask = mask.repeat(anchor_count, contrast_count)
        pos_mask=mask
        neg_mask=1-pos_mask
        pos_logits=logits*pos_mask
        pos_exp_logits=torch.exp(pos_logits)
        neg_logits=logits*neg_mask
        neg_exp_logits=torch.exp(neg_logits)
        neg_exp_logits_sum=neg_exp_logits.sum(1,keepdim=True)
        log_prob=pos_logits-torch.log(pos_exp_logits+neg_exp_logits_sum+1e-10)
        mean_log_prob=log_prob.sum(1)/pos_mask.sum(1)
        loss=-(self.temperature / self.base_temperature)*mean_log_prob
        loss=loss.mean()
        assert not torch.isnan(loss), 'Server memory loss is nan'
        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        if self.args.pretrain: 
            labels = labels.unsqueeze(1).float().clone()
            labels = labels.squeeze(1).long()
            assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
            batch_size = feats.shape[0]
            labels = labels.contiguous().view(batch_size, -1)
            predict = predict.contiguous().view(batch_size, -1).cuda()
            feats = feats.permute(0, 2, 3, 1)
            feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])      
            feats_, labels_ = self._hard_anchor_sampling(feats, predict, labels) 
            loss = self._contrastive(feats_, labels_, queue=queue)       


        elif self.args.train: 
            batch_size = feats.shape[0]

            predict = predict.contiguous().view(batch_size, -1)
            feats = feats.permute(0, 2, 3, 1)
            feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
            feats_,labels_=self._anchor_sampling(feats,predict)
            loss = self._contrastive(feats_, labels_, queue=queue) 
        return loss