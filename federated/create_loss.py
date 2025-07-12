import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def turn_gt(num_class,gt):  
    turn_gt=torch.empty(gt.shape[0],0,gt.shape[2],gt.shape[3],device=gt.device)
    for i in range(num_class):
        each_gt=torch.zeros_like(gt)
        each_gt[gt==i]=1
        turn_gt=torch.cat((turn_gt,each_gt),dim=1)
    return turn_gt

def turn_prediction(num_class,pred):
    """
    turn the dimension from [batch_size,num_class,..,..] to [batch_size,...,...]
    
    """
    background_channel = pred[:, 0, :, :]
    foreground_channel = pred[:, 1, :, :]
    turned_pred=torch.zeros(pred.shape[0],pred.shape[2],pred.shape[3])
    turned_pred[foreground_channel>background_channel]=1
    return turned_pred 
    

def stable_softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=1, keepdim=True)

def unspv_consistency_loss_fea(feat_w,feat_s):
    """
    input:feat_w,feat_s
    aim: regularize feat_w and feat_s
    
    """
    assert feat_w.size()==feat_s.size()
    mse_loss=(feat_w-feat_s)**2
    return mse_loss

class spv_loss(nn.Module):
    def __init__(self,smooth=1.0,activation='sigmoid'):
        super(spv_loss,self).__init__()
        self.smooth=smooth 
        self.activation=activation 
        
    
    def turn_gt(self,pred,gt): 
        num_class=pred.shape[1]
        turn_gt=torch.empty(gt.shape[0],0,gt.shape[2],gt.shape[3])
        for i in range(num_class):
            each_gt=torch.zeros_like(gt)
            each_gt[gt==i]=1
            turn_gt=torch.cat((turn_gt,each_gt),dim=1)
        return turn_gt

    def dice_coef(self,pred,gt):
        """ computational formula
        """
       
        softmax_pred = torch.nn.functional.softmax(pred, dim=1) 
        seg_pred = torch.argmax(softmax_pred, dim=1) 
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1 

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1) 
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1) 
            dice = (2. *  intersection )/ (union + 1e-5) 
         
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class
    
    def forward(self, pred, gt):
        sigmoid_pred=stable_softmax(pred)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]  ####2
        
        # conver label to one-hot
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1 
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1 #
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1 
        label = torch.cat([bg, label1, label2], dim=1)  
        
        loss = 0
        smooth = 1e-5


        for i in range(num_class):  
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...]) 
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss
    
    
    def spv_entropy_loss(self,pred,gt,c=3):
        """ computational formula
        """
        turn_gt=self.turn_gt(pred,gt)
        pred=F.softmax(pred,dim=1)  
        y1=-1*torch.sum(turn_gt*torch.log(pred+1e-6),dim=0)/torch.tensor(np.log(c)).cuda()
        spv_ent=torch.mean(y1)
        return spv_ent


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        """ computational formula
        """
       
        softmax_pred = torch.nn.functional.softmax(pred, dim=1) 
        seg_pred = torch.argmax(softmax_pred, dim=1)  
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred) 
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1) 
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1) 
            dice = (2. *  intersection )/ (union + 1e-5) 
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)  
        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1] 
        
        # conver label to one-hot
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1 
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1 
        label = torch.cat([bg, label1, label2], dim=1)  
        
        loss = 0
        smooth = 1e-5


        for i in range(num_class):  
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...]) 
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss 




