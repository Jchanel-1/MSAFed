
"""
Loss for brain segmentaion (not used)
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
import numpy as np



def entropy_loss(p, c=3):
    # p N*C*W*H*D
     p = F.softmax(p, dim=1)  ###对维度1进行softmax     ###[5 2 384 384]  经过softmax得到预测概率
     y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) / torch.tensor(np.log(c)).cuda()  ###加上10^-6确保不会出现log0的情况
     ent = torch.mean(y1)
     return ent

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        """ computational formula
        """
       
        softmax_pred = torch.nn.functional.softmax(pred, dim=1) ###对每个样本的预测结果进行softmax
        seg_pred = torch.argmax(softmax_pred, dim=1)   ##返回最大的预测值对应的类别
        all_dice = 0
        gt = gt.squeeze(dim=1)###抽取维度为1的张量
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred) ##创建一个与seg_pred形状相同的全0张量
            each_pred[seg_pred==i] = 1 ###这边是两分类任务?  所以相当于把预测值

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1) ##算交集数量
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1) ###全部元素都是0和1了，so简单在每个样本上求和其实就行
            dice = (2. *  intersection )/ (union + 1e-5)  ###防止除0 
         
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class#,seg_pred  ####除以num_class 求平均dice ，即每个类别的平均dice


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)  ###[5,2,384,384]
        

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]  ####2
        
        # conver label to one-hot
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1 ###background
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1 ###标签1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1 ###这边应该全0 对于前列腺分割而言，只有一个label
        label = torch.cat([bg, label1, label2], dim=1)  ###由背景、label1、label2拼接成的单热编码
        
        loss = 0
        smooth = 1e-5


        for i in range(num_class):  ##计算每一个dice_loss
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...]) 
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss
        
