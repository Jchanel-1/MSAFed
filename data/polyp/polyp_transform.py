from torchvision import transforms
import numpy as np
from torchvision.transforms import ColorJitter, GaussianBlur
import torch

from PIL import ImageFilter

class Transform_polyp(object):
    def __init__(self,args,color_jitter_prob=0.8,gaussian_blur_prob=0.8,sigma=(0.1,2.0),horizontalflip_prob=0.5): 
        self.color_jitter_prob=color_jitter_prob
        self.gaussian_blur_prob=gaussian_blur_prob
        self.sigma=sigma
        self.horizontalflip_prob=horizontalflip_prob
        print("the prob of color_jitter_prob is {} ".format(self.color_jitter_prob))
        print("the prob of gaussian_blur_prob is {} ".format(self.gaussian_blur_prob))
        print("the parameters of color_jitter are 0.4 0.4 0.4 0.1")
        print("the prob of horizontalflip is {}".format(self.horizontalflip_prob))
        print("the kernel size for Gaussian is {}".format('5'))
        print("the sigma for Gaussian is {}".format('0.1~2.0'))


    def __call__(self,images,type):
        if type=='strong':
            images_s=images.clone()
            for idx,img in enumerate(images_s):
                if np.random.random()<self.color_jitter_prob:
                    Color_transform=ColorJitter(0.4,0.4,0.4,0.1)  
                    img=Color_transform(img)

                if np.random.random()<self.gaussian_blur_prob:
                    radius=np.random.uniform(0.1,2.0)
                    GaussianBlur_transforom=GaussianBlur(kernel_size=5,sigma=radius)
                    img=GaussianBlur_transforom(img)
            images_s,flip_mask=self.transforms_for_flip(images_s)
            images_s,rot_mask=self.transforms_for_rot(images_s)

            return images_s,flip_mask,rot_mask

        elif type=="weak":
            images_w=images.clone()
            images_w,flip_mask=self.transforms_for_flip(images_w) 
            images_w,rot_mask=self.transforms_for_rot(images_w)
            return images_w,flip_mask,rot_mask

        else:
            assert False,"Invalid type:{}, not in strong or weak".format(type)
    
    def transforms_for_flip(self,images):
        flip_mask=np.random.randint(0,2,images.shape[0])
        for idx in range(images.shape[0]):
            if flip_mask[idx]==1:
                images[idx]=torch.flip(images[idx],[1])
        return images,flip_mask

    def transforms_for_rot(self,images):
        rot_mask = np.random.randint(0, 4, images.shape[0])
        for idx in range(images.shape[0]):
            images[idx]=torch.rot90(images[idx],int(rot_mask[idx]),dims=[1,2])
        return images,rot_mask

    def transforms_back_spatial(self,preds,rot_mask, flip_mask):

        for idx in range(preds.shape[0]):

            preds[idx] = torch.rot90(preds[idx], int(rot_mask[idx]), dims=[2,1])

            if flip_mask[idx] == 1:
                preds[idx] = torch.flip(preds[idx], [1])

        return preds


    
