import json
import cv2
import numpy as np

from torch.utils.data import Dataset

import random
import torch
from torchvision.datasets import CocoCaptions, VisionDataset
from einops import rearrange
from saicinpainting.training.data.masks import get_mask_generator
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import os, sys
from torchvision import transforms
sys.path.append(os.getcwd())

from ldm.data.coco_transforms import create_transforms
from PIL import Image

class visualization_Dataset(Dataset):
    def __init__(self,dataset_path='/home/cn/inpainting_examples/data'):
        transform_type="imagenet_val"
        image_resolution=256
        split='val'
        self.transform = create_transforms(transform_type, image_resolution, split, is_eval=False)
        
        self.mask_path='/home/cn/inpainting_examples/mask'
        self.dataset_path=dataset_path

        with open("/home/cn/inpainting_examples/caption_2.json") as f:
            data_dict= json.load(f)
        
        self.dataset=[]
        for key,value in data_dict.items():
            self.dataset.append([key,value])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        #读入image与对应的text
        i_name, text = self.dataset[item][0],self.dataset[item][1]
        img_name=i_name
        mask_name=i_name[:-4]+'_mask.png'

        img=Image.open(os.path.join(self.dataset_path,img_name))
        if img.mode!='RGB':
            img=img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        mask_path=os.path.join(self.mask_path,mask_name)

        mask= Image.open(mask_path).resize((256,256))
        mask = np.array(mask.convert("L"))

        #把mask转换为灰度图像，0表示黑，255表示白
        mask = mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        mask=torch.unsqueeze(mask,0)
        #这个mask是载入进来的，不是随机生成的
        
        # to align ldm's input   这个更改维度之后变成[256,256,3]了，为什么是这个形状呢？
        target=img
        #target不用动，target会完整的输入输出，需要修改的是作为hint的source
        #这个mask操作的时候要先经过一个处理，不然会mask到错误的地方，刚刚已经看到了

        source=(1-mask)*img
        #source=torch.cat((source, mask), dim=0)

        
        target=rearrange(img, "c h w -> h w c")  
        source=rearrange(source, "c h w -> h w c")

        
        #plt.imshow(source)
        #plt.savefig('source.jpg')  
        
        return dict(jpg=target, txt=text, hint=source, mask=mask)


if __name__=='__main__':
    dataset=visualization_Dataset()
    data = dataset.__getitem__(1)
    print(data['mask'].shape)


'''
第一次测量的fid是26.564  medium的是26.564   26.44第二次
可以看看第二次测量的是多少
测一下thin的指标是多少
36.07072289879761
36.37
python -m pytorch_fid  testing/Imagenet_thin  training/image_mask/resized_image/    --dims 2048 --device cuda:3 --batch-size 100
python -m pytorch_fid  /home/cn/ControlNet/testing/after_postprocess/mscoco/thin  training/mscoco/crop_coco_image    --dims 2048 --device cuda:3 --batch-size 100

cldm  
un_concat medium 32.9723
un_thin

bld
medium
thin

'''