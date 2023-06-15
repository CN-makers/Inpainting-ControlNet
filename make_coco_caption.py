from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from imagenet_test_dataset import ImageNet_test_Dataset
from coco_datsaset import CocoTrainValid
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
import torch 
import numpy as np
from PIL import Image
from einops import rearrange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


N=50
save_image_path='testing/Imagenet_thin'

#dataset = ImageNet_test_Dataset(mode='thin')
dataset = CocoTrainValid(
        root="training/mscoco", split="valid", image_resolution=256, 
        transform_type="imagenet_val", is_eval=False,mode='thin')

coco_caption_list=[]
test_loader = DataLoader(dataset, num_workers=0, batch_size=N, shuffle=False,drop_last=False)
for cnt,batch in enumerate(test_loader):
    target,txt,source,mask=batch['jpg'],batch['txt'],batch['hint'],batch['mask']
    coco_caption_list.extend(txt)

coco_caption_list=np.array(coco_caption_list)
np.save('training/mscoco/coco_caption_list.npy',coco_caption_list)

tmp=np.load('training/mscoco/coco_caption_list.npy')


print(tmp)