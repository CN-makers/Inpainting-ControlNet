import torch
import os
import json
from torchmetrics.multimodal import CLIPScore
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from einops import rearrange

image_dict={
    'cldm_thin':'testing/Imagenet_thin',
    'cldm_medium':'testing/Imagenet_thin',
    'bld_medium':'/home/cn/blended-latent-diffusion/outputs/edit_results/samples/images',
    'bld_thin':'/home/cn/blended-latent-diffusion/outputs/edit_results/samples/images',
    'bd_medium':'/home/cn/blended-diffusion/output/medium',
    'real_score':'training/image_mask/resized_image',
    'after_postprocess_medium':'testing/after_postprocess/thin'
}

transforms_ = [
            transforms.ToTensor(),
        ]
transforms_ = transforms.Compose(transforms_)
def get_clip_score_imagenet(image_path=image_dict['real_score']):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score_list=[]

    with open("training/imagenet_class.json") as f:
        data_dict= json.load(f)
    data_label=[]
    for k,v in data_dict.items():
        data_label.append(v)

    image_name=os.listdir(image_path)
    image_name=sorted(image_name,key=lambda x:(x[15:-12]))

    #image_name=sorted(image_name,key=lambda x:int(x[0:-4]))

    for i,sub_name in enumerate(tqdm( image_name)):
        img=Image.open(os.path.join(image_path,sub_name))    
        #img=np.array(img)
        #img=torch.from_numpy(img)
        #img=rearrange(img, "h w c -> c h w")
        img=transforms_(img)

        score_list.append(  float(metric(img, 'A photo of '+ data_label[i])))
    #就是通过metric的CLIPScore来判断这个clip-score了
    return sum(score_list)/len(score_list)

def get_clip_score_coco(image_path='/home/cn/blended-diffusion/output/coco_medium'):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score_list=[]

    #image_path='training/mscoco/crop_coco_image'
    data_label=np.load('training/mscoco/coco_caption_list.npy')

    image_name=os.listdir(image_path)

    image_name=sorted(image_name,key=lambda x:int(x[0:-4]))

    for i,sub_name in enumerate(tqdm( image_name)):
        img=Image.open(os.path.join(image_path,sub_name))    
        #img=np.array(img)
        #img=torch.from_numpy(img)
        #img=rearrange(img, "h w c -> c h w")
        img=transforms_(img)

        score_list.append(  float(metric(img,data_label[i])))
    #就是通过metric的CLIPScore来判断这个clip-score了
    return sum(score_list)/len(score_list)


if __name__=="__main__":
    image_net_score=get_clip_score_imagenet()
    print(image_net_score)



'''
要不然再开个文件夹再专门
29.0073

ImageNet的CLIP——score:
medium 29.855
thin 29.7974
real 30.216926

BLDM  
medium  29.077
thin 

BLEND DIFFUSION
medium
thin

不加prompt前缀的时候，相似度是29.098
加了prompt前缀的时候，相似度是30.2205

coco 1000  
medium 30.338679023742674
thin 30.305341381073
real 27.2799
这说明编辑后的图像比原图像更贴近于caption

python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/random_thin_256.yaml /home/cn/lama/mscoco/val2017_generation/  /home/cn/lama/mscoco/generation_test/random_thin_256
python -m pytorch_fid  testing/coco/medium  training/mscoco/crop_coco_image    --dims 2048 --device cuda:3 --batch-size 100
python -m pytorch_fid  testing/after_postprocess/medium  training/image_mask/resized_image/    --dims 2048 --device cuda:3 --batch-size 100
'''