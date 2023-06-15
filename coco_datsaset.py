import random
import torch
from torchvision.datasets import CocoCaptions, VisionDataset
from einops import rearrange
from saicinpainting.training.data.masks import get_mask_generator
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from torchvision import transforms
sys.path.append(os.getcwd())

from ldm.data.coco_transforms import create_transforms

mask_generator_kind='mixed'
f = open('models/coco_mask.yaml', 'r')
ystr = f.read()
mask_gen_kwargs=yaml.load(ystr,Loader=yaml.FullLoader)
# print(mask_gen_kwargs)

class CocoTrainValid(VisionDataset):
    splits = {'valid', 'train'}
    def __init__(self, root, split, image_resolution, transform_type=None, is_eval=False,mode='medium'):
        assert split in self.splits, f'{split} is not in {self.splits}'
        
        assert transform_type in {"dalle", "dalle-vqvae", "clip", "clip-dvae", "none", "imagenet_train", "imagenet_val"}
        transform = create_transforms(transform_type, image_resolution, split, is_eval)
        super().__init__(root, transform=transform)

        self.split = split
        self.mask_generator=get_mask_generator(kind=mask_generator_kind,kwargs=mask_gen_kwargs)
        self.iter_i=0
        #给出thin_mask与medium_mask的路径
        self.mode=mode
        self.medium_path='training/image_mask/random_medium'
        self.thin_path='training/image_mask/random_thin'

        if split == "valid":
            self.dataset = CocoCaptions(root=f'{self.root}/coco_image_sample', annFile=f'{self.root}/annotations/captions_val2017.json')
        else:
            self.dataset = CocoCaptions(root=f'{self.root}/train2014', annFile=f'{self.root}/annotations/captions_train2014.json')

        if split=='valid':
            if self.mode=='medium':
                self.mask_list=sorted(os.listdir(self.medium_path))
            elif self.mode=='thin':
                self.mask_list=sorted((os.listdir(self.thin_path)))
    


    def __len__(self):
        # return 100  # for quick debug
        if self.split=='valid':
            return 1000
        else:
            return len(self.dataset)

    def __getitem__(self, item):
        img, text = self.dataset[item]
        
        if self.transform:
            img = self.transform(img)

        # text = ' '.join(text)  # text is a list of sentences. Concat them.
        #这个有用吗？
        if self.split == 'train':
            rnd_txt = random.randint(0, len(text)-1)
            text = text[rnd_txt]
            mask=self.mask_generator(img,iter_i=self.iter_i)
            mask = torch.from_numpy(mask)
        elif self.split=='valid':
            
            text = text[0]
            mask_name=self.mask_list[self.iter_i]

            if self.mode=='medium':
                mask_path=os.path.join(self.medium_path,mask_name)
            elif self.mode=='thin':
                mask_path=os.path.join(self.thin_path,mask_name)

            mask= Image.open(mask_path)
            mask = np.array(mask.convert("L"))
            #把mask转换为灰度图像，0表示黑，255表示白
            mask = mask.astype(np.float32) / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)
            mask=torch.unsqueeze(mask,0)
        # to align ldm's input   这个更改维度之后变成[256,256,3]了，为什么是这个形状呢？
        target=img
        #target不用动，target会完整的输入输出，需要修改的是作为hint的source
        #这个mask操作的时候要先经过一个处理，不然会mask到错误的地方，刚刚已经看到了
        
        source=(1-mask)*img
        #source=torch.cat((source, mask), dim=0)

        self.iter_i+=1
        
        target=rearrange(img, "c h w -> h w c")  
        source=rearrange(source, "c h w -> h w c")

        
        #plt.imshow(source)
        #plt.savefig('source.jpg')  
        
        return dict(jpg=target, txt=text, hint=source, mask=mask)


'''
        tokenizer_config={
            "target": "modules.clip_text_encoder.my_tokenizer.my_tokenize.Tokenize",
            "params": {
                "context_length": 77,
                "add_start_and_end": True,
                "with_mask": True,
                "pad_value": 0,
                "clip_embedding": False,
                "tokenizer_config": {
                     'target': 'modules.clip_text_encoder.clip.simple_tokenizer.SimpleTokenizer',
                     'params':{
                        'end_idx': 49152 # 16384 fo DALL-E
                        },
                }
            }
        },
'''

if __name__ == "__main__":
    dataset = CocoTrainValid(
        root="training/mscoco", split="train", image_resolution=256, 
        transform_type="imagenet_val", is_eval=False,)

    data = dataset.__getitem__(0)
    print(dataset.__len__())
    print(data['hint'].shape)
    print(data['mask'].shape)
    my_mask=data['mask'].squeeze(0)
    convert_mask=1-my_mask
    plt.imshow(my_mask,plt.cm.gray)
    plt.savefig('test.jpg')

    plt.imshow(convert_mask,plt.cm.gray)
    plt.savefig('try.jpg')


    
    '''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)

    for i, data in enumerate(dataloader):
        print(data["caption"])
        print(data["raw_text"])
        
        import torchvision
        torchvision.utils.save_image(data["image"], "image.png", normalize=True)
        exit()
    '''