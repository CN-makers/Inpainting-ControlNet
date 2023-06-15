from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from imagenet_test_dataset import ImageNet_test_Dataset
from coco_datsaset import CocoTrainValid
from visualization_dataset import visualization_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
import torch 
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision import transforms
from ldm.util import  default

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_model('./models/cldm_v21_editing.yaml').cpu()
model.load_state_dict(load_state_dict('lightning_logs/version_70/checkpoints/epoch=17-step=11645.ckpt', location='cuda'))
model = model.to(device)
N=8
save_image_path='testing/visualization_2'

dataset=visualization_Dataset()
#dataset = ImageNet_test_Dataset(mode='thin')
#dataset = CocoTrainValid(
#        root="training/mscoco", split="valid", image_resolution=256, 
#        transform_type="imagenet_val", is_eval=False,mode='thin')

test_loader = DataLoader(dataset, num_workers=0, batch_size=N, shuffle=False,drop_last=False)
#从log_image开始入手，到samplers
with torch.no_grad():
    #读入dataloader
    #对dataloader的数据做预处理
    #随机生成噪声，形状为batch_size,4，32，32
    #构成输入部分，放入采样器
    for cnt,batch in enumerate(test_loader):
        #这个z处理的有问题，不太对，应该看看训练的时候z是怎么处理成z_t的，然后加入到我的这个工作中
        z, c = model.get_input(batch, model.first_stage_key)

        #noise =  torch.randn_like(z)
        #t=torch.tensor([603,  64, 329, 216, 130, 548, 829, 780, 270, 752]).to(device)
        #z_noisy=model.q_sample( z, t=t, noise=noise)
        c_cat, c, mask= c["c_concat"][0][:N], c["c_crossattn"][0][:N],c["mask"][:N]
        
        masks=transforms.Resize((32,32))(mask).cuda()
        #注意在这里masks是32，32的，然后mask的256*256的
        
        uc_cross = model.get_unconditional_conditioning(N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

        samples_cfg, _ = model.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                            batch_size=N, ddim=50,
                                            ddim_steps=50, eta=0.0,
                                            unconditional_guidance_scale=9.0,
                                            unconditional_conditioning=uc_full,
                                            x0=z,
                                            mask=masks
                                            )
        #z就是图片中的隐变量信息，应该是一个4*32*32的一个张量图,
        #但是这个z不是z_t，按理来说还应该得到每一步的z_t才行
        #pixel-level blending
        samples_cfg=samples_cfg*masks+z*(1-masks)
        x_samples = model.decode_first_stage(samples_cfg)

        
        #存成对应的图片以供之后操作

        for j in range(N):
            x = x_samples[j].cpu()
            source,mask=batch['hint'][j],batch['mask'][j]


            source=rearrange(source, "h w c -> c h w")
            x=source*(1-mask)+x*mask

            #这样是有用的，然后试一下后面的优化技巧,之后看一下怎么处理这部分的操作
            x = torch.clamp(x, -1., 1.)

            x = (x + 1.0) / 2.0
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            x=x.numpy()
            x = (255 * x).astype(np.uint8)
            x = Image.fromarray(x)
            x.save("{}/{}.png".format(save_image_path, cnt*N+j))

        #然后要把sample的这些图片保存的一个目录下面