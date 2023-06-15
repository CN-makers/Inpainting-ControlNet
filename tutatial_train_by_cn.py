from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from coco_datsaset import CocoTrainValid
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd21_editing_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

#就是说，首先这个zero convolution在controlnet一开始是没有意义的，是可以通过初始化来加快他们的速度的，只要后面是0就可以了
#然后，他这个sd block也没有必要训练，因为

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
#这个model就是controlLDM,就是说这个model这可以把输入变成通道为9
model = create_model('./models/cldm_v21_editing.yaml').cpu()
#model.load_state_dict(load_state_dict(resume_path, location='cpu'))
#假设我改好了之后，以后再Load的时候用这个就好，但是训练的时候，第一次load应该是要换一种load方式的
model.init_from_ckpt(resume_path)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = CocoTrainValid(
        root="training/mscoco", split="train", image_resolution=256, 
        transform_type="imagenet_val", is_eval=False,)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
#test的时候让shuffle变成false，然后__len__改成1000应该就没问题了。
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger],max_epochs=10)


# Train!
trainer.fit(model, dataloader)
