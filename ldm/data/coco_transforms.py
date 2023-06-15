# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


try:
    from torchvision.transforms.functional import get_image_size
except ImportError:
    from torchvision.transforms.functional import _get_image_size as get_image_size


class AugmentationDALLE(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def forward(self, img):
        w, h = get_image_size(img)
        s_min = min(w, h)

        off_h = torch.randint(low=3 * (h - s_min) // 8,
                              high=max(3 * (h - s_min) // 8 + 1, 5 * (h - s_min) // 8),
                              size=(1,)).item()
        off_w = torch.randint(low=3 * (w - s_min) // 8,
                              high=max(3 * (w - s_min) // 8 + 1, 5 * (w - s_min) // 8),
                              size=(1,)).item()

        img = F.crop(img, top=off_h, left=off_w, height=s_min, width=s_min)

        t_max = max(min(s_min, round(9 / 8 * self.size)), self.size)
        t = torch.randint(low=self.size, high=t_max + 1, size=(1,)).item()
        img = F.resize(img, [t, t])
        return img


class Rescale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return (1 - 2 * 0.1) * img + 0.1


def create_transforms(transform_type, image_resolution, split='train', is_eval=False):
    if transform_type == 'dalle':
        if split == 'train' and not is_eval:
            transforms_ = [
                AugmentationDALLE(size=image_resolution),
                transforms.RandomCrop(size=(image_resolution, image_resolution)),
                transforms.ToTensor(),
                Rescale()
            ]
        else:
            transforms_ = [
                transforms.Resize(size=(image_resolution, image_resolution)),
                transforms.ToTensor(),
                Rescale()
            ]
    elif transform_type == 'dalle-vqvae':
        if split == 'train' and not is_eval:
            transforms_ = [
                AugmentationDALLE(size=image_resolution),
                transforms.RandomCrop(size=(image_resolution, image_resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            transforms_ = [
                transforms.Resize(size=(image_resolution, image_resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
    elif transform_type == 'clip':
        if split == 'train' and not is_eval:
            transforms_ = [
                transforms.Resize(size=(image_resolution, image_resolution)),
                transforms.RandomResizedCrop(size=image_resolution, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            transforms_ = [
                transforms.Resize(size=(image_resolution, image_resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
    elif transform_type == 'clip-dvae':
        if split == 'train' and not is_eval:
            transforms_ = [
                transforms.Resize(size=(image_resolution, image_resolution)),
                transforms.RandomResizedCrop(size=image_resolution, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                Rescale()
            ]
        else:
            transforms_ = [
                transforms.Resize(size=(image_resolution, image_resolution)),
                transforms.ToTensor(),
                Rescale()
            ]
    elif transform_type == 'none':
        transforms_ = []
    elif transform_type == "imagenet_train":
        transforms_ = [
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif transform_type == "imagenet_val":
        transforms_ = [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    else:
        raise NotImplementedError('%s not implemented..' % transform_type)

    transforms_ = transforms.Compose(transforms_)

    return transforms_
