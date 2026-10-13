# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     root = os.path.join(args.data_path, "train" if is_train else "val")
#     dataset = datasets.ImageFolder(root, transform=transform)

#     return dataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, "train" if is_train else "val")
    full_dataset = datasets.ImageFolder(root, transform=transform)
    
    # 如果需要只使用100个类别
    if hasattr(args, 'use_subset_classes') and args.use_subset_classes:
        # 获取所有类别
        all_classes = full_dataset.classes
        # 选择前100个类别（或者你可以指定特定的类别）
        selected_classes = all_classes[:100]
        # 创建一个类别名称到索引的映射
        selected_class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        # 创建一个原始索引到新索引的映射
        original_to_new_idx = {full_dataset.class_to_idx[cls]: idx for cls, idx in selected_class_to_idx.items()}
        
        # 筛选出属于选定类别的样本
        selected_indices = [i for i, (_, label) in enumerate(full_dataset.samples)
                           if label in original_to_new_idx]
        
        # 使用torch.utils.data.Subset来创建子集
        from torch.utils.data import Subset
        dataset = Subset(full_dataset, selected_indices)
        
        # 更新dataset的属性
        dataset.classes = selected_classes
        dataset.class_to_idx = selected_class_to_idx
        
        # 创建一个自定义的targets属性来存储新的标签
        original_targets = [full_dataset.targets[i] for i in selected_indices]
        dataset.targets = [original_to_new_idx[target] for target in original_targets]
        
        # 重新构建samples列表
        dataset.samples = [(full_dataset.samples[i][0], original_to_new_idx[full_dataset.samples[i][1]]) 
                          for i in selected_indices]
    else:
        dataset = full_dataset

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 232
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
