# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:42:42 2024

code from https://github.com/bluecdm/Long-tailed-recognition/blob/main/dataset/imbalance_cifar.py
"""

import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from PIL import Image


#CIFAR10    
class CIFAR10sub_imb(torchvision.datasets.CIFAR10):
    def __init__(self, root, indexs, all_imbsel_idx, imb_type = 'exp', imb_factor = 0.1, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         download=download)
        # np.random.seed(rand_number)
        if all_imbsel_idx is None:
            self.all_imbsel_idx = []
            img_num_list = self.get_img_num_per_cls(10, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
        else:
            self.all_imbsel_idx = all_imbsel_idx
            self.data = self.data[all_imbsel_idx]
            self.targets = np.array(self.targets)[all_imbsel_idx]
            
            
        self.all_imbsel_idx = np.array(self.all_imbsel_idx)
        
        # self.targets = np.array(self.targets)
        
        if indexs is not None:
            indexs = np.array(indexs)
            # indexs = self.all_imbsel_idx[indexs]
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = self.all_imbsel_idx[indexs]#indexs

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index#self.indexs[index]
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            self.all_imbsel_idx += selec_idx.tolist()
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

