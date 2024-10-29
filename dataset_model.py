# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:38:54 2023

@author: wenzt
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from torchvision.datasets.folder import ImageFolder, default_loader
import torch.nn.functional as F 
from glob import glob
import torchvision
import torchvision.transforms as transforms
import math
import pathlib
from typing import Any,Tuple


def get_dataset(args, transform, index = None, train=True):
 
    if args.dataset_name == 'cifar10':
        dataset = CIFAR10sub(root = args.dataset_path, indexs = index, train = train, transform = transform)
    
    elif args.dataset_name == 'cifar10LT':
        from cifar10_imb_dataset import CIFAR10sub_imb
        if train:
            if args.all_imbsel_idx is None:
                all_imbsel_idx = None
            else:
                all_imbsel_idx = np.load(args.all_imbsel_idx)
            dataset = CIFAR10sub_imb(root = args.dataset_path, indexs = index, all_imbsel_idx = all_imbsel_idx, imb_factor = args.imb_factor, train = train, transform = transform)
            if args.all_imbsel_idx is None:
                np.save( os.path.join(args.outpath, 'allidx_imb.npy'),  dataset.all_imbsel_idx)
                args.all_imbsel_idx = os.path.join(args.outpath, 'allidx_imb.npy')
        else:
            dataset = CIFAR10sub(root = args.dataset_path, indexs = index, train = train, transform = transform)
    
    elif args.dataset_name == 'cifar100':
        dataset = CIFAR100sub(root = args.dataset_path, indexs = index, train = train, transform = transform)     
    
    
    elif args.dataset_name == 'imagenet':
        if train:
            dataset = ImageFolderWithIndex(root = os.path.join(args.dataset_path, 'train'), indexs= index, transform = transform)
        else:
            assert index is None, 'wrong test idx setting'
            dataset = ImageFolderWithIndex(root = os.path.join(args.dataset_path, 'val'), indexs= index, transform = transform)
            # datasets.ImageFolder(root = os.path.join(args.dataset_path, 'val'), transform = transform)
    elif args.dataset_name == 'imagenet100':
        if train:
            dataset = ImageNetSubset(args.img100_subfile, args.dataset_path, index, split = 'train', transform = transform)
        else:
            assert index is None, 'wrong test idx setting'
            dataset = ImageNetSubset(args.img100_subfile, args.dataset_path, index, split = 'val', transform = transform)
    
    elif args.dataset_name == 'pets':
        if train:
            dataset = PetSubset(args.dataset_path, index = index, target_types='category', split = 'trainval', download=True, transform = transform)
        else:
            assert index is None, 'wrong test idx setting'
            dataset = PetSubset(args.dataset_path, index = index, target_types='category', split = 'test', download=True, transform = transform)
   
    elif args.dataset_name == 'svhn':
        if train:
            dataset = SVHNsub(root=args.dataset_path, indexs = index, split = 'train', download=True, transform=transform)
        else:
            assert index is None, 'wrong test idx setting'
            dataset = SVHNsub(root=args.dataset_path, indexs = index, split = 'test', download=True, transform=transform)
        
    else:
        raise NotImplementedError

    return dataset

def get_augmentation(args, train):
    
    if train:
        if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar10LT': # for cifar10 and cifar10LT
            if args.network == 'res18':
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ])
            else:
                transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        elif args.dataset_name == 'cifar100':
            if args.network == 'res50':
                transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            else:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
        
        elif args.dataset_name == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                           transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            #transforms.Resize(size = [224,224]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
        elif args.dataset_name == 'imagenet100':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                           transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            #transforms.Resize(size = [224,224]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
            
        elif args.dataset_name == 'pets':
            if args.network == 'res50':
                transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            else:
                raise NotImplementedError
        
        elif args.dataset_name == 'svhn':
            transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)),
                            ])
        
        
        else:
            raise NotImplementedError
    else:
        if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar10LT':
            if args.network == 'res18':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ])
            else:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
        
        elif args.dataset_name == 'cifar100':
            if args.network == 'res50' or 'clip' in args.network:
                transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
            
        elif args.dataset_name == 'imagenet' or args.dataset_name == 'vic_cape_howe':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
        elif args.dataset_name == 'imagenet100':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        
        elif args.dataset_name == 'pets':
            if args.network == 'res50' or 'clip' in args.network:
                transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            else:
                raise NotImplementedError
                
        elif args.dataset_name == 'svhn':
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)),
                    ])
        
        else:
            raise NotImplementedError
    
    return transform

def get_network(args):
    
    if args.dataset_name == 'feas':
        model = None
    elif args.network == 'res50':
        from torchvision import models
        model = models.resnet50()
    elif args.network == 'res18':
        from cifar_resnet_1 import resnet18
        model = resnet18()
    elif args.network == 'wrn288':
        from wideresnet import build_wideresnet
        model = build_wideresnet(28, 8, 100)# only for cifar100
    elif 'clip' in args.network:
        from clip_backbone import CLIPBackbone
        model = CLIPBackbone(args)
    else:
        raise NotImplementedError
    
    return model

class FeasDataset(Dataset):
    def __init__(self, x, y, noise_scale):
        self.x = x
        self.y = y
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.y) 

    def __getitem__(self, idx):
        if self.noise_scale is None:
            return self.x[idx,:], self.y[idx], idx
        else:
            return np.random.normal(self.x[idx,:], scale = self.noise_scale[self.y[idx],:]), self.y[idx], idx
        
class PartFeasDataset(Dataset):
    def __init__(self, feas_path, y, idx, noise_scale, num_gap):
        
        idx.sort()
        
        num_part = math.ceil( 1281167 / num_gap )
        x = []
        for i in range(num_part):
            tfeas = np.load(os.path.join(feas_path, 'feas' + str(i) + '.npy'))
            tidx0 = np.argwhere(idx >= i*num_gap)[:,0]
            tidx1 = np.argwhere(idx < (i+1)*num_gap)[:,0]
            tidx = list(set(tidx0.tolist()) & set(tidx1.tolist()))
            tidx = idx[tidx] % num_gap
            if len(x) == 0:
                x = tfeas[tidx,:]
            else:
                x = np.vstack((x, tfeas[tidx,:]))
            del tfeas
            
        self.x = x
        self.y = y[idx]
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.y) 

    def __getitem__(self, idx):
        if self.noise_scale is None:
            return self.x[idx,:], self.y[idx], idx
        else:
            return np.random.normal(self.x[idx,:], scale = self.noise_scale[self.y[idx],:]), self.y[idx], idx


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=[2048], out_dim=2048):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim[0]),
            torch.nn.BatchNorm1d(hidden_dim[0]),
            torch.nn.ReLU(inplace=True)
        )
        
        if len(hidden_dim) == 1:
            self.layer2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim[0], out_dim)
                # torch.nn.BatchNorm1d(out_dim)
            )
            self.num_layers = 1
        else:
            self.layer2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim[0], hidden_dim[1]),
                torch.nn.BatchNorm1d(hidden_dim[1]),
                torch.nn.ReLU(inplace=True)
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim[1], out_dim)
                # torch.nn.BatchNorm1d(out_dim)
            )
            self.num_layers = 2
            
        self.emb = None
        
    # def set_layers(self, num_layers):
    #     self.num_layers = num_layers

    def forward(self, x):
        
        x = self.layer1(x)
        if self.num_layers == 1:
            self.emb = x.clone()
            x = self.layer2(x)
        else:
            x = self.layer2(x)
            self.emb = x.clone()
            x = self.layer3(x)
        return x     
    
    # def forward(self, x):
        
    #     x1 = self.layer1(x)
    #     if self.num_layers == 1:
    #         self.emb = x1.clone()
    #         x = self.layer2(x1)
    #     else:
    #         x = self.layer2(x1)
    #         self.emb = x.clone()
    #         x = self.layer3(x)
    #     return x, x1
    


### imagenet     
class ImageFolderWithIndex(ImageFolder):
    
    def __init__(self, root, indexs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

#CIFAR10    
class CIFAR10sub(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index#self.indexs[index]

#cifar100
class CIFAR100sub(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         download=download)
        
        self.targets = np.array(self.targets)

        if indexs is not None:        
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index#, self.indexs[index]

#imagenet100    
class ImageNetSubset(Dataset):
    def __init__(self, subset_file, root, index, split='train', 
                    transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root,  split)
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            # subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            ### check if order is consistent on different devices, done
            # with open('C:\\document\\data\\check\\f' + str(i) + '.txt','w') as f:
            #     for ittt in files:
            #         f.write(ittt +'\n')
            for f in files:
                imgs.append((f, i)) 
        # self.imgs = imgs 
        self.classes = class_names
        
        if index is not None:
            self.imgs = [imgs[i] for i in index]
        else:
            self.imgs = imgs

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB') 
            
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        # im_size = img.size
        
        # class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index
    
class PetSubset(torchvision.datasets.OxfordIIITPet):
    def __init__(self, root, index, target_types='category', split = 'trainval',
                    transforms = None, transform=None,  target_transform=None, download = True):
        super(PetSubset, self).__init__(root,  transform=None,  target_transform=None)

        self._split = split#verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = target_types
        self._target_types = [target_types]#[ verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types]

        # super().__init__(root, split = 'trainval', transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]
        
        self.index = index
        if index is not None:
            self._images = [self._images[i] for i in index]
            self._labels = [self._labels[i] for i in index]
        
        self.transform = transform
        
        self.target_transform = target_transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)
        
        if self.transform:
            image = self.transform(image)

        return image, target, idx
    
    
class SVHNsub(datasets.SVHN):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, split=split,
                         transform=transform,
                         download=download)
        
        self.labels = np.array(self.labels)
        
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
            self.indexs = indexs

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index