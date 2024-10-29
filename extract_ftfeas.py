# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:53:46 2023

@author: wenzt
"""

import numpy as np
import os
import torch

from dataset_model import get_augmentation, get_dataset, get_network
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--outpath_base', default='./res/', type=str,
                    help='path of extracted features')
parser.add_argument('--dataset_name', default='cifar10', type=str,
                    help='name of dataset')
parser.add_argument('--dataset_path', default='./data/cifar10', type=str,
                    help='for fine-tune and freezing & mlp')
parser.add_argument('--checkpoints_path', default='./checks/checkpoint_400_.pth.tar', type=str,
                    help='path of fine-tune model')
parser.add_argument('--batchsize', default=128, type=int,
                    help='batch size')
parser.add_argument('--network', default='res50', type=str,
                    help='[res18,res50,res50x2,res50x4]')

args = parser.parse_args()

def get_emb(all_loader, model):

    if model is not None:
        model.eval()
        
    totemb = []
    for cur_iter, (inputs, labels,_) in enumerate(all_loader):
        if cur_iter % 10 == 1:
            print(cur_iter)
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        with torch.no_grad():
            emb = model(inputs)

        if len(totemb) == 0:
            totemb = emb.cpu().numpy().copy()
        else:
            totemb = np.vstack(( totemb, emb.cpu().numpy().copy() ))

    return totemb

# model loading
model = get_network(args)
model.fc = torch.nn.Identity()

checkpoint = torch.load(args.checkpoints_path, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])

model = model.cuda()

# data loading

transform_test = get_augmentation(args, train = False)

allset = get_dataset(args, transform_test, index = None, train = True )
testset = get_dataset(args, transform_test, index = None, train = False )

all_loader = torch.utils.data.DataLoader(
    allset,
    batch_size = args.batchsize,
    shuffle = False,
    drop_last = False
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size = args.batchsize,
    shuffle = False,
    drop_last = False
)

feas = get_emb(all_loader, model)
feas_test = get_emb(test_loader, model)

np.save(os.path.join(args.outpath_base, 'ftfeas.npy'), feas)
np.save(os.path.join(args.outpath_base, 'ftfeas_test.npy'), feas_test)