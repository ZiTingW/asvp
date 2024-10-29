# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:18:32 2023

### check whether the proj weight is necessary
@author: wenzt
"""

import numpy as np
import os
import torch

from dataset_model import FeasDataset, ImageFolderWithIndex, MLP, get_augmentation, get_dataset, get_network
from utils import train_mlp, evaluation, get_output_emb, train_freeze_mlp, train_fine_tune

import argparse
import shutil

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sampling_strategy', default='random', type=str,
                    help='Sampling strategy')
parser.add_argument('--al_budget', default='[500]*8', type=str,#[10000]*10
                    help='dims of classifier')
parser.add_argument('--expid', default='ft_hyper1', type=str,
                    help='order of exps')
parser.add_argument('--outpath_base', default='./res/', type=str,
                    help='path of results')
parser.add_argument('--dataset_name', default='cifar100', type=str,
                    help='name of dataset [imagenet, feas, cifar10, cifar100, imagenet100]')
parser.add_argument('--dataset_path', default='./data/cifar10/', type=str,
                    help='for fine-tune and freezing & mlp')
parser.add_argument('--selfmodel_path', default='./pretrain/simsiam-cifar100-experiment-wrn288_1225013145.pth', type=str,
                    help='path of selfsup model')

parser.add_argument('--load_proj_weight', default=False, type=bool,
                    help='initialized classifier weights from projector')
parser.add_argument('--load_al_weight', default=False, type=bool,
                    help='initialized classifier weights from last Active learning round')

parser.add_argument('--train_eps', default=200, type=int,#50 for imagenet, 100 for others
                    help='# of training epoch')
parser.add_argument('--lr', default=0.001, type=float,#0.05
                    help='learning rate')
parser.add_argument('--cls_lr', default=0.001, type=float,#0.05
                    help='learning rate for classifier')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', default=3e-4, type=float,
                    help='weight_decay')
parser.add_argument('--nesterov', default=True, type=bool,
                    help='nesterov')
parser.add_argument('--milestone', default='120, 160', type=str,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--early_stop', default=200, type=int,
                    help='efficient AL baseline, early stop')

parser.add_argument('--network', default='wrn288', type=str,
                    help='[res18,res50,res50x2,res50x4,wrn288]')

parser.add_argument('--batchsize_train', default=100, type=int,
                    help='path of testset label')
parser.add_argument('--grad_accu', default=1, type=int,
                    help='num grad accum')
parser.add_argument('--batchsize_al_forward', default=512, type=int,##256
                    help='path of testset label')
parser.add_argument('--batchsize_evaluation', default=512, type=int,
                    help='path of testset label')
parser.add_argument('--classifier_dim', default='512,512,100', type=str,
                    help='dims of classifier')
parser.add_argument('--classifier_type', default='Linear', type=str,
                    help='Linear or MLP')

parser.add_argument('--training_mode', default=2, type=int,
                    help='0:MLP_proxy(ours), 1:freezing encoder and training classifier, 2:Fine-tuning')

parser.add_argument('--distributed_training', default=False, type=bool,
                    help='using nn.dataparaller')

parser.add_argument('--alidx_name', default='alidx.npy', type=str,
                    help='path of selfsup model')
parser.add_argument('--mlpproxy_expid', default='mlpproxy_hyper7', type=str,
                    help='order of exps')
parser.add_argument('--mlpproxy_dataset', default='feas', type=str,
                    help='order of exps')
parser.add_argument('--mlpproxy_trainmode', default='_training_strategy0', type=str,
                    help='order of exps')
parser.add_argument('--alidxpath', default=None, type=str,
                    help='another choice to input alidx')

args = parser.parse_args()

args.milestone = args.milestone.split(',')
args.milestone = [int(i) for i in args.milestone]


print(args.lr)
print(args.cls_lr)
print(args.expid)

indim_classifier, hiddim_classifier, outdim_classifier = args.classifier_dim.split(',')
indim_classifier, hiddim_classifier, outdim_classifier = int(indim_classifier), [int(hiddim_classifier)], int(outdim_classifier)
# indim_classifier, hiddim_classifier, outdim_classifier = 512,[128],10#[4096,4096]

num_budget = eval(args.al_budget)

num_al_itr = len(num_budget)


sampling_strategy = args.sampling_strategy#'badge'#'coreset_self'
expid = args.expid#'1_waug'#1


dataset = args.dataset_name#'imagenet100'#

outpath = os.path.join(args.outpath_base, dataset) #os.path.join('C:\\document\\code\\MLP_proxy\\res\\', dataset) # + exp name
exp_name = dataset + '_' + sampling_strategy + '_exp' + str(expid) + '_training_strategy' + str(args.training_mode)
outpath = os.path.join(outpath, exp_name)
os.makedirs(outpath, exist_ok=True) 


if args.alidxpath is None:
    alidx_path = os.path.join(args.outpath_base, args.mlpproxy_dataset, 
                              args.mlpproxy_dataset + '_' +sampling_strategy + '_exp' + args.mlpproxy_expid + args.mlpproxy_trainmode,
                              args.alidx_name)
else:
    alidx_path = args.alidxpath
hyperalidx = np.load(alidx_path)

#record configuration file
shutil.copy(os.path.join('.','ft_mlpproxy.py'), outpath)

if args.selfmodel_path == "None":
    args.selfmodel_path = None

selfmodel_path = args.selfmodel_path

transform_train = get_augmentation(args, train = True)
transform_test = get_augmentation(args, train = False)

testset = get_dataset(args, transform_test, index = None, train = False )
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size = args.batchsize_evaluation,
    shuffle = False,
    drop_last = False
)


if args.classifier_type == 'MLP':
    classifier = MLP(indim_classifier, hiddim_classifier, outdim_classifier)
elif args.classifier_type == 'Linear':
    classifier = torch.nn.Linear(indim_classifier,outdim_classifier)
else:
    raise NotImplementedError
    
totacc = []
tracc = []

###load model and initiliaze with self-sup weight 
model = get_network(args)

if selfmodel_path is not None:
    checkpoint = torch.load(selfmodel_path, map_location=torch.device('cpu'))

    if args.training_mode != 0:
        
        #print(model)
        encoder_dict = model.state_dict()
        if args.network == 'res50':
            #state_dict = {k[7:]:v for k,v in checkpoint['online_backbone'].items() if k[7:] in encoder_dict.keys()}#byol
            state_dict = {k[27:]:v for k,v in checkpoint['state_dict'].items() if k[27:] in encoder_dict.keys()}#byol eman
        elif args.network == 'res18' or args.network == 'wrn288':
            if 'simclr' in args.selfmodel_path:
                state_dict = {}
                for k in checkpoint:
                    newk = k[9:]
                    if 'shortcut' in newk:        
                        newk = newk.replace('shortcut', 'downsample')
                    if newk in encoder_dict.keys():
                        state_dict[newk] = checkpoint[k]
            else:
                state_dict = {k[9:]:v for k,v in checkpoint['state_dict'].items() if k[9:] in encoder_dict.keys()} 

        else:
            raise NotImplementedError
        
        encoder_dict.update(state_dict)
        
        msg = model.load_state_dict(encoder_dict)
        print('load pretrained model ', len(state_dict), msg)
    
    model.fc = torch.nn.Identity()  


import time

s = time.time()
s0 = time.time()

alidx = []
for alitr in range(num_al_itr):
    
    if (alitr == 0 and len(alidx) == 0) or (alitr > 0 and len(alidx) > 0):#not resume al 

        alidx = hyperalidx[:np.sum(num_budget[:alitr+1])]
        np.save(os.path.join(outpath, 'alidx.npy'), np.array(alidx))
    
    trainset = get_dataset(args, transform_train, index = alidx, train = True )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size = args.batchsize_train,
        shuffle=True
    )
    
    if (args.load_al_weight and alitr == 0) or (not args.load_al_weight):
        if args.classifier_type == 'MLP':
            classifier = MLP(indim_classifier, hiddim_classifier, outdim_classifier)
        elif args.classifier_type == 'Linear':
            classifier = torch.nn.Linear(indim_classifier,outdim_classifier)
        else:
            raise NotImplementedError
    
    # if args.load_proj_weight:
    #     classifier.load_state_dict(model_dict)
    
    classifier.cuda()
    if args.distributed_training:
        classifier = torch.nn.DataParallel(classifier)
    
    model = get_network(args)
    if selfmodel_path is not None:
        model.load_state_dict(encoder_dict)
    model.fc = torch.nn.Identity() 
    model.cuda()
    if args.distributed_training:
        model = torch.nn.DataParallel(model)
    
    print('point 2 model load', time.time() - s)
    s = time.time()
    
    ### training
    if args.training_mode == 1:
        classifier, trainloss = train_freeze_mlp(train_loader, model, classifier, args)
        torch.save({'epoch': args.train_eps, 'classifier_state_dict': classifier.state_dict()}, os.path.join(outpath, 'checkpoint_' + str(len(alidx)) + '_.pth.tar'))
    elif args.training_mode == 2: 
        model, classifier, trainloss = train_fine_tune(train_loader, model, classifier, args)
        torch.save({'epoch': args.train_eps, 'classifier_state_dict': classifier.state_dict(), 'model_state_dict': model.state_dict()}, os.path.join(outpath, 'checkpoint_' + str(len(alidx)) + '_.pth.tar'))
    else:
        raise NotImplementedError
    
    print('point 3 training', time.time() - s)
    print('training loss: ', trainloss)
    s = time.time()
    
    ### evaluation
    acc = evaluation(test_loader, classifier, model = model)
    tacc = evaluation(train_loader, classifier, model = model)
    
    print('point 4 evaluation', time.time() - s)
    s = time.time()
    
    #np.save(outpath + 'totpre' + str(len(alidx)) + '.npy', totpre)
    #np.save(outpath + 'alfeas' + str(len(alidx)) + '.npy', totemb)
    totacc += [acc]
    tracc += [tacc]
    print('AL lblset size is ', len(alidx), 'time ', time.time() - s)
    s = time.time()
    print('test acc: ', acc)
    print('train acc: ', tacc)
    np.save(os.path.join(outpath, 'acc.npy'), np.array(totacc))

### save
np.save(os.path.join(outpath, 'acc.npy'), np.array(totacc))

print('total time:', time.time() - s0)