# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:41:20 2023

@author: wenzt
"""

import numpy as np
import math
from copy import deepcopy
import time
    
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

#partitioned badge hyperparams, from active learning scale imgnet
POOLING_H = 16
POOLING_AREA = 512

def get_output_emb(all_loader, classifier, includEmb, args, model = None):
    
    if classifier is not None:
        classifier.eval()
    if model is not None:
        model.eval()
        
    totpre = []
    if includEmb:
        totemb = []
    # for cur_iter, (inputs, labels) in enumerate(all_loader):
    for cur_iter, (inputs, labels,_) in enumerate(all_loader):
        # if cur_iter % 10 == 1:
        #     print(cur_iter)
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        with torch.no_grad():
            if includEmb:
                if model is not None:
                    preds = model(inputs)
                    if classifier is not None:
                        if args.classifier_type == 'Linear':
                            emb = preds.clone()
                            preds = classifier(preds)
                        else:
                            preds = classifier(preds)
                            emb = classifier.emb
                else:
                    preds = classifier(inputs)
                    emb = classifier.emb
            else:
                if model is not None:
                    preds = model(inputs)
                    if classifier is not None:
                        preds = classifier(preds)
                else:
                    preds = classifier(inputs)
                    
            preds = F.softmax(preds, dim=1)
        if len(totpre) == 0:
            totpre = preds.cpu().numpy()
            if includEmb:
                totemb = emb.cpu().numpy().copy()
        else:
            totpre = np.vstack((totpre, preds.cpu().numpy() ))
            if includEmb:
                totemb = np.vstack((totemb, emb.cpu().numpy() ))

    
    if classifier is not None:
        classifier.train()
    if model is not None:
        model.train()
        
    if includEmb:
        return totpre, totemb
    else:
        return totpre, None

def cal_margin(all_loader, classifier, model):

    if classifier is not None:
        classifier.eval()
    if model is not None:
        model.eval()

    margins = []
    for cur_iter, (inputs, labels, _) in enumerate(all_loader):
        if cur_iter % 10 == 1:
            print(cur_iter)
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        with torch.no_grad():
            if model is not None:
                preds = model(inputs)
                if classifier is not None:
                    preds = classifier(preds)
            else:
                preds = classifier(inputs)
            preds = F.softmax(preds, dim=1)

        topprobs, topprobs_idx = preds.topk(dim=1, k=2, largest=True, sorted=True)
        batch_output_margins = topprobs[:, 0] - topprobs[:, 1]
        margins += [batch_output_margins.cpu()]

    margins = torch.cat(margins, dim=0)

    return margins.cpu().numpy()

def get_pre_prob(all_loader, model, classifier):

    if model is not None:
        model.eval()
    if classifier is not None:
        classifier.eval()
        
    totemb = []
    totprob = []
    # for cur_iter, (inputs, labels) in enumerate(all_loader):
    for cur_iter, (inputs, labels,_) in enumerate(all_loader):
        if cur_iter % 10 == 1:
            print(cur_iter)
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        with torch.no_grad():
            if model is not None:
                emb = model(inputs)
                if classifier is not None:
                    emb = classifier(emb)
            else:
                emb = classifier(inputs)
            emb = F.softmax(emb, dim=1)

        if len(totemb) == 0:
            totemb = emb.cpu().numpy().copy().argmax(axis=1).tolist()
            totprob = emb.cpu().numpy().copy().max(axis=1).tolist()
        else:
            totemb += emb.cpu().numpy().copy().argmax(axis=1).tolist()
            totprob += emb.cpu().numpy().copy().max(axis=1).tolist()

    return np.array(totemb), np.array(totprob)


def get_grad_embedding( classifier, all_loader, args, model = None):# Y is model predictions for unlabeld samples
    
    classifier.eval()
    if model is not None:
        model.eval()
    embedding = []
    
    for idxs, (x, _,_) in enumerate(all_loader):
        if idxs % 20 == 1:
            print('cal_grad_emb', idxs)
        
        with torch.no_grad():
            x = x.type(torch.cuda.FloatTensor)
            if model is not None:
                x = model(x)
            cout = classifier(x)
            if args.classifier_type == 'Linear':
                out = x.clone()
            else:
                out = classifier.emb#classifier.module.emb
            max_logit, max_logit_idx = cout.max(dim=1)
            
        cout.requires_grad = True
        loss = torch.nn.CrossEntropyLoss()(cout, max_logit_idx)
        grad = torch.autograd.grad(loss, cout)[0]
        
        with torch.no_grad():
            grad_embed = grad[:, :, None] * out[:, None, :]
            pool_h = min(POOLING_H, grad_embed.size(1))
            pool_w = int(float(POOLING_AREA) / pool_h)
            grad_embed = F.adaptive_avg_pool2d(grad_embed, (pool_h, pool_w))
            grad_embed = grad_embed.view(grad_embed.size(0), -1).cpu().numpy()
        
        if len(embedding) == 0:
            embedding = grad_embed.copy()
        else:
            embedding = np.vstack((embedding, grad_embed.copy()))
                        
    return embedding

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def adjust_learning_rate(optimizer, epoch, totepo, lr, lr_classifier, lrcos, schedule_milstone):
    """Decay the learning rate based on schedule"""
    num_groups = len(optimizer.param_groups)
    assert 1 <= num_groups <= 2
    lrs = []
    if num_groups == 1:
        lrs += [lr]
    elif num_groups == 2:
        lrs += [lr, lr_classifier]
    assert len(lrs) == num_groups
    for group_id, param_group in enumerate(optimizer.param_groups):
        lr = lrs[group_id]
        if lrcos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / totepo))
        else:  # stepwise lr schedule
            for milestone in schedule_milstone:
                lr *= 0.1 if epoch >= milestone else 1.
        param_group['lr'] = lr


def train_mlp(train_loader, classifier, args, class_weight = None):
    
    optimizer =  torch.optim.SGD(
        classifier.parameters(),
        lr = args.lr,
        momentum = args.momentum,
        weight_decay = args.weight_decay,#0.0003,
        nesterov = args.nesterov
    )
    
    num_epoch = args.train_eps
    
    scheduler = get_cosine_schedule_with_warmup( optimizer, 0, num_epoch*len(train_loader) )


    classifier.train()
    
    if class_weight is None:
        CE = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        CE = torch.nn.CrossEntropyLoss(reduction='mean', weight=class_weight)
    
    trainloss = []
    
    for ep in range(num_epoch):
        for cur_iter, (inputs, labels, _) in enumerate(train_loader):
            
            inputs = inputs.type(torch.cuda.FloatTensor)
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)

            preds = classifier(inputs)
            # Compute the loss
            loss = CE(preds, labels.long())#(preds, labels)
            # Perform the backward pass
            
            trainloss += [loss.item()]
            optimizer.zero_grad()
            loss.backward()
            # Update the parametersSWA
            optimizer.step()
            scheduler.step()
            
        # print(ep, trainloss[-1], (preds.argmax(axis=1) == labels).sum() )

    # totpre, totemb = get_output_emb(all_loader, model, classifier)
    
    return classifier, trainloss


def train_fine_tune(train_loader, model, classifier, args, class_weight = None):
    
    s = time.time()
    
    if classifier is not None:
        optimizer = torch.optim.SGD([
            {'params': model.parameters()},
            {'params': classifier.parameters(), 'lr':  args.cls_lr}###0.1
        ],  args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)###0.001
    else:
        feature_params, classifier_params = [], []
        feature_names, classifier_names = [], []
        for name, param in model.named_parameters():
            if 'fc.' in name:
                classifier_params += [param]
                classifier_names += [name]
            else:
                feature_params += [param]
                feature_names += [name]

        optimizer = torch.optim.SGD([
            {'params': feature_params},
            {'params': classifier_params, 'lr': args.cls_lr}
        ], args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov= args.nesterov)#

    num_epoch = args.train_eps
    
    #scheduler = get_cosine_schedule_with_warmup( optimizer, 0, num_epoch*len(train_loader) )
    
    model.train()
    if classifier is not None:
        classifier.train()

    trainloss = []

    # CE = torch.nn.CrossEntropyLoss(reduction='mean')
    if class_weight is None:
        CE = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        CE = torch.nn.CrossEntropyLoss(reduction='mean', weight=class_weight)
        
    optimizer.zero_grad()
    for epoch in range(num_epoch):
        
        adjust_learning_rate(optimizer, epoch, args.train_eps, args.lr, args.cls_lr, False, args.milestone)
        if epoch % 10 == 1:
            print('time', time.time() - s)

        for idx, (images, labels, _) in enumerate(train_loader):
                
            preds = model(images.cuda())#classifier(feature)# 
            if classifier is not None:
                preds = classifier(preds)
                
            loss = CE(preds, labels.long().cuda())
            trainloss += [loss.item()]
            loss = loss/args.grad_accu
            loss.backward()
            if idx % args.grad_accu == 0:
                optimizer.step()
                optimizer.zero_grad()
            #lr = lr_scheduler.step()
            #scheduler.step()
            
        if epoch >= (args.early_stop-1):
            break
    
    return model, classifier, trainloss


def train_freeze_mlp(train_loader, model, classifier, args, class_weight = None):
    
    s = time.time()
    
    optimizer =  torch.optim.SGD(
        classifier.parameters(),
        lr = args.cls_lr,#,30
        momentum = args.momentum,
        weight_decay = args.weight_decay,#0.0003,
        nesterov = args.nesterov
    )

    num_epoch = args.train_eps
    
    scheduler = get_cosine_schedule_with_warmup( optimizer, 0, num_epoch*len(train_loader) )
    
    model.eval()
    classifier.train()
    
    # CE = torch.nn.CrossEntropyLoss(reduction='mean')
    if class_weight is None:
        CE = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        CE = torch.nn.CrossEntropyLoss(reduction='mean', weight=class_weight)
    
    trainloss = []
    
    for epoch in range(num_epoch):

        if epoch % 10 == 1:
            print('time', time.time() - s)

        for idx, (images, labels, _) in enumerate(train_loader):

            with torch.no_grad():
                preds = model(images.cuda())
            preds = classifier(preds)
                
            loss = CE(preds, labels.long().cuda())
            trainloss += [loss.item()]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr = lr_scheduler.step()
            scheduler.step()
    
    return classifier, trainloss


def evaluation(test_loader, classifier, model = None):

    testpre_m, testl = [],[]
    if classifier is None and model is None:
        print('wrong evaluation model setting')
        return None, None
    
    if model is not None:
        model.eval()
    if classifier is not None:
        classifier.eval()
        
    for cur_iter, (inputs, labels,_) in enumerate(test_loader):
        if cur_iter % 10 == 1:
            print(cur_iter) 
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        with torch.no_grad():
            if model is not None:
                if classifier is None:
                    preds = model(inputs)
                else:
                    preds = classifier(model(inputs))
            else:    
                preds = classifier(inputs)
                
        if len(testpre_m) == 0:
            testpre_m = preds.cpu().numpy()
        else:
            testpre_m = np.vstack((testpre_m, preds.cpu().numpy() ))
        testl += labels.cpu().numpy().tolist()
    
    tspre = testpre_m.argmax(axis=1)    
    tacc = (tspre == np.array(testl)).sum() / len(testl)
    # acc += [tacc]
    print('test acc: ', tacc)  
    
    if classifier is not None:
        classifier.train()
    if model is not None:
        model.train()
    
    return tacc
