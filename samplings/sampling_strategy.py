# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 18:07:52 2021

active strategy
coreset

@author: wenzt
"""

import torch
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans
import torch.nn.functional as F

def cal_similarity_matrix(x, y):#labeled_embeddings:x ; unlabeled_embeddings: y
    
    yt = torch.transpose(y,1,0)
    x2 = torch.mul(x,x)
    y2 = torch.mul(y,y)
    xy = torch.mm(x,yt)
    
    onex = torch.ones(1,y.size(0))
    oney = torch.ones(x.size(0),1)
    
    x2s = torch.sum(x2,dim=1).unsqueeze(1)
    x22 = torch.mm(x2s,onex)
    
    y2s = torch.sum(y2,dim=1).unsqueeze(0)
    y22 = torch.mm(oney,y2s)
    
    dist = x22 + y22 - 2*xy
    
    return dist
    
def acquire_new_sample(budget, candnamelist, emb_lab, emb_un):#select diverse sampels based on embeddings
    
    selectedid = []
    new_sample = []
    
    emb_lab = [i.numpy() for i in emb_lab]
    emb_lab = np.array(emb_lab)
    emb_lab = torch.from_numpy(emb_lab)
    
    emb_un = [i.numpy() for i in emb_un]
    emb_un = np.array(emb_un)
    emb_un = torch.from_numpy(emb_un)
    
    v = cal_similarity_matrix(emb_lab, emb_un)
    for tsample in range(budget):
        v,_ = torch.min(v,dim=0)
        selectedid += [np.argmax((v.numpy()))]
        #print(np.max((v.numpy())))
        new_sample += [candnamelist[selectedid[-1]]]
        del candnamelist[selectedid[-1]]
        # similarity = torch.cat([similarity[:,:selectedid[-1]],similarity[:,selectedid[-1]+1:]],dim=1)
        emb_new = emb_un[selectedid[-1],:].unsqueeze(0)
        emb_un = torch.cat([ emb_un[:selectedid[-1],:], emb_un[selectedid[-1]+1:,:] ], dim=0)
        addsimrow = cal_similarity_matrix(emb_new, emb_un)
        v = torch.cat((v[:selectedid[-1]], v[(selectedid[-1]+1):]))
        v  = torch.cat([v.unsqueeze(0),addsimrow])
    
    return new_sample

# label_feas = f1m0[train_lbl_idx,:]
# unlabel_feas = f1m0[train_unlbl_idx,:]
# new_sample = acquire_new_sample(50, train_unlbl_idx, torch.from_numpy(label_feas), torch.from_numpy(unlabel_feas))
# label_feas = totfeas[coreset,:]
# uidx = list( set([i for i in range(50000)]) - set(coreset.tolist()) )
# unlabel_feas = totfeas[uidx,:]
# new_sample = acquire_new_sample(5000, uidx, torch.from_numpy(label_feas), torch.from_numpy(unlabel_feas))


### random selection
def random_select(budget, unlbl_idx):
    
    lbl_idx = np.random.randint(0,len(unlbl_idx) - 1, budget).tolist()
    
    return lbl_idx

def random_per_class(labels, budget, n_class):
    lbl_per_class = budget // n_class
    lbl_idx = []
    # unlbl_idx = []
    for i in range(n_class):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        lbl_idx.extend(idx[:lbl_per_class])
        # unlbl_idx.extend(idx[lbl_per_class:])
    return lbl_idx#, unlbl_idx

### kMedoids
def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return C,M.tolist()#, C #M indicates selected sampels, C indicates corresponding cluster idx




def euclidean_distances(x, y, squared=True):
    """Compute pairwise (squared) Euclidean distances.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x*x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y*y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances


###badge

def kmeans_plus(uidx, emb, num_budget):
    
    dist = euclidean_distances(emb, emb, squared=True)
    import time
    print('embedding dist calculated')
    
    uidx = np.array(uidx)
    candidx = [i for i in range(len(uidx))]
    
    newidx = [np.random.randint(0,len(candidx))]
    restidx = list( set(candidx) - set(newidx) )
    s0 = time.time()
    s = s0
    for i in range(num_budget - 1):
        tdist = dist[np.ix_(newidx,restidx)]
        tdist = tdist.min(axis=0)
        prob = ( tdist.T / tdist.sum() ).T
        newidx += np.random.choice(a=restidx, size=1, replace=False, p=prob).tolist()
        restidx = list( set(candidx) - set(newidx) )
        
        if i % int(num_budget/10) ==0:
            print('kmeans++ sampling ', i, time.time()-s)
            s = time.time()
    newidx = uidx[newidx]
    uidx = uidx.tolist()
    
    return newidx.tolist()

###badge for imagenet waiting for test
def kmeans_plus_partition(uidx, emb, num_budget, num_part):
    
    tuidx = uidx.copy()
    np.random.shuffle(tuidx)
    num_samples_group = int(len(uidx) / num_part)
    uidx_group = [tuidx[i*num_samples_group:(i+1)*num_samples_group] for i in range(num_part)]
    newidx = []
    for i in range(num_part):
        newidx += kmeans_plus(uidx_group[i], emb, int(num_budget / num_part))
    
    return newidx

### uncertainty min max softmax output
def uncertainty_sampling(totpre, uidx, num_budget):
    
    uidx = np.array(uidx)
    prob = totpre[uidx,:].max(axis=1)
    newidx = uidx[np.argsort(prob)[:num_budget]]
    
    return newidx.tolist()

### entropy
def entropy_sampling(totpre, uidx, num_budget):
    
    batchsize = 2048*5
    
    uidx = np.array(uidx)
    # totpre = totpre1[uidx,:]
    totpre = torch.from_numpy(totpre)
    numitr = math.ceil(len(totpre) / batchsize)
    
    entropy = []
    for i in range(numitr):
        prob = totpre[i*batchsize:(i+1)*batchsize,:].cuda()
        tentropy = - prob * torch.log(prob) 
        tentropy = tentropy.sum(dim=1)
        entropy += tentropy.cpu().numpy().tolist()

    newidx = uidx[np.argsort(entropy)[-num_budget:]]
    
    return newidx.tolist()

### confidence
def confidence_sampling_model(uidx, all_loader, classifier, model, num_budget, args):
    
    if classifier is not None:
        classifier.eval()
    if model is not None:
        model.eval()
        
    confidences = []
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
        
        topprobs,_ = preds.max(dim = 1)
        confidences += [topprobs.cpu()]
    
    confidences = torch.cat(confidences, dim=0)
    idx = torch.sort(confidences, descending=False).indices[:num_budget]
    uidx = np.array(uidx)

    if classifier is not None:
        classifier.train()
    if model is not None:
        model.train()
    
    if num_budget == 1:
        return [uidx[idx]]
    else:
        return uidx[idx].tolist()

###margin
# def margin_sampling(margins, uidx, num_budget):
    
#     idx = torch.sort(margins, descending=False).indices[:num_budget]
#     uidx = np.array(uidx)
    
#     return uidx[idx].tolist()

def margin_sampling(totpre, uidx, num_budget):
    
    totpre = torch.from_numpy(totpre)
    
    topprobs, topprobs_idx = totpre.topk(dim=1, k=2, largest=True, sorted=True)
    batch_output_margins = topprobs[:, 0] - topprobs[:, 1]
    margins = batch_output_margins
    
    idx = torch.sort(margins, descending=False).indices[:num_budget]
    uidx = np.array(uidx)
    
    return uidx[idx].tolist()

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

    if classifier is not None:
        classifier.train()
    if model is not None:
        model.train()
        
    return margins

def margin_sampling_model(uidx, all_loader, classifier, model, num_budget, args):
    
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
    idx = torch.sort(margins, descending=False).indices[:num_budget]
    uidx = np.array(uidx)

    if classifier is not None:
        classifier.train()
    if model is not None:
        model.train()
    
    if num_budget == 1:
        return [uidx[idx]]
    else:
        return uidx[idx].tolist()


