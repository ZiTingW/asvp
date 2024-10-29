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


###GMM
def GMM(budget, totfeas, dist):
    
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=budget).fit(totfeas)
    labels = gmm.predict(totfeas)
    
    new_sample = []
    for ic in range(budget):
        sidx = np.argwhere(labels == ic)[:,0]
        sdist = dist[sidx,:]
        sdist = sdist[:,sidx]
        sumdist = sdist.sum(axis=0)
        new_sample += [ sidx[sumdist.argmin()] ]
    
    return new_sample


###TypiClust_rp

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

def constrain_kmeans(totfeas, lbl_idx, lbl_target, num_cluster, max_itr = 1000):

    mu = []
    unlbl_idx = list( set([i for i in range(len(totfeas))]) - set(lbl_idx) )
    lbl_idx = np.array(lbl_idx)
    for i in range(num_cluster):
        if i in lbl_target:
            idx = lbl_idx[np.argwhere(lbl_target == i)[:,0].astype('int')]
            mu += [ totfeas[idx,:].copy().mean(axis=0)  ]
        else:
            idx = unlbl_idx[ np.random.randint(len(unlbl_idx)) ]
            unlbl_idx.remove(idx)
            mu += [ totfeas[idx,:].copy() ]

    mu = np.array(mu)
    tdist  = euclidean_distances(mu, totfeas, squared=False)#metrics.pairwise.pairwise_distances(mu, totfeas, metric='euclidean')
    tlabel = tdist.argmin(axis=0)
    oldlabel = tlabel.copy()

    # lbl_idx = lbl_idx.tolist()

    for itr in range(max_itr):
        for ic in range(num_cluster):
            idx = np.argwhere(tlabel == ic)[:,0].tolist()
            idx = list( set(idx) - set(lbl_idx.tolist()) )
            if ic in lbl_target:
                idx += lbl_idx[np.argwhere(lbl_target == ic)[:,0].astype('int')].tolist()
            mu[ic,:] = totfeas[idx,:].mean(axis=0)

        tdist  = euclidean_distances(mu, totfeas, squared=False)#metrics.pairwise.pairwise_distances(mu, totfeas, metric='euclidean')
        tlabel = tdist.argmin(axis=0)

        same_r = (tlabel == oldlabel).sum() / len(tlabel)
        if same_r == 1:
            break

        oldlabel = tlabel.copy()

    return tlabel, same_r

def cal_typi(dist, candidx, clusteridx, K = 20):
    
    tdist = dist[candidx,:]
    tdist = tdist[:,clusteridx]
    idx = tdist.argsort()[0,:K]
    tdist = tdist[0,idx]
    typi = 1 / np.mean(tdist)
    
    return typi

def query_typiclust_first(cluster, dist, B = 20):

    new_sample = []
    for ib in range(B):
        clusteridx = np.argwhere(cluster == ib)[:,0]
        tottypi = []
        for icand in clusteridx:
            ttypi = cal_typi(dist, [icand], clusteridx, K = 20)
            tottypi += [ttypi]

        new_sample += [ clusteridx[np.argmax(tottypi)] ]

    return new_sample

def query_typiclust_first2(dist, cidx):#dist within one cluster

    new_sample = []
    tottypi = []
    clusteridx = [i for i in range(len(cidx))]
    for icand in range(len(cidx)):
        ttypi = cal_typi(dist, [icand], clusteridx, K = 20)
        tottypi += [ttypi]

    new_sample += [ cidx[np.argmax(tottypi)] ]

    return new_sample

def query_typiclust(cluster, candcluster, dist, B = 20):
    
    new_sample = []
    assert B == len(candcluster), 'wrong cand cluster / budget'
    for ib in candcluster:
        clusteridx = np.argwhere(cluster == ib)[:,0]
        tottypi = []
        for icand in clusteridx:
            ttypi = cal_typi(dist, [icand], clusteridx, K = 20)
            tottypi += [ttypi]
        
        new_sample += [ clusteridx[np.argmax(tottypi)] ]
        
    return new_sample

def find_uncover_cluster(lblidx, cluster, B = 20): #增补当新增空闲聚类数不够的处理:找重叠最少的类
    
    #remove empty cluster
    candidx = []
    for i in np.unique(cluster):
        if (cluster == i).sum() >= 1:
            candidx += [i]
    
    uncover_class =  list( set(cluster.tolist()) - set(cluster[lblidx].tolist()) )
    
    cand_uncoverclass = list( set(uncover_class) & set(candidx) )
    
    if len(cand_uncoverclass) >= B:
        totnum = []
        for i in cand_uncoverclass:
            totnum += [(cluster == i).sum()]
        
        candcluster = np.argsort(totnum)[-B:]
        cand_uncoverclass = np.array(cand_uncoverclass)
        candcluster = cand_uncoverclass[candcluster]
    else:
        num_overlap = np.bincount(cluster[lblidx], minlength = max(cluster)+1)
        cand_num_overlap = num_overlap[candidx]
        candcluster = np.argsort(cand_num_overlap)[:B]
        candcluster = np.array(candidx)[candcluster]
    
    return candcluster

# dist = euclidean_distances(totfeas, totfeas, squared=False)
# #bootstrap
# cluster20,_ = constrain_kmeans(totfeas, [], [], 20, max_itr = 1000)
# lblidx = query_typiclust_first(cluster20, dist, B = 20)
# #t+1 cycle
# num_budget_gap = 20
# num_cluster = 20
# B = 20
# for ial in range(4):
#     num_cluster += num_budget_gap
#     cluster,_ = constrain_kmeans(totfeas, [], [], num_cluster, max_itr = 1000)
#     candcluster = find_uncover_cluster(lblidx, cluster, B)
#     new = query_typiclust(cluster, candcluster, dist, B)
#     lblidx += new

# def sampling_typicluster(totfeas, alidx, num_budget):
    
#     dist = euclidean_distances(totfeas, totfeas, squared=False)
#     if len(alidx) == 0:
#         cluster,_ = constrain_kmeans(totfeas, [], [], num_budget, max_itr = 1000)
#         newidx = query_typiclust_first(cluster, dist, B = num_budget)
#     else:
#         num_cluster = len(alidx) + num_budget
#         if num_cluster > 500:
#             num_cluster = 500
#         cluster,_ = constrain_kmeans(totfeas, [], [], num_cluster, max_itr = 1000)
#         candcluster = find_uncover_cluster(alidx, cluster, num_budget)
#         newidx = query_typiclust(cluster, candcluster, dist, num_budget)

#     return newidx

### using minibatch kmeans for large scale data
def sampling_typicluster(totfeas, alidx, num_budget, args):
    
    
    newidx = []
    if len(alidx) == 0:
        kmeans = MiniBatchKMeans(n_clusters=num_budget, batch_size=4096)
        cluster = kmeans.fit_predict(totfeas)
        
        for ic in range(num_budget):
            cidx = np.argwhere(cluster == ic)[:,0]
            if len(cidx) != 0:
                dist = euclidean_distances(totfeas[cidx,:], totfeas[cidx,:], squared=False)
                tidx = query_typiclust_first2(dist, cidx )
                newidx += tidx
    else:
        num_cluster = len(alidx) + num_budget
        if num_cluster > args.max_cluster_typiclust:#500:
            num_cluster = args.max_cluster_typiclust# 500
        kmeans = MiniBatchKMeans(n_clusters=num_cluster, batch_size=4096)
        cluster = kmeans.fit_predict(totfeas)
        candcluster = find_uncover_cluster(alidx, cluster, num_budget)

        for ic in candcluster:
            cidx = np.argwhere(cluster == ic)[:,0]
            if len(cidx) != 0:
                dist = euclidean_distances(totfeas[cidx,:], totfeas[cidx,:], squared=False)
                tidx = query_typiclust_first2(dist, cidx )
                newidx += tidx
        
    uidx =  list( set([i for i in range(len(totfeas))]) - set(alidx) )
    np.random.shuffle(uidx)
    num = num_budget - len(newidx)
    newidx += uidx[:num]
        
    return newidx


#probcover
def estimate_thresh(dist, cluster, thresh = 0.95, num_bins = 20):
    
    totpurity = []
    maxdist = dist.max()
    num_gap = maxdist*0.7 / num_bins
    idist = 1
    while(True):
        tdist = num_gap*idist
        purity,tnum = 0,0
        for i in range(len(cluster)):
            idx = np.argwhere(dist[i,:] <= tdist)[:,0]
            purity += ( (cluster[i] == cluster[idx]).sum() )
            tnum += len(idx) 
        totpurity += [purity / tnum]
        if totpurity[-1] < thresh:
            break
        idist += 1
    
    return tdist - num_gap

def construct_dg(totfeas, num_class, thresh):
    
    dist = euclidean_distances(totfeas, totfeas, squared=False)
    cluster,_ = constrain_kmeans(totfeas, [], [], num_class)
    
    thresh_dist = estimate_thresh(dist, cluster, thresh = thresh)#
    print('probcover purity ' + str(thresh) + ' is equal to dist ', thresh_dist)
    
    dg = {}
    # thresh = 0.02
    for i in range(len(totfeas)):
        tidx = np.argwhere(dist[i,:] <= thresh_dist)[:,0]
        dg[i] = tidx.tolist()
    
    return dg

def find_new_one(dg):

    num = 0
    tidx = 0
    for i in dg:
        if len(dg[i]) > num:
            num = len(dg[i])
            tidx = i

    return tidx, num

def update_dg(dg, covered_idx):
    if len(dg[covered_idx]) == 0:
        del dg[covered_idx]
    
    else:
        tcover = set(dg[covered_idx])
        removeidx = []
        for ind in dg:
            dg[ind] = list(set(dg[ind]) - tcover)
            if len(dg[ind]) == 0:
                removeidx += [ind]
        
        removeidx += [covered_idx]
        removeidx = list(set(removeidx))
        for ir in removeidx:
            del dg[ir]
    return dg

def sampling_probcover(dg, num_budget):
    
    alidx = []
    for i in range(num_budget):
        tidx,_ = find_new_one(dg)
        alidx += [tidx]
        dg = update_dg(dg, tidx)
        
    return alidx


import pandas as pd
# official version
def est_probth(totfeas, num_class, thresh):
    
    dist = euclidean_distances(totfeas, totfeas, squared=False)
    kmeans = MiniBatchKMeans(n_clusters=num_class, batch_size=4096)
    cluster = kmeans.fit_predict(totfeas)
    
    thresh_dist = estimate_thresh(dist, cluster, thresh = thresh)#
    print('probcover purity ' + str(thresh) + ' is equal to dist ', thresh_dist)
    
    return thresh_dist

def construct_dg_2(totfeas, num_class, delta):
    
    #print('probcover purity ' + str(thresh) + ' is equal to dist ', thresh)
    
    batch_size = 500

    xs, ys, ds = [], [], []
    print(f'Start constructing graph using delta={delta}')
    # distance computations are done in GPU
    cuda_feats = torch.tensor(totfeas).cuda()
    for i in range(len(totfeas) // batch_size):
        # distance comparisons are done in batches to reduce memory consumption
        cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(cur_feats, cuda_feats)
        mask = dist < delta
        # saving edges using indices list - saves memory.
        x, y = mask.nonzero().T
        xs.append(x.cpu() + batch_size * i)
        ys.append(y.cpu())
        ds.append(dist[mask].cpu())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    print(f'Finished constructing graph using delta={delta}')
    print(f'Graph contains {len(df)} edges.')
    
    return df


def sampling_probcover_2(df, budgetSize, pool_size):
    
    # budgetSize = 2000
    relevant_indices = [i for i in range(pool_size)]#[i for i in range(len(totfeas))]

    selected = []
    # removing incoming edges to all covered samples from the existing labeled set
    edge_from_seen = np.isin(df.x, np.arange(0))
    covered_samples = df.y[edge_from_seen].unique()
    cur_df = df[(~np.isin(df.y, covered_samples))]

    for i in range(budgetSize):
        coverage = len(covered_samples) / len(relevant_indices)
        # selecting the sample with the highest degree
        degrees = np.bincount(cur_df.x, minlength=len(relevant_indices))
        print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
        cur = degrees.argmax()
        # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

        # removing incoming edges to newly covered samples
        new_covered_samples = cur_df.y[(cur_df.x == cur)].values
        assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
        cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

        covered_samples = np.concatenate([covered_samples, new_covered_samples])
        selected.append(cur)
        
    return selected


###badge
# def kmeans_plus(uidx, emb, num_budget):
    
#     newidx = [uidx[np.random.randint(0,len(uidx))]]
#     restidx = list( set(uidx) - set(newidx) )
#     for i in range(num_budget - 1):
#         dist = euclidean_distances(emb[newidx,:], emb[restidx,:], squared=True)###TODO avoid repeated dist comp. for efficiency
#         prob = ( dist.T / dist.sum(axis=1) ).T
#         newidx += np.random.choice(a=restidx, size=1, replace=False, p=prob[0,:]).tolist()
#         restidx = list( set(uidx) - set(newidx) )
    
#     return newidx


# def kmeans_plus(uidx, emb, num_budget):
    
#     dist = euclidean_distances(emb[uidx,:], emb[uidx,:], squared=True)
#     import time
#     print('embedding dist calculated')
    
#     uidx = np.array(uidx)
#     candidx = [i for i in range(len(uidx))]
    
#     newidx = [np.random.randint(0,len(candidx))]
#     restidx = list( set(candidx) - set(newidx) )
#     s0 = time.time()
#     s = s0
#     for i in range(num_budget - 1):
#         tdist = dist[np.ix_(newidx,restidx)]
#         tdist = tdist.min(axis=0)
#         prob = ( tdist.T / tdist.sum() ).T
#         newidx += np.random.choice(a=restidx, size=1, replace=False, p=prob).tolist()
#         restidx = list( set(candidx) - set(newidx) )
        
#         if i % int(num_budget/10) ==0:
#             print('kmeans++ sampling ', i, time.time()-s)
#             s = time.time()
#     newidx = uidx[newidx]
#     uidx = uidx.tolist()
    
#     return newidx.tolist()

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


###cluster_margin
def round_robin(hac_cluster, num_budget):
    
    k = num_budget
    cluster_list = []
    # print("Round Robin")
    for i in range(20):#hac n_clusters
        cluster = []
        cluster_list.append(cluster)
        
    for real_idx in range(len(hac_cluster)):
        i = hac_cluster[real_idx]
        cluster_list[i].append(real_idx)
    
    for i in range(len(cluster_list)):
        np.random.shuffle(cluster_list[i])
        
    cluster_list.sort(key=lambda x:len(x))
    
    index_select = []
    cluster_index = 0
    # print("Select cluster",len(set(hac_list)))
    while k > 0:
        if len(cluster_list[cluster_index]) > 0:
            index_select.append(cluster_list[cluster_index].pop(0)) 
            k -= 1
        if cluster_index < len(cluster_list) - 1:
            cluster_index += 1
        else:
            cluster_index = 0

    return index_select

def initial_hac(cand_feas):
    
    # from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import MiniBatchKMeans
    # from sklearn.neighbors import NearestCentroid
    
    # initial_cluster = AgglomerativeClustering(n_clusters = 20, linkage = 'ward').fit(cand_feas).labels_
    initial_cluster = MiniBatchKMeans(n_clusters=20).fit_predict(cand_feas)
    # clf = NearestCentroid()
    # clf.fit(cand_feas, initial_cluster)
    # initial_center = clf.centroids_
    
    return initial_cluster#initial_center, initial_cluster#

def cluster_margin_sampling(candidx, cand_feas, initial_center, initial_cluster, num_budget):
    
    dist = euclidean_distances(initial_center, cand_feas)
    cluster = initial_cluster[dist.argmin(axis=0)]
    
    ###sampling
    newidx = round_robin(cluster, num_budget)
    candidx = np.array(candidx)
    
    return candidx[newidx].tolist(), initial_cluster, initial_center


#learning loss
def est_loss(classifier, lossmodel, unlabeled_loader):
    classifier.eval()
    lossmodel.eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels, _) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            _, emb = classifier(inputs)
            # emb = classifier.emb
            pred_loss = lossmodel([inputs, emb]) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def learning_loss_sampling(uidx, loss, num_budget):
    
    idx = torch.sort(loss, descending=False).indices[-num_budget:]
    uidx = np.array(uidx)
    
    return uidx[idx].tolist()
        
