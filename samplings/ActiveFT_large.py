#active fine-tuning code from https://github.com/yichen928/ActiveFT

import os
import numpy as np
import random
import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from utils import *
import math

torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10


class SampleModel(nn.Module):
    def __init__(self, features, sample_num, temperature, init, distance, balance=1.0, slice = None, batch_size = 100000, alidx = None):
        super(SampleModel, self).__init__()
        self.features = features
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance
        self.slice = slice
        if slice is None:
            self.slice = self.total_num
        self.batch_size = batch_size

        self.init = init
        self.distance = distance
        
        self.alidx = alidx

        centroids = self.init_centroids()
        if init == 'hybrid':
            self.centroids_alidx = centroids[:len(alidx),:]
            self.centroids_new = nn.Parameter(centroids[len(alidx):,:]).cuda()
            # self.centroids = torch.cat( [self.centroids_alidx, self.centroids_new], dim=0 )
        else:
            self.centroids = nn.Parameter(centroids).cuda()
            # print('new centers shape', self.centroids.shape)
        

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(sample_ids, self.sample_num)
        # elif self.init == "fps":
        #     dist_func = functools.partial(get_distance, type=self.distance)
        #     sample_ids = farthest_distance_sample(self.features, self.sample_num, dist_func)
        elif self.init == 'hybrid':
            sample_ids = self.alidx.copy()
            new = [i for i in range(self.total_num) if i not in sample_ids]
            sample_ids += random.sample(new, self.sample_num - len(self.alidx))

        centroids = self.features[sample_ids].clone()
        return centroids

    def get_loss(self):
        if self.init == 'hybrid':
            centroids0 = F.normalize(self.centroids_alidx, dim=1)
            centroids1 = F.normalize(self.centroids_new, dim=1)
            centroids = torch.cat( [centroids0, centroids1], dim=0 )
        else:
            centroids = F.normalize(self.centroids, dim=1)
        
        ### small dataset
        # prod = torch.matmul(self.features, centroids.transpose(1, 0))  # (n, k)
        # prod = prod / self.temperature
        # prod_exp = torch.exp(prod)
        # prod_exp_pos, pos_k = torch.max(prod_exp, dim=1)  # (n, )

        # cent_prod = torch.matmul(centroids.detach(), centroids.transpose(1, 0))  # (k, k)
        # cent_prod = cent_prod / self.temperature
        # cent_prod_exp = torch.exp(cent_prod)
        # cent_prob_exp_sum = torch.sum(cent_prod_exp, dim=0)  # (k, )
        
        
        ### for large dataset
        sample_ids = list(range(self.total_num))
        sample_ids = random.sample(sample_ids, self.batch_size)
        features = self.features[sample_ids]
        sample_slice_num = math.ceil(1.0 * self.sample_num / self.slice)
        batch_slice_num = math.ceil(1.0 * self.batch_size / self.slice)    
        
        prod_exp_pos = []
        pos_k = []
        for sid in range(batch_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            prod = torch.matmul(features[start: end], centroids.transpose(1, 0))  # (slice_num, k)
            prod = prod / self.temperature
            prod_exp = torch.exp(prod)
            prod_exp_pos_cur, pos_k_cur = torch.max(prod_exp, dim=1)  # (slice_num, )
            prod_exp_pos.append(prod_exp_pos_cur)
            pos_k.append(pos_k_cur)
        pos_k = torch.cat(pos_k, dim=0)
        prod_exp_pos = torch.cat(prod_exp_pos, dim=0)

        cent_prob_exp_sum = []
        for sid in range(sample_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            cent_prod = torch.matmul(centroids.detach(), centroids[start:end].transpose(1, 0))  # (k, slice_num)
            cent_prod = cent_prod / self.temperature
            cent_prod_exp = torch.exp(cent_prod)
            cent_prob_exp_sum_cur = torch.sum(cent_prod_exp, dim=0)  # (slice_num, )
            cent_prob_exp_sum.append(cent_prob_exp_sum_cur)
        cent_prob_exp_sum = torch.cat(cent_prob_exp_sum, dim=0)

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J

def optimize_dist(features, sample_num, args, alidx):
    #  features: (n, c)
    sample_model = SampleModel(features, sample_num, args.activeft_temperature, args.activeft_init, args.activeft_distance, args.activeft_balance, args.activeft_slice, args.activeft_batch_size, alidx)
    sample_model = sample_model.cuda()

    optimizer = optim.Adam(sample_model.parameters(), lr=args.activeft_lr)
    if args.activeft_scheduler != "none":
        if args.activeft_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.activeft_max_iter, eta_min=1e-6)
        else:
            raise NotImplementedError

    for i in range(args.activeft_max_iter):
        loss = sample_model.get_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.activeft_scheduler != "none":
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print("Iter: %d, lr: %.6f, loss: %f" % (i, lr, loss.item()))

    if args.activeft_init == 'hybrid':
        centroids = torch.cat( [sample_model.centroids_alidx, sample_model.centroids_new], dim=0 )
    else:    
        centroids = sample_model.centroids
        
    # centroids = F.normalize(centroids.detach(), dim=1)
    # dist = torch.matmul(centroids, features.transpose(1, 0))  # (k, n)
    # dist = dist.cpu().numpy()
    # ids_sort = (-dist).argsort(axis=1)
    
    # #keep sample_ids order
    # sample_ids = []
    # for i in range(ids_sort.shape[0]):
    #     for j in range(ids_sort.shape[1]):
    #         if ids_sort[i, j].item() not in sample_ids:
    #             # sample_ids.add(ids_sort[i, j].item())
    #             sample_ids += [ids_sort[i, j].item()]
    #             break
            
    ###slice version for large dataset
    centroids = F.normalize(centroids.detach(), dim=1)
    slice = 3000
    sample_slice_num = math.ceil(centroids.shape[0] / slice)
    # sample_ids = set()
    sample_ids = []
    # _, ids_sort = torch.sort(dist, dim=1, descending=True)
    for sid in range(sample_slice_num):
        start = sid * slice
        end = min((sid + 1) * slice, centroids.shape[0])
        dist = torch.matmul(centroids[start:end], features.transpose(1, 0))  # (slice_num, n)
        # _, ids_sort = torch.sort(dist, dim=1, descending=True)
        dist = dist.cpu().numpy()
        ids_sort = (-dist).argsort(axis=1)
        for i in range(ids_sort.shape[0]):
            for j in range(ids_sort.shape[1]):
                if ids_sort[i, j].item() not in sample_ids:
                    # sample_ids.add(ids_sort[i, j].item())
                    sample_ids += [ids_sort[i, j].item()]
                    break            
            

    print(len(sample_ids))
    # sample_ids = list(sample_ids)
    return sample_ids


def ActiveFT_sampling(features, num_budget, alidx, args):
    
    features = torch.Tensor(features).cuda()
    features = F.normalize(features, dim=1)
    
    if len(alidx) == 0:
        alidx = optimize_dist(features, num_budget, args, alidx)
    else:
        args.activeft_init = 'hybrid'
        alidx = optimize_dist(features, num_budget + len(alidx), args, alidx)
    
    return alidx


# ###test
# parser = argparse.ArgumentParser('Visualize extracted features')
# parser.add_argument('--feature_path', default='C:\\document\\code\\AS4L\\pretrain_encoder\\totfeas_byol_eman.npy', type=str,
#                     help='path of saved features')
# parser.add_argument('--activeft_temperature', default=0.07, type=float, help='temperature for softmax')
# parser.add_argument('--activeft_max_iter', default=2, type=int, help='max iterations')#100
# parser.add_argument('--activeft_lr', default=0.001, type=float, help='learning rate')
# parser.add_argument('--activeft_init', default='random', type=str, choices=['random', 'fps'])
# parser.add_argument('--activeft_distance', default='euclidean', type=str, help='euclidean or cosine')
# parser.add_argument('--activeft_scheduler', default='none', type=str, help='scheduler')
# parser.add_argument('--activeft_balance', default=1.0, type=float, help='balance ratio')
# parser.add_argument('--activeft_batch_size', default=25000, type=int, help='batch size for SGD')
# parser.add_argument('--activeft_slice', default=100, type=int, help='size of slice to save memory')
# args = parser.parse_args()

# features = np.load(args.feature_path)

# features = features[:640000,:]
# features = torch.Tensor(features)
# features = F.normalize(features, dim=1).cuda()

# alidx = optimize_dist(features, 10000, args, [])
