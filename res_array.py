# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:30:40 2023

@author: wenzt
"""

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path_base', default = './', type=str,
                    help='AL, badge_partition_subset, the size of subset')
parser.add_argument('--dataset_name', default = 'c10', type=str,
                    help='AL, badge_partition_subset, the size of subset')
parser.add_argument('--prefix1', default = 'c10', type=str,
                    help='AL, badge_partition_subset, the size of subset')
parser.add_argument('--prefix2', default = '_r50byoleman_mlpproxy_lpft_training_strategy3', type=str,
                    help='AL, badge_partition_subset, the size of subset')
parser.add_argument('--num_exp', default = '[i for i in range(1,6)]', type=str,
                    help='AL, badge_partition_subset, the size of subset')
parser.add_argument('--res_name', default = 'totacc_margin_r50byoleman_mlpproxy_lpft.npy', type=str,
                    help='AL, badge_partition_subset, the size of subset')

args = parser.parse_args()

num_exp = eval(args.num_exp)

totacc = []
for i in num_exp:
    path = os.path.join(args.path_base, args.dataset_name, args.prefix1 + str(i) + args.prefix2, 'acc.npy')
    t = np.load(path)
    totacc += [t]
    
np.save( os.path.join(args.path_base, args.dataset_name, args.res_name), np.array(totacc))

