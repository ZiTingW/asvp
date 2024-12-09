# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:28:37 2024

@author: wenzt
"""

import matplotlib
import matplotlib.pyplot as plt
font = {'size': 22, 'family': 'Helvetica'}
plt.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['axes.unicode_minus'] = False
# font = {'size': 26, 'family': 'Helvetica'}
plt.rc('font', **font)

import numpy as np

def plot_feas_w_lbl(X_embed_wrn, totlabel, classname = None, lblidx = None, outputpath = None):

    #classname = {0:'plane', 1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
    class_feas = {}
    for i in range(len(X_embed_wrn)):
        if totlabel[i] in class_feas:
            class_feas[totlabel[i]] = np.vstack(( class_feas[totlabel[i]], X_embed_wrn[i,:]))
        else:
            class_feas[totlabel[i]] = np.array([X_embed_wrn[i,:]])


    plt.figure(figsize=(12, 12))
    for i in range(max(totlabel)+1):
        if i in class_feas:
            if classname is None:
                plt.plot(class_feas[i][:,0], class_feas[i][:,1], '+', markersize = 4)
                # plt.plot(class_feas[i][:,0], class_feas[i][:,1], '+', label = 'cluster' + str(i), markersize = 4)
            else:
                plt.plot(class_feas[i][:,0], class_feas[i][:,1], '.', label = classname[i], markersize = 4)
    #plt.plot(X_embedded_pl03[lbl_idx20,0], X_embedded_pl03[lbl_idx20,1],'k+',label = 'labeled')

    #idx = list( set(lbl_idx30) - set(al_lbl_idx) )
    if lblidx is not None:
        if classname is None:
            plt.plot(X_embed_wrn[lblidx,0], X_embed_wrn[lblidx,1],'k*', markersize = 12)
        else:
            plt.plot(X_embed_wrn[lblidx,0], X_embed_wrn[lblidx,1],'k*',label = 'labeled samples', markersize = 12)
    #plt.plot(X_embed_wrn[new,0], X_embed_wrn[new,1],'k*',label = 'new')
    
    if classname is not None:
        plt.legend(prop=font, bbox_to_anchor=(1, 0.75), markerscale=4)
    # plt.xlabel('frame number',font)
    # plt.ylabel("error(degree)",font)
    #plt.title('cifar10, self-supervised feature',font)
    #plt.grid()
    plt.axis('off')
    plt.tick_params(labelsize=23)
    
    if outputpath is not None:
        # plt.savefig(outputpath + '.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(outputpath + '.png')

    return