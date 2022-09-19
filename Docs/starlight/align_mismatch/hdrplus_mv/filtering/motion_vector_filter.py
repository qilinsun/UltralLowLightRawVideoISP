#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:13:25 2022

@author: gravitychen
"""

import numpy as np
from matplotlib import pyplot as plt

MVs = np.load("/Users/gravitychen/Desktop/motionVectors.npy")

def vector_filtering(MVs):
    for idx_mv_frame in range(len(MVs)-1):
        currrent_mv = MVs[idx_mv_frame]
        next_mv = MVs[idx_mv_frame+1]
    
        diff_mv = currrent_mv - next_mv
        diff_mv_length = (diff_mv[...,0]**2+diff_mv[...,1]**2)**0.5
        plt.imshow(diff_mv_length,vmin=0,vmax=90)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"vector_{idx_mv_frame}")
        # plt.show()
        # plt.pause(3)
        
        diff_mv_length_filter = diff_mv_length * (diff_mv_length<35)
        plt.imshow(diff_mv_length_filter,vmin=0,vmax=90)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"vector_filter_{idx_mv_frame}")
        # plt.show()
        # plt.pause(3)


vector_filtering(MVs)