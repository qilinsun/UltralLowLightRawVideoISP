import numpy as np
import torch
import torch.nn.functional as F
import os
import rawpy
from glob import glob
import cv2
from numba_PM_v2 import NNS, reconstruction
import matplotlib.pyplot as plt
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfileBM3D, BM4DProfileBM3DComplex, BM4DProfileComplex
from experiment_funcs import generate_noise, get_experiment_kernel, get_psnr, get_cropped_psnr_3d
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, estimate_sigma, denoise_nl_means, denoise_tv_bregman
from einops import rearrange, repeat
from bm3d import bm3d, BM3DProfile, BM3DStages
from est_noise import noise_estimate
from bm3d_1st_step import bm3d_1st_step
import flowpy

raw_path = '/media/cuhksz-aci-03/数据/CUHK_SZ/motion/0.1/'
seq_id = "iso2500"
star_id = 0
num_load = 1

class AutoGammaCorrection: # img in HSV Space
    def __init__(self, img):
        self.img = img
    def execute(self):
        img_h = self.img.shape[0]
        img_w = self.img.shape[1]
        Y = self.img[:,:,2].copy()
        gc_img = np.zeros((img_h, img_w), np.float32)
        Y = Y.astype(np.float32)/255.  # 255. is better right?
        Yavg = np.exp(np.log(Y+1e-20).mean())
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                if Y[y, x]>0:
                    gc_img[y, x] = np.log(Y[y,x]/Yavg + 1)/np.log(1/Yavg+1)
                # if x==1500:
                #     print(x, Y[y, x], gc_img[y, x])
        return gc_img


def pack_raw(raw):

    im = raw.raw_image_visible.astype(np.float32)
    # im = im[1104:2384, 1080:3240] # 0.1 still iso2500
    im = im[660:1940, 1480:3640] # 0.1 motion iso2500
    # im = im[912:2192, 1204:3364] # 0.01 still iso25600
    # im = im[720:2000, 724:2884] # 0.01 motion iso51200

    return im

rawpyParam = {
            'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
            'half_size': False,
            'use_camera_wb' : True,
            'use_auto_wb' : False,
            'no_auto_bright': True,
            'output_color': rawpy.ColorSpace.sRGB,
            'output_bps' : 16}

dark_raw_path = '/home/cuhksz-aci-03/Desktop/dark_fpn/iso2500/'
# load image
darkraw_all = glob(dark_raw_path + '*.ARW')
total_num = len(darkraw_all)
ind = []
for i in range(0,len(darkraw_all)):
    ind.append(int(darkraw_all[i].split('/')[-1].split('.')[0]))
ind = np.argsort(np.array(ind))
filepaths_all_sorted = np.array(darkraw_all)[ind]
raw_avg = 0
for i in range(total_num):
    raw_image_read = rawpy.imread(filepaths_all_sorted[i])
    raw_image = raw_image_read.raw_image_visible.astype(np.float32)
    raw_avg = raw_image + raw_avg

raw_avg = raw_avg[660:1940, 1480:3640] / 58

def load_video_seq(folder_name, seqID, start_ind, num_to_load):
    # base_name_seq = folder_name + 'seq' + str(seqID) + '/'  # starlight
    base_name_seq = folder_name + str(seqID) + '/'
    filepaths_all = glob(base_name_seq + '*.ARW')
    total_num = len(filepaths_all)

    ind = []
    for i in range(0, len(filepaths_all)):
        ind.append(int(filepaths_all[i].split('/')[-1].split('.')[0]))
    ind = np.argsort(np.array(ind))
    filepaths_all_sorted = np.array(filepaths_all)[ind]

    if num_to_load == 'all':
        num_to_load = total_num
        print('loading ', num_to_load, 'frames')
    # full_im = np.empty((num_to_load, 540, 960, 4))  # DRV 1836 2748
    full_im = np.empty((num_to_load, 1280, 2160))
    # full_im = np.empty((num_to_load, 568, 628))
    for i in range(0, num_to_load):
        raw_img = rawpy.imread(filepaths_all_sorted[start_ind + i])
        full_im[i] = pack_raw(raw_img)

    return full_im

input_seq = load_video_seq(raw_path, seq_id, star_id, num_load)
input_seq_1 = load_video_seq(raw_path, seq_id, star_id, num_load)

nHard = 19
kHard = 8
NHard = 20
pHard = 3
lambdaHard3D = 2.7 # 2.7  # ! Threshold for Hard Thresholding
tauMatchHard = 500000   # ! threshold determinates similarity between patches
useSD_h = False
tau_2D_hard = 'BIOR'

def symetrize(img, N):
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad

for num in range(len(input_seq)):
    for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):


        nv = noise_estimate(raw_avg[di::2, dj::2])

        noise_img = symetrize(input_seq[num, di::2, dj::2], nHard)


        pre_img = bm3d_1st_step(nv, noise_img, nHard, kHard, NHard, pHard, lambdaHard3D, tauMatchHard, useSD_h,
                              tau_2D_hard)

        input_seq_1[num, di::2, dj::2] = pre_img[nHard: -nHard, nHard: -nHard]

        input_seq_1[num, di::2, dj::2] = np.clip(input_seq_1[num, di::2, dj::2], 0,  16383)

raw = '/media/cuhksz-aci-03/数据/CUHK_SZ/motion/0.1/iso2500/03568.ARW'
raw = rawpy.imread(raw)


raw.raw_image_visible[660:1940, 1480:3640] = (input_seq_1[0])
post_raw = raw.postprocess(**rawpyParam)
cfa = (post_raw[660:1940, 1480:3640] / 65535. * 255.).astype(np.float32)  # scale and set type for cv2
hsv = cv2.cvtColor(cfa, cv2.COLOR_RGB2HSV)
hsvOperator = AutoGammaCorrection(hsv)
enhanceV = hsvOperator.execute()
hsv[..., -1] = enhanceV * 255.
enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
# enhanceRGB = edge_enhancement(enhanceRGB)
enhanceRGB = cv2.cvtColor(enhanceRGB, cv2.COLOR_RGB2BGR)

img_write = cv2.normalize(((enhanceRGB)**(1/2.2)), dst=None, alpha=0, beta=255,
              norm_type=cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/20230529/predenoisingv1_motion_0.1_2500' + '.png', (img_write))
