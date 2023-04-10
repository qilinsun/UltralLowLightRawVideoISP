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


raw_path = '/media/cuhksz-aci-03/数据/CUHK_SZ/still/0.01/'
seq_id = "iso25600"
star_id = 0
num_load = 16
p_size = 5
itr = 1
img_match = []
re_img_match = []


def gamma_correction(img, c=1, g=2.2):
    out = img.copy()
    out /= 16383.
    out = (1/c * out) ** (1/g)
    out *= 16383
    out = out.astype(np.float32)

    return out

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
    im = im[912:2192, 1204:3364]

    return im

rawpyParam = {
            'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
            'half_size': False,
            'use_camera_wb' : True,
            'use_auto_wb' : False,
            'no_auto_bright': True,
            'output_color': rawpy.ColorSpace.sRGB,
            'output_bps' : 16}

dark_raw_path = '/home/cuhksz-aci-03/Desktop/dark_fpn/iso25600/'
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

raw_avg = raw_avg[912:2192, 1204:3364] / 66

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
    for i in range(0, num_to_load):
        raw_img = rawpy.imread(filepaths_all_sorted[start_ind + i])
        full_im[i] = pack_raw(raw_img)

    return full_im

input_seq = load_video_seq(raw_path, seq_id, star_id, num_load)
input_seq_1 = load_video_seq(raw_path, seq_id, star_id, num_load)

nHard = 16
kHard = 8
NHard = 16
pHard = 3
lambdaHard3D = 2.7 # 2.7  # ! Threshold for Hard Thresholding
tauMatchHard = 250000   # ! threshold determinates similarity between patches
useSD_h = False
tau_2D_hard = 'BIOR'

for num in range(len(input_seq)):
    for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.

        nv = noise_estimate(raw_avg[di::2, dj::2])
        input_seq[num, di::2, dj::2] = bm3d_1st_step(nv, input_seq[num, di::2, dj::2], nHard, kHard, NHard, pHard, lambdaHard3D, tauMatchHard, useSD_h,
                              tau_2D_hard)
        input_seq[num, di::2, dj::2] = np.clip(input_seq[num, di::2, dj::2], 0,  16383)

img_match.append((input_seq[0]))
for i in range(input_seq.shape[0] - 1):
    img_ori = np.expand_dims((input_seq[0, ...]), axis=-1)

    merge_h, merge_w = input_seq[0, ...].shape
    img_ori_rggb = np.concatenate((img_ori[0:merge_h:2, 0:merge_w:2, :],  # R
                                       img_ori[0:merge_h:2, 1:merge_w:2, :],  # G
                                       img_ori[1:merge_h:2, 1:merge_w:2, :],  # B
                                       img_ori[1:merge_h:2, 0:merge_w:2, :]), axis=-1)  # G

    img = gamma_correction(img_ori_rggb)

    ref_ori = np.expand_dims((input_seq[i + 1, ...]), axis=-1)

    ref_merge_h, ref_merge_w = input_seq[i + 1, ...].shape
    ref_ori_rggb = np.concatenate((ref_ori[0:ref_merge_h:2, 0:ref_merge_w:2, :],  # R
                                       ref_ori[0:ref_merge_h:2, 1:ref_merge_w:2, :],  # G
                                       ref_ori[1:ref_merge_h:2, 1:ref_merge_w:2, :],  # B
                                       ref_ori[1:ref_merge_h:2, 0:ref_merge_w:2, :]), axis=-1)  # G

    ref = gamma_correction(ref_ori_rggb)
    f, dist, score_list = NNS(img, ref, p_size, itr)
    img_recon = reconstruction(f, img_ori_rggb, ref_ori_rggb)
    img_recon_1 = np.empty([merge_h, merge_w])
    img_recon_1[::2, ::2] = img_recon[..., 0]  # R
    img_recon_1[::2, 1::2] = img_recon[..., 1]  # G
    img_recon_1[1::2, ::2] = img_recon[..., 3]  # G
    img_recon_1[1::2, 1::2] = img_recon[..., 2]  # B

    img_match.append((img_recon_1))
y = np.array(img_match)
y = y.transpose(1, 2, 0)

for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
    noise_type = 'g1'
    noise_var = noise_estimate(np.sqrt(y[di::2, dj::2, 0]+(3/8)), 4)
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    # kernel = get_experiment_kernel(noise_type, noise_var)
    # noise, psd, kernel = generate_noise(kernel, seed, y[di::2, dj::2, :].shape)
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    # z = np.atleast_3d(y[di::2, dj::2, :]) + np.atleast_3d(noise)

    # Call BM4D With the default settings.
    # y[di::2, dj::2, :] = bm4d(z, psd)

    # For 8x8x1 blocks instead of 4x4x4/5x5x5
    # y_est = bm4d(z, psd, '8x8')

    # To include refiltering:
    # y_est = bm4d(z, psd, 'refilter')

    # For other settings, use BM4DProfile.
    profile = BM4DProfile() # equivalent to profile = BM4DProfile('np');
    # profile.set_sharpen(1.2)
    profile.gamma = 2.2  # redefine value of gamma parameter
    y_est = bm4d(np.sqrt(y[di::2, dj::2, :]+(3/8)), (noise_var), profile, BM4DStages.ALL_STAGES)
    y[di::2, dj::2, :] = y_est
    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm4d(z, sqrt(noise_var))

    # print("PSNR: ", get_psnr(y_est, y))

    # plt.title("y, z, y_est")


    # disp_mat = np.concatenate((y[:, :, i], np.squeeze(z[:, :, i]),
    #                                y_est[:, :, i]), axis=1)
    # plt.imshow(np.squeeze(disp_mat))
    # plt.show()



raw = '/media/cuhksz-aci-03/数据/CUHK_SZ/still/0.01/iso25600/03861.ARW'
raw = rawpy.imread(raw)

raw.raw_image_visible[912:2192, 1204:3364] = (y[..., 0]**2-(3/8))
post_raw = raw.postprocess(**rawpyParam)
cfa = (post_raw[912:2192, 1204:3364] / 65535. * 255.).astype(np.float32)  # scale and set type for cv2
hsv = cv2.cvtColor(cfa, cv2.COLOR_RGB2HSV)
hsvOperator = AutoGammaCorrection(hsv)
enhanceV = hsvOperator.execute()
hsv[..., -1] = enhanceV * 255.
enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
# enhanceRGB = edge_enhancement(enhanceRGB)
enhanceRGB = cv2.cvtColor(enhanceRGB, cv2.COLOR_RGB2BGR)

img_write = cv2.normalize(((enhanceRGB)**(1/2.2)), dst=None, alpha=0, beta=255,
              norm_type=cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/downup_framev2_' + str(0) + '.png', (img_write))
