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

# sys.path.append("./fast-openISP")
# from pipeline import Pipeline
# from util.yacs import Config


# def starlightFastISP(rawIn):
#     """
#     Usage:
#         rgb = starFastISP(packing[0])
#             input shape : (rawH,rawW)
#     Keys of pipeline.execute output:
#         (['bayer', 'rgb_image', 'y_image', 'cbcr_image', 'edge_map', 'output'])
#     """
#     cfg = Config('./fast-openISP/configs/tiff.yaml')
#     pipeline = Pipeline(cfg)
#
#     rawIn = rawIn.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
#     rawOut, _ = pipeline.execute(rawIn)
#
#     return rawOut

raw_path = '/media/cuhksz-aci-03/数据/CUHK_SZ/still/0.01/'
seq_id = "iso25600"
star_id = 0
num_load = 16
p_size = 5
itr = 1
img_match = []
re_img_match = []

# def gammasRGB(image, mode='compress'):
# 	'''sRGB transfer function'''
# 	if mode == 'compress':
# 		if np.issubdtype(image.dtype, np.unsignedinteger):
# 			return uGammaCompress_(image, 0.0031308, 12.92, 1.055, 1. / 2.4)
# 		else:
# 			return fGammaCompress_(image, 0.0031308, 12.92, 1.055, 1. / 2.4)
# 	else:
# 		if np.issubdtype(image.dtype, np.unsignedinteger):
# 			return uGammaDecompress_(image, 0.04045, 12.92, 1.055, 2.4)
# 		else:
# 			return fGammaDecompress_(image, 0.04045, 12.92, 1.055, 2.4)

def edge_enhancement(rgb_img):
    rgb_img = rgb_img/255.
    rgb_img = rearrange(rgb_img, 'h w c -> 1 c h w')

    kernel = torch.tensor([[0, -1, 0], [-1, 5., -1], [0, -1, 0]])
    kernel = repeat(kernel, 'h w -> c 1 h w', c=3)

    ee_img = F.conv2d(torch.tensor(rgb_img), kernel, padding=1, groups=3)
    ee_img = ee_img.detach().numpy().squeeze()
    ee_img = rearrange(ee_img, 'c h w -> h w c')
    ee_img = ((ee_img - ee_img.min()) / (ee_img.max() - ee_img.min())) * 255.

    return ee_img

def gamma_correction(img, c=1, g=2.2):
    out = img.copy()
    out /= 16383.
    out = (1/c * out) ** (1/g)
    out *= 16383
    out = out.astype(np.float32)

    return out

def pyramid_downsample(image):
    level = 3
    temp = image.copy()

    for i in range(level):
        dst = cv2.pyrDown(temp)
        temp = dst.copy()

    return dst
#
# def lapalian_upsample(image):

def image_interpolation(img,new_dimension,inter_method):
    inter_img = cv2.resize(img,new_dimension,interpolation=inter_method)
    return inter_img

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

def vevid_simple(rawimg, G, bais):
    out_img = np.arctan2(-G * (rawimg + bais), rawimg)

    return out_img

def pack_raw(raw):

    im = raw.raw_image_visible.astype(np.float32)
    # im = im[1380:1892, 1840:2352]
    # im = im[1104:2384, 1080:3240]
    # im = im[980:1492, 1800:2312]
    im = im[912:2192, 1204:3364]
    # im = im[1680:2192, 2852:3364]

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


# nv = 536.7693
# for num in range(len(input_seq)):
#     # noise_variance = estimate_sigma((input_seq[num, ...]), multichannel=True)
#     # noise_variance = np.sqrt(0.002 * np.max(np.sqrt(input_seq[num, ...] + (3 / 8))) ** 2)
#     mergedImage_exp = np.expand_dims((input_seq[num, ...]), axis=-1)
#     raw_avg = np.expand_dims((raw_avg), axis=-1)
#     merge_h, merge_w = input_seq[num].shape
#     mergedImage_exp = mergedImage_exp
#     merge_rggb = np.concatenate((mergedImage_exp[0:merge_h:2, 0:merge_w:2, :],  # R
#                                  mergedImage_exp[0:merge_h:2, 1:merge_w:2, :],  # G
#                                  mergedImage_exp[1:merge_h:2, 1:merge_w:2, :],  # B
#                                  mergedImage_exp[1:merge_h:2, 0:merge_w:2, :]), axis=-1)  # G
#
#     noise_rggb = np.concatenate((raw_avg[0:merge_h:2, 0:merge_w:2, :],  # R
#                                  raw_avg[0:merge_h:2, 1:merge_w:2, :],  # G
#                                  raw_avg[1:merge_h:2, 1:merge_w:2, :],  # B
#                                  raw_avg[1:merge_h:2, 0:merge_w:2, :]), axis=-1)
#     # denoise_img = denoise_tv_chambolle((merge_rggb), weight=0.08, n_iter_max=200, multichannel=True)
#     # denoise_img = denoise_wavelet((merge_rggb), noise_variance, multichannel=True)
#     profile = BM3DProfile()
#     profile.gamma = 2.2
#     pre_img = bm3d((merge_rggb), noise_rggb, profile, BM3DStages.HARD_THRESHOLDING)
#     mergimg = np.empty([merge_h, merge_w])
#     mergimg[::2, ::2] = pre_img[..., 0]  # R
#     mergimg[::2, 1::2] = pre_img[..., 1]  # G
#     mergimg[1::2, ::2] = pre_img[..., 3]  # G
#     mergimg[1::2, 1::2] = pre_img[..., 2]  # B
#
#     input_seq[num, ...] = mergimg

# for num in range(len(input_seq)):
#     for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
#         # noise_var = estimate_sigma(np.sqrt(input_seq[num, di::2, dj::2] + (3 / 8)))
#         noise_var = np.sqrt(0.0001 * np.max(np.sqrt(input_seq[num, di::2, dj::2] + (3 / 8))) ** 2)
#         profile = BM3DProfile()
#         profile.gamma = 6
#         pre_img = bm3d(np.sqrt(input_seq[num, di::2, dj::2] + (3 / 8)), noise_var, profile, BM3DStages.HARD_THRESHOLDING)
#         input_seq[num, di::2, dj::2] = pre_img**2-(3/8)

# img_match.append((input_seq[0]))
# img_ori_match.append(np.sqrt(input_seq_1[0]+(3/8)))


# img_match_array = np.array(img_match)
# y = img_match_array.transpose(1, 2, 0)
# y_1 = img_match_array.transpose(1, 2, 0)

# img_match_array_1 = np.array(img_ori_match)
# y_1 = img_match_array_1.transpose(1, 2, 0)
# y = input_seq.transpose(1, 2, 0)

# for num in range(len(img_match_array)):
#     for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
#         # noise_variance = estimate_sigma(np.sqrt(y[di::2, dj::2, num]+(3/8)))
#         # y[di::2, dj::2, num] = (denoise_wavelet(np.sqrt(y[di::2, dj::2, num]+(3/8)), noise_variance))
#         # y[di::2, dj::2, num] = (c(np.sqrt(y[di::2, dj::2, num]+(3/8)), weight=0.5, n_iter_max=100))
#         y[di::2, dj::2, num] = (
#             denoise_nl_means(np.sqrt(y[di::2, dj::2, num] + (3 / 8)), 8, 5, 0.1))

# for num in range(len(input_seq)):
#     # noise_variance = estimate_sigma((input_seq[num, ...]), multichannel=True)
#     # noise_variance = np.sqrt(0.002 * np.max(np.sqrt(input_seq[num, ...] + (3 / 8))) ** 2)
#     mergedImage_exp = np.expand_dims((y[..., num]), axis=-1)
#     raw_avg = np.expand_dims((raw_avg), axis=-1)
#     merge_h, merge_w = y[..., num].shape
#     mergedImage_exp = mergedImage_exp
#     merge_rggb = np.concatenate((mergedImage_exp[0:merge_h:2, 0:merge_w:2, :],  # R
#                                  mergedImage_exp[0:merge_h:2, 1:merge_w:2, :],  # G
#                                  mergedImage_exp[1:merge_h:2, 1:merge_w:2, :],  # B
#                                  mergedImage_exp[1:merge_h:2, 0:merge_w:2, :]), axis=-1)  # G
#
#     noise_rggb = np.concatenate((raw_avg[0:merge_h:2, 0:merge_w:2, :],  # R
#                                  raw_avg[0:merge_h:2, 1:merge_w:2, :],  # G
#                                  raw_avg[1:merge_h:2, 1:merge_w:2, :],  # B
#                                  raw_avg[1:merge_h:2, 0:merge_w:2, :]), axis=-1)
#     # denoise_img = denoise_tv_chambolle((merge_rggb), weight=0.08, n_iter_max=200, multichannel=True)
#     # denoise_img = denoise_wavelet((merge_rggb), noise_variance, multichannel=True)
#     profile = BM3DProfile()
#     profile.gamma = 2.2
#     pre_img = bm3d((merge_rggb), noise_rggb, profile, BM3DStages.HARD_THRESHOLDING)
#     mergimg = np.empty([merge_h, merge_w])
#     mergimg[::2, ::2] = pre_img[..., 0]  # R
#     mergimg[::2, 1::2] = pre_img[..., 1]  # G
#     mergimg[1::2, ::2] = pre_img[..., 3]  # G
#     mergimg[1::2, 1::2] = pre_img[..., 2]  # B
#
#     y[..., num] = (mergimg)
sigma = 20
nHard = 16
kHard = 8
NHard = 16
pHard = 3
lambdaHard3D = 2.7 # 2.7  # ! Threshold for Hard Thresholding
tauMatchHard = 250000 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
useSD_h = False
tau_2D_hard = 'BIOR'

for num in range(len(input_seq)):
    for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.

        # profile_bm3d = BM4DProfileBM3D()
        # # profile_bm3d.set_sharpen(1.2)
        nv = noise_estimate(raw_avg[di::2, dj::2])
        # nv1 = noise_estimate(input_seq[num, di::2, dj::2])
        # pre_img = bm4d(input_seq[num, di::2, dj::2], nv, profile_bm3d, BM4DStages.HARD_THRESHOLDING)
        # pre_img, blockmatch = bm4d(input_seq[num, di::2, dj::2], raw_avg[di::2, dj::2], profile_bm3d, BM4DStages.HARD_THRESHOLDING, (True, False))
        # input_seq[num, di::2, dj::2] = np.squeeze(pre_img, -1)
        # # y[di::2, dj::2, num] = pre_img

        input_seq[num, di::2, dj::2] = bm3d_1st_step(nv, input_seq[num, di::2, dj::2], nHard, kHard, NHard, pHard, lambdaHard3D, tauMatchHard, useSD_h,
                              tau_2D_hard)
        input_seq[num, di::2, dj::2] = np.clip(input_seq[num, di::2, dj::2], 0,  16383)
# y = input_seq.transpose(1, 2, 0)

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
# y_1 = np.array(img_match)
# y_1 = y_1.transpose(1, 2, 0)


for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
    noise_type = 'g1'
    # lambdaS = 128.5632
    # lambdaR = 677036.032
    # noise_var = np.sqrt(2e-2*np.max(np.sqrt(y[di::2, dj::2, 0]+(3/8))) ** 2) # Noise variance
    # noise_var = estimate_sigma(np.sqrt(y[di::2, dj::2, 0]+(3/8)))
    noise_var = noise_estimate(np.sqrt(y[di::2, dj::2, 0]+(3/8)), 4)
    # noise_var = 164.95/65535*16383
    # noise_var = 104.45/65535*16383
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


# for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
#             noise_variance = estimate_sigma((y[di::2, dj::2, 0]))
#             y[di::2, dj::2, 0] = (
#                 denoise_wavelet((y[di::2, dj::2, 0]), noise_variance))
#             y[di::2, dj::2, 0] = (
#                             denoise_bilateral((y[di::2, dj::2, 0]), 0.05, 15))


# # noise_variance = estimate_sigma(np.sqrt(input_seq[num, ...] + (3 / 8)))
# mergedImage_exp = np.expand_dims((y[:, :, 0]), axis=-1)
# merge_h, merge_w = y[:, :, 0].shape
# mergedImage_exp = mergedImage_exp
# merge_rggb = np.concatenate((mergedImage_exp[0:merge_h:2, 0:merge_w:2, :],  # R
#                             mergedImage_exp[0:merge_h:2, 1:merge_w:2, :],  # G
#                             mergedImage_exp[1:merge_h:2, 1:merge_w:2, :],  # B
#                             mergedImage_exp[1:merge_h:2, 0:merge_w:2, :]), axis=-1)  # G
# denoise_img = denoise_tv_bregman((merge_rggb), weight=70, max_iter=200, eps=0.001, multichannel=True)
# mergimg = np.empty([merge_h, merge_w])
# mergimg[::2, ::2] = denoise_img[..., 0]  # R
# mergimg[::2, 1::2] = denoise_img[..., 1]  # G
# mergimg[1::2, ::2] = denoise_img[..., 3]  # G
# mergimg[1::2, 1::2] = denoise_img[..., 2]  # B

# y[:, :, 0] = mergimg

raw = '/media/cuhksz-aci-03/数据/CUHK_SZ/still/0.01/iso25600/03861.ARW'
raw = rawpy.imread(raw)

compression = 3.8
gain = 1.1
contrast = 1.0

# white_balance = raw.camera_whitebalance
# print('white balance', white_balance)
# white_balance_r = white_balance[0] / white_balance[1]
# white_balance_g0 = 1
# white_balance_g1 = 1
# white_balance_b = white_balance[2] / white_balance[1]
# cfa_pattern = raw.raw_pattern
# cfa_pattern = decode_pattern(cfa_pattern)
# ccm = raw.color_matrix
# black_point = int(raw.black_level_per_channel[0])
# white_point = int(raw.white_level)

# final_img = finish_image((y[..., 0]**2-(3/8)), y.shape[1], y.shape[0], black_point, white_point, white_balance_r, white_balance_g0, white_balance_g1,
#                  white_balance_b, compression, gain, contrast, cfa_pattern, ccm)

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
# yuv_ori = cv2.cvtColor(img_write.astype(np.uint8), cv2.COLOR_RGB2YUV_I420)
# filter_rgb = cv2.ximgproc.guidedFilter(img_write.astype(np.uint8), img_write.astype(np.uint8), 30, 0.05*16383, -1)
# yuv_filimg = cv2.cvtColor(filter_rgb, cv2.COLOR_RGB2YUV_I420)
# yuv_filimg[..., 0] = yuv_ori[..., 0]
# # yuv_filimg[..., 0] = (yuv_ori[..., 0] - yuv_ori[..., 0].min()) / (yuv_ori[..., 0].max() - yuv_ori[..., 0].min())
# rbg_filimg = cv2.cvtColor(yuv_filimg, cv2.COLOR_YUV2RGB_I420)
# rbg_filimg = cv2.cvtColor(rbg_filimg, cv2.COLOR_RGB2BGR)
# # rbg_filimg = (rbg_filimg - rbg_filimg.min()) / (rbg_filimg.max() - rbg_filimg.min()) * 255
# # rbg_filimg = np.clip(rbg_filimg, 0, 255)
# # rbg_filimg = cv2.cvtColor(rbg_filimg, cv2.COLOR_RGB2BGR)

cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/downup_framev2_' + str(0) + '.png', (img_write))


# for i in range(len(img_match_array)):
#     # raw_img = input_seq[i]
#     raw.raw_image_visible[740:2020, 2096:4256] = img_match_array[i]
#     post_raw = raw.postprocess(**rawpyParam)
#     cfa = (post_raw[740:2020, 2096:4256] / 65535. * 255.).astype(np.float32)  # scale and set type for cv2
#     hsv = cv2.cvtColor(cfa, cv2.COLOR_RGB2HSV)
#     hsvOperator = AutoGammaCorrection(hsv)
#     enhanceV = hsvOperator.execute()
#     hsv[..., -1] = enhanceV * 255.
#     enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#     enhanceRGB = cv2.cvtColor(enhanceRGB, cv2.COLOR_RGB2BGR)
#     img_write = cv2.normalize(((enhanceRGB)**1/2.2), dst=None, alpha=0, beta=255,
#                   norm_type=cv2.NORM_MINMAX).astype(np.uint8)
#     cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/downup_frame_' + str(i) + '.png', (img_write))