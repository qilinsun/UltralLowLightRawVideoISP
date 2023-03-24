import numpy as np
import torch
import torch.nn.functional as F
import os
import rawpy
from glob import glob
import cv2
from numba_PM_v2 import NNS, reconstruction
import matplotlib.pyplot as plt
from bm4d import bm4d, BM4DProfile
from experiment_funcs import generate_noise, get_experiment_kernel, get_psnr, get_cropped_psnr_3d
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, estimate_sigma, denoise_nl_means, denoise_tv_bregman
from einops import rearrange, repeat


raw_path = '/media/cuhksz-aci-03/数据/DRV_noprocess/short/'
seq_id = "0094"
star_id = 0
num_load = 16
p_size = 3
itr = 1
img_match = []


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
    im = im[1124:1636, 2920:3432]
    # im = im[740:2020, 2096:4256]

    return im

rawpyParam = {
            'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
            'half_size': False,
            'use_camera_wb' : True,
            'use_auto_wb' : False,
            'no_auto_bright': True,
            'output_color': rawpy.ColorSpace.sRGB,
            'output_bps' : 16}

def load_video_seq(folder_name, seqID, start_ind, num_to_load):
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
    full_im = np.empty((num_to_load, 1080, 1920))
    for i in range(0, num_to_load):
        # raw_img = rawpy.imread(filepaths_all_sorted[start_ind + i])
        # full_im[i] = pack_raw(raw_img)
        # CRVD
        tiff_img = cv2.imread(filepaths_all_sorted[start_ind +i], -1)
        full_im[i] = pack_gbrg_raw(tiff_img)

    return full_im

input_seq = load_video_seq(raw_path, seq_id, star_id, num_load)


# 预去噪
for num in range(len(input_seq)):
    # noise_variance = estimate_sigma(np.sqrt(input_seq[num, ...] + (3 / 8)))
    mergedImage_exp = np.expand_dims(np.sqrt(input_seq[num, ...]+(3/8)), axis=-1)
    merge_h, merge_w = input_seq[num].shape
    mergedImage_exp = mergedImage_exp
    merge_rggb = np.concatenate((mergedImage_exp[0:merge_h:2, 0:merge_w:2, :],  # R
                                 mergedImage_exp[0:merge_h:2, 1:merge_w:2, :],  # G
                                 mergedImage_exp[1:merge_h:2, 1:merge_w:2, :],  # B
                                 mergedImage_exp[1:merge_h:2, 0:merge_w:2, :]), axis=-1)  # G
    denoise_img = denoise_tv_chambolle((merge_rggb), weight=0.3, n_iter_max=200, multichannel=True)
    mergimg = np.empty([merge_h, merge_w])
    mergimg[::2, ::2] = denoise_img[..., 0]  # R
    mergimg[::2, 1::2] = denoise_img[..., 1]  # G
    mergimg[1::2, ::2] = denoise_img[..., 3]  # G
    mergimg[1::2, 1::2] = denoise_img[..., 2]  # B

    input_seq[num, ...] = mergimg**2-(3/8)

# 帧间匹配对齐
img_match.append(np.sqrt(input_seq[0]+(3/8)))
for i in range(input_seq.shape[0]-1):
    img_ori = input_seq[0]
    # img_down = image_interpolation(img_ori, (int(img_ori.shape[1]/8), int(img_ori.shape[0]/8)), inter_method=cv2.INTER_NEAREST)
    img = gamma_correction(img_ori)

    # img = vevid_simple(img_ori/16383, G=0.5, bais=0.6)
    # img = ((img - img.min()) / (img.max() - img.min())) * 16383.
    ref_ori = input_seq[i+1]
    # ref_down = image_interpolation(ref_ori, (int(ref_ori.shape[1]/8), int(ref_ori.shape[0]/8)), inter_method=cv2.INTER_NEAREST)
    ref = gamma_correction(ref_ori)
    # ref = vevid_simple(ref_ori / 16383, G=0.5, bais=0.6)
    # ref = ((ref - ref.min()) / (ref.max() - ref.min())) * 16383.
    f, dist, score_list = NNS(img, ref, p_size, itr)
    img_recon = reconstruction(f, img_ori, ref_ori)
    # img_recon = image_interpolation(img_recon, (int(img_ori.shape[1]), int(img_ori.shape[0])), inter_method=cv2.INTER_NEAREST)
    img_match.append(np.sqrt(img_recon+(3/8)))

img_match_array = np.array(img_match)
y = img_match_array.transpose(1, 2, 0)

for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'g1'
    noise_var = np.sqrt(0.002 * np.max((y[di::2, dj::2, 0])) ** 2) # Noise variance
    # noise_var = estimate_sigma((y[di::2, dj::2, 0]))
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
    profile.gamma = 6  # redefine value of gamma parameter
    y_est = bm4d((y[di::2, dj::2, :]), (noise_var), profile)
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

# 缓解块效应
mergedImage_exp = np.expand_dims((y[:, :, 0]), axis=-1)
merge_h, merge_w = y[:, :, 0].shape
mergedImage_exp = mergedImage_exp
merge_rggb = np.concatenate((mergedImage_exp[0:merge_h:2, 0:merge_w:2, :],  # R
                            mergedImage_exp[0:merge_h:2, 1:merge_w:2, :],  # G
                            mergedImage_exp[1:merge_h:2, 1:merge_w:2, :],  # B
                            mergedImage_exp[1:merge_h:2, 0:merge_w:2, :]), axis=-1)  # G
denoise_img = denoise_tv_bregman((merge_rggb), weight=70, max_iter=200, eps=0.001, multichannel=True)
mergimg = np.empty([merge_h, merge_w])
mergimg[::2, ::2] = denoise_img[..., 0]  # R
mergimg[::2, 1::2] = denoise_img[..., 1]  # G
mergimg[1::2, ::2] = denoise_img[..., 3]  # G
mergimg[1::2, 1::2] = denoise_img[..., 2]  # B

y[:, :, 0] = mergimg

raw = '/media/cuhksz-aci-03/数据/DRV_noprocess/short/0094/0001.ARW'
raw = rawpy.imread(raw)

raw.raw_image_visible[1124:1636, 2920:3432] = y[:, :, 0]**2-(3/8)
post_raw = raw.postprocess(**rawpyParam)
cfa = (post_raw[1124:1636, 2920:3432] / 65535. * 255.).astype(np.float32)  # scale and set type for cv2

# To HSV and local tone mapping
hsv = cv2.cvtColor(cfa, cv2.COLOR_RGB2HSV)
hsvOperator = AutoGammaCorrection(hsv)
enhanceV = hsvOperator.execute()
hsv[..., -1] = enhanceV * 255.
enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
img_write = cv2.normalize(((enhanceRGB)), dst=None, alpha=0, beta=255,
              norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# 去彩噪
yuv_ori = cv2.cvtColor(img_write.astype(np.uint8), cv2.COLOR_RGB2YUV_I420)
filter_rgb = cv2.ximgproc.guidedFilter(img_write.astype(np.uint8), img_write.astype(np.uint8), 30, 0.05*16383, -1)
yuv_filimg = cv2.cvtColor(filter_rgb, cv2.COLOR_RGB2YUV_I420)
yuv_filimg[..., 0] = yuv_ori[..., 0]
rbg_filimg = cv2.cvtColor(yuv_filimg, cv2.COLOR_YUV2RGB_I420)
rbg_filimg = cv2.cvtColor(rbg_filimg, cv2.COLOR_RGB2BGR)


cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/0.1_6400_' + str(0) + '.png', (img_write))
