import numpy as np
import cv2

from utils_1 import ind_initialize, get_kaiserWindow, sd_weighting
from precompute_BM import precompute_BM
from bior_2d import bior_2d_forward, bior_2d_reverse
from dct_2d import dct_2d_forward, dct_2d_reverse
from image_to_patches import image2patches
from build_3D_group import build_3D_group
from ht_filtering_hadamard import ht_filtering_hadamard
import pywt

def filter_low(fshift, block, radius_ratio):
    filter_img = np.ones(fshift.shape)
    crow, col = int(fshift.shape[0]/2), int(fshift.shape[1]/2)
    radius = int(radius_ratio*block.shape[0])

    if len(block.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0,0,0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)

    return filter_img*block

def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg

def bm3d_1st_step(sigma, img_noisy, nHard, kHard, NHard, pHard, lambdaHard3D, tauMatch, useSD, tau_2D):
    height, width = img_noisy.shape[0], img_noisy.shape[1]

    row_ind = ind_initialize(height - kHard + 1, nHard, pHard)
    column_ind = ind_initialize(width - kHard + 1, nHard, pHard)

    kaiserWindow = get_kaiserWindow(kHard)
    ri_rj_N__ni_nj, threshold_count = precompute_BM(img_noisy, kHW=kHard, NHW=NHard, nHW=nHard, tauMatch=tauMatch)
    group_len = int(np.sum(threshold_count))
    group_3D_table = np.zeros((group_len, kHard, kHard))
    weight_table = np.zeros((height, width))

    all_patches = image2patches(img_noisy, kHard, kHard)  # i_j_ipatch_jpatch__v
    if tau_2D == 'DCT':
        fre_all_patches = dct_2d_forward(all_patches)
    else:  # 'BIOR'
        fre_all_patches = bior_2d_forward(all_patches)

    acc_pointer = 0
    positions = np.zeros((ri_rj_N__ni_nj.shape[0], ri_rj_N__ni_nj.shape[1], 16, 2))
    new_group = np.zeros((16, 8, 8))
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            group_3D = build_3D_group(fre_all_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            # reverse dct
            re_group_3d = bior_2d_reverse(group_3D.transpose((2, 0, 1)))
            num, ph, pw = re_group_3d.shape
	    # 去除块处理
            cA_list = []
            radius_ratio = 0.6
            for i in range(num):
                f = np.fft.fftn(re_group_3d[i])
                # 计算高频分量
                fshift = np.fft.fftshift(f)
                hight_parts_fshift = filter_low(fshift.copy(), re_group_3d[i], radius_ratio=radius_ratio)
                # high_parts_img = ifft(hight_parts_fshift)
                # img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                #         np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)
                # cA_mean = np.mean(img_new_high * 16383)
                cA_mean = np.mean(hight_parts_fshift)

                # cA, (cH, cV, cD) = pywt.dwt2(re_group_3d[i], 'haar')
                # cA_mean = np.mean(cA)
                cA_list.append(cA_mean)
            sort_ind = np.argsort(-np.array(cA_list))
            nSx_r = nSx_r - int(num / 5)
		
            for j in range(nSx_r):
                # re_group_3d.pop(sort_ind[-1])
                # position.pop(sort_ind[-1])
                new_group[j, :, :] = re_group_3d[sort_ind[j], :, :]
                p = ri_rj_N__ni_nj[i_r, j_r, sort_ind[j], :]
                positions[i_r, j_r, j, :] = p
		
            group_3D = bior_2d_forward(new_group)
            group_3D, weight = ht_filtering_hadamard(group_3D, sigma, lambdaHard3D, not useSD)
            # group_3D = group_3D.transpose((2, 0, 1))
            group_3D_table[acc_pointer:acc_pointer + nSx_r] = group_3D
            acc_pointer += nSx_r

            if useSD:
                weight = sd_weighting(group_3D)

            weight_table[i_r, j_r] = weight

    if tau_2D == 'DCT':
        group_3D_table = dct_2d_reverse(group_3D_table)
    else:  # 'BIOR'
        group_3D_table = bior_2d_reverse(group_3D_table)

    # group_3D_table = np.maximum(group_3D_table, 0)
    # for i in range(1000):
    #     patch = group_3D_table[i]
    #     print(i, '----------------------------')
    #     print(patch)
    #     print(np.min(patch))
    #     print(np.max(patch))
    #     print(np.sum(patch))
    #     cv2.imshow('', patch.astype(np.uint8))
    #     cv2.waitKey()

    # aggregation part
    numerator = np.zeros_like(img_noisy, dtype=np.float64)
    denominator = np.zeros((img_noisy.shape[0] - 2 * nHard, img_noisy.shape[1] - 2 * nHard), dtype=np.float64)
    denominator = np.pad(denominator, nHard, 'constant', constant_values=1.)
    acc_pointer = 0

    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]-4
            # nSx_r = threshold_count[i_r, j_r]
            # N_ni_nj = ri_rj_N__ni_nj[i_r, j_r]
            N_ni_nj = positions[i_r, j_r]
            group_3D = group_3D_table[acc_pointer:acc_pointer + nSx_r]
            acc_pointer += nSx_r
            weight = weight_table[i_r, j_r]
            for n in range(nSx_r):
                ni, nj = N_ni_nj[n]
                patch = group_3D[n]

                numerator[int(ni):int(ni) + kHard, int(nj):int(nj) + kHard] += patch * kaiserWindow * weight
                denominator[int(ni):int(ni) + kHard, int(nj):int(nj) + kHard] += kaiserWindow * weight

    img_basic = numerator / (denominator+1e-6)
    return img_basic


if __name__ == '__main__':
    from utils import add_gaussian_noise, symetrize

    # <hyper parameter> -------------------------------------------------------------------------------
    sigma = 20

    nHard = 16
    kHard = 8
    NHard = 16
    pHard = 3
    lambdaHard3D = 2.7  # ! Threshold for Hard Thresholding
    tauMatchHard = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
    useSD_h = False
    tau_2D_hard = 'BIOR'
    # <\ hyper parameter> -----------------------------------------------------------------------------

    img = cv2.imread('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/20230401/motion_0.01_51200/bm3d_bm4d.png', cv2.IMREAD_GRAYSCALE)
    img_noisy = add_gaussian_noise(img, sigma)
    # img_noisy = cv2.imread('matlab_officialfg_compare/noisy_image.png', cv2.IMREAD_GRAYSCALE)

    img_noisy_p = symetrize(img_noisy, nHard)
    img_basic = bm3d_1st_step(sigma, img_noisy_p, nHard, kHard, NHard, pHard, lambdaHard3D, tauMatchHard, useSD_h,
                              tau_2D_hard)
    img_basic = img_basic[nHard: -nHard, nHard: -nHard]

    cv2.imwrite('y_basic.png', img_basic.astype(np.uint8))
