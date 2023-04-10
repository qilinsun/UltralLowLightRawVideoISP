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
    positions = np.zeros((ri_rj_N__ni_nj.shape[0], ri_rj_N__ni_nj.shape[1], 12, 2))
    new_group = np.zeros((12, 8, 8))
    for i_r in row_ind:
        for j_r in column_ind:
            nSx_r = threshold_count[i_r, j_r]
            group_3D = build_3D_group(fre_all_patches, ri_rj_N__ni_nj[i_r, j_r], nSx_r)
            # reverse dct
            re_group_3d = bior_2d_reverse(group_3D.transpose((2, 0, 1)))
            num, ph, pw = re_group_3d.shape
            mask_0 = np.ones((ph, pw, 2))
            mask_0[int(ph / 2) - 2:int(ph / 2) + 2, int(pw / 2) - 2:int(pw / 2) + 2] = 0

            mask_1 = np.zeros((ph, pw, 2))
            mask_1[int(ph / 2) - 2:int(ph / 2) + 2, int(pw / 2) - 2:int(pw / 2) + 2] = 1
            
            # 计算出所有相似图块的低频分量，去除低频分量最高(高频分量最少)的几个图块，其余图块做硬阈值处理及后续的聚合
            cA_list = []
            for i in range(num):
                cA, (cH, cV, cD) = pywt.dwt2(re_group_3d[i], 'haar')
                cA_mean = np.mean(cA)
                cA_list.append(cA_mean)
            sort_ind = np.argsort(cA_list)
            nSx_r = nSx_r - int(num / 4)
         
            for j in range(nSx_r):
               
                new_group[j, :, :] = re_group_3d[sort_ind[j], :, :]
                positions[i_r, j_r, j, :] = ri_rj_N__ni_nj[i_r, j_r, sort_ind[j], :]

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
