from .imageUtils import getTiles
import numpy as np
from einops import repeat
import os
import cv2
from .imageUtils import *


def motion_angle(motionvetor):
    motionvetor_y = motionvetor[..., 0]
    motionvetor_x = motionvetor[..., 1]

    # 使用反正切计算弧度值
    angle = np.arctan2(motionvetor_y, motionvetor_x)
    # 将弧度转化为度
    # angle = np.rad2deg(cos_theta)

    return angle

def motion_vector_length(motionvetor):
    n, h, w, _ = motionvetor.shape
    length = np.zeros((n, h, w))
    angle = np.zeros((n, h, w))
    for i in range(len(motionvetor)):
        if i == 0:
            mv = motionvetor[i]
            mv_length = np.sqrt(mv[...,0]**2 + mv[...,1]**2)
            length[i, :, :] = mv_length
            mv_angle = motion_angle(mv)
            angle[i, :, :] = mv_angle
        else:
            cur_mv = motionvetor[i]
            ref_mv = motionvetor[i-1]
            diff_mv = cur_mv - ref_mv
            diff_mv_length = np.sqrt(diff_mv[...,0]**2 + diff_mv[...,1]**2)
            diff_mv_angle = motion_angle(diff_mv)
            length[i, :, :] = diff_mv_length
            angle[i, :, :] = diff_mv_angle

    return length, angle

def get_motion_blur(length, angle, aligntiles):
    # 点扩散函数
    n, h, w, size1, size2 = aligntiles.shape
    PSF_sum = np.zeros((aligntiles.shape[0]-1, aligntiles.shape[1], aligntiles.shape[2], aligntiles.shape[3],aligntiles.shape[4]))
    PSF_aver = np.zeros((int(h * size1/2+size1/2), int(w * size2/2+size1/2)))
    PSF_vis = np.zeros((n-1, int(h * size1/2+size1/2), int(w * size2/2+size1/2)))
    for i in range(aligntiles.shape[0]-1):

        x_center = (aligntiles[i].shape[2] - 1) / 2
        y_center = (aligntiles[i].shape[3] - 1) / 2

        motion_length = length[i]
        motion_angle = angle[i]

        sin_val = np.sin(motion_angle)
        cos_val = np.cos(motion_angle)

        # 计算每个tiles的psf 
        for j in range(motion_length.shape[0]):
            for n in range((motion_length.shape[1])):
                PSF = np.zeros((aligntiles.shape[3], aligntiles.shape[4])).astype(np.uint8)
                if motion_length[j][n] == 0:
                    PSF_sum[i, j, n, ...] = PSF
                else:
                    for m in range(int(np.round(motion_length[j][n]))):
                        x_offset = np.round(sin_val[j, n] * m)
                        y_offset = np.round(cos_val[j, n] * m)
                        x_1 = x_center - x_offset
                        y_1 = y_center + y_offset
                        if 0 <= x_1 < (aligntiles.shape[3])  and 0 <= y_1 < (aligntiles.shape[4]):
                            x = x_1
                            y = y_1
                        else:
                            x = x_center
                            y = y_center

                        PSF[int(x), int(y)] = 1
          
                    PSF = PSF / PSF.sum()
                    PSF_sum[i, j, n, ...] = PSF
                
                    PSF_aver[(int(size1/2) * j):(int(size1/2) * j + size1), (int(size2/2)* n):(int(size2/2) * n + size2)] = PSF

        PSF_vis[i, ...] = PSF_aver

    for v in range(PSF_vis.shape[0]):
        psf_path = "/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/psf_result/" + str(v) + '.png'
        cv2.imwrite(psf_path, PSF_vis[v])

    return PSF_sum
