import numpy as np
import os
from utils import downsample, patchesRMS
import cv2
from einops import rearrange,repeat
from scipy import signal

def Sobel(img,filter1,filter2):
    h,w=img.shape[:2]
    new_img=np.zeros((h+2,w+2),np.float32)
    new_img[1:h+1,1:w+1]=img#填充
    out=[]
    for i in range(1,h+1):
        for j in range(1,w+1):
            dx=np.sum(np.multiply(new_img[i-1:i+2,j-1:j+2],filter1))
            dy=np.sum(np.multiply(new_img[i-1:i+2,j-1:j+2],filter2))
            out.append(
                np.clip(int(np.sqrt(dx**2+dy**2)),0,1)
            )
    out=np.array(out).reshape(h,w)
    return out


def image_interpolation(img,new_dimension,inter_method):
    inter_img = cv2.resize(img,new_dimension,interpolation=inter_method)
    return inter_img

def esti_motion(raw_img1, raw_img2, K):
    filter1 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    filter2 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]]) #*.25

    # normalize
    raw_img1 = raw_img1 / 16383.
    raw_img2 = raw_img2 / 16383.
    # downsample
    down_raw1 = downsample(raw_img1, kernel='bayer')
    down_raw2 = downsample(raw_img2, kernel='bayer')

    # motion estimation
    mode = 'same'
    # fx = signal.convolve2d(down_raw1, kernel_x, boundary='symm', mode=mode)
    # fy = signal.convolve2d(down_raw1, kernel_y, boundary='symm', mode=mode)
    gradient_raw1 = signal.convolve2d(down_raw1, kernel_t, boundary='symm', mode=mode)
    gradient_intensity = signal.convolve2d(down_raw2, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(down_raw1, -kernel_t,
                                                                                          boundary='symm', mode=mode)

    # gradient_raw1 = Sobel(down_raw1, filter1, filter2)
    # gradient_intensity = Sobel(intensity, filter1, filter2)
    intensgradient_raw1_norm = np.abs(gradient_raw1)
    RMS_patch = down_raw1.reshape(64, 64, 16, 16)
    RMS = patchesRMS(RMS_patch)
    RMS = repeat(RMS, 'h w -> h w c d', c=1, d=1)
    noiseVariance = repeat(RMS, 'h w 1 1-> h w c d', c=16, d=16)
    RMS_reshape = noiseVariance.reshape(down_raw1.shape)
    threshold = K * RMS_reshape
    for x in range(intensgradient_raw1_norm.shape[0]):
        for y in range(intensgradient_raw1_norm.shape[1]):
            if intensgradient_raw1_norm[x, y] < threshold[x, y]:
                # intensgradient_raw1 = np.delete(gradient_raw1, x, y)
                intensgradient_raw1_norm[x, y] = 1e-6

    # 运动细化
    motion = np.abs(gradient_intensity) / (intensgradient_raw1_norm)
    motion_downsam = image_interpolation(motion, (512, 512), inter_method=cv2.INTER_NEAREST)
    motion_patch = motion_downsam.reshape(64, 64, 8, 8)
    for i in range(motion_patch.shape[2]):
        for j in range(motion_patch.shape[3]):
            num = motion_patch.shape[0] * motion_patch.shape[1]
            pres_num = num - int(num * 0.9)
            motion_bin = (motion_patch[:,:,i,j]).reshape(1, -1)

            ind = (np.argpartition(motion_bin, -pres_num)[:, -pres_num:])

            for indx in range(ind.shape[1]):
                motion_bin[:, ind[:, indx]] = 0

            motion_patch[:,:,i,j] = motion_bin.reshape(64, 64)

    motion_refine = motion_patch.reshape(512, 512)

    return motion_refine

