import numpy as np
import os
import rawpy
from glob import glob
import cv2
import flowpy

import matplotlib.pyplot as plt
import deepmatching as dm
from DeepFlow.deepflow2 import deepflow2
from cv2.xfeatures2d import matchGMS
from genericUtils import getSigned, isTypeInt
from numba import vectorize, guvectorize, uint8, uint16, float32, float64
import signal

raw_path = '/media/cuhksz-aci-03/数据/CUHK_SZ/motion/0.01/'
seq_id = "iso51200"
star_id = 0
num_load = 16

@vectorize([uint8(float32), uint8(float64)], target='parallel')
def convert8bit_(x):
    return 0 if x <= 0 else (255 if x >= 1 else uint8(x * 255 + 0.5))

@vectorize([uint8(float32), uint8(float64)], target='parallel')
def convert16bit_(x):
    return 0 if x <= 0 else ((2**16 - 1) if x >= 1 else uint16(x * (2**16 - 1) + 0.5))

@vectorize([uint16(uint16, uint16, uint16, uint16)], target='parallel')
def umean4_(a, b, c, d):
    return np.right_shift(a + b + c + d + 2, 2) #????????????????


@vectorize([float32(float32, float32, float32, float32), float64(float64, float64, float64, float64)], target='parallel')
def fmean4_(a, b, c, d):
    return (a + b + c + d) * 0.25


def downsample(image, kernel='gaussian', factor=2):
    '''Apply a convolution by a kernel if required, then downsample an image.
    Args:
        image: the input image (WARNING: single channel only!)
        kernel: None / str ('gaussian' / 'bayer') / 2d numpy array
        factor: downsampling factor
    '''
    # Special case
    if factor == 1:
        return image

    # Filter the image before downsampling it
    if kernel is None:
        filteredImage = image
    # elif kernel == 'gaussian':
    #     # gaussian kernel std is proportional to downsampling factor
    #     filteredImage = gaussian_filter(image, sigma=factor * 0.5, order=0, output=None, mode='reflect')
    #     #*print('>>> image.shape  ',image.shape)
    #     #*print('>>> filteredImage.shape  ',filteredImage.shape)
    #     #*plt.subplot(1,2,1);
    #     #*plt.title('image')
    #     #*plt.imshow(image,cmap='gray',vmin=780,vmax=850)
    #     #*plt.subplot(1,2,2);
    #     #*plt.title('filteredImage')
    #     #*plt.imshow(filteredImage,cmap='gray',vmin=780,vmax=850)
    #     #*plt.show()
    elif kernel == 'bayer':
        # Bayer means that a simple 2x2 aggregation is required
        if isTypeInt(image):
            # starlight
            return umean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
            # # sony
            # return umean4_(image[2::4, 0::4], image[4::4, 0::4], image[2::4, 2::4], image[4::4, 2::4])
        else:
            # starlight
            return fmean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
            # # sony
            # return umean4_(image[2::4, 0::4], image[4::4, 0::4], image[2::4, 2::4], image[4::4, 2::4])
    else:
        # filter by convoluting with the input kernel
        filteredImage = signal.convolve2d(image, kernel, boundary='symm', mode='valid')

    # Shape of the downsampled image
    h2, w2 = np.floor(np.array(filteredImage.shape) / float(factor)).astype(np.int)

    # Extract the pixels
    if isTypeInt(image):
        return np.rint(filteredImage[:h2 * factor:factor, :w2 * factor:factor]).astype(image.dtype)
    else:
        return filteredImage[:h2 * factor:factor, :w2 * factor:factor]

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


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
    im = im[720:2000, 724:2884]

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
    # full_im = np.empty((num_to_load, 2848, 4256))
    for i in range(0, num_to_load):
        raw_img = rawpy.imread(filepaths_all_sorted[start_ind + i])
        full_im[i] = pack_raw(raw_img)

    return full_im

input_seq = load_video_seq(raw_path, seq_id, star_id, num_load)

#计算光流对齐
shifts = np.zeros((len(input_seq)-1, 2), dtype=np.float32)
image_align = []
for i in range(input_seq.shape[0]-1):
    img_ori = downsample(input_seq[i], kernel='bayer')
    ref_ori = downsample(input_seq[i+1], kernel='bayer')

    dis = cv2.optflow.createOptFlow_DeepFlow()
    flow = dis.calc(img_ori, ref_ori, None)
    # flow_map = flow_to_image(flow)
    # plt.imshow(flow_map)
    # plt.show()
    img_align = flowpy.backward_warp(ref_ori/16383*255, flow/16383*255)
    # plt.imshow(img_align, 'gray')
    # plt.show()
    # 仿射变换矩阵的计算
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = np.identity(2)
    M[:, 2] = np.mean(flow.reshape(-1, 2), axis=0)
    # 计算位移
    shifts[i-1] = M[:, 2]

        
    print('------The match stage is end!-------')
    
# 计算中间帧相对于第一帧的位移
mid_shift = np.sum(shifts, axis=0)/len(shifts)
mid_frame = len(input_seq) // 2

# 计算每一帧相对于中间帧的位移
frame_shifts = np.zeros((len(input_seq), 2), dtype=np.float32)
for i in range(len(input_seq)):
    frame_shifts[i] = mid_shift if i == mid_frame else shifts[i-1] - mid_shift

# 对齐
aligned_imgs = []
for i in range(len(input_seq)):
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = np.identity(2)
    M[:, 2] = frame_shifts[i]

    aligned_img = cv2.warpAffine(input_seq[i], M, (input_seq[i].shape[1], input_seq[i].shape[0]))
    aligned_imgs.append(aligned_img)

aligned_imgs = np.array(aligned_imgs)
