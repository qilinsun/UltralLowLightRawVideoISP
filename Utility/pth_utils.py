import os
import sys
import cv2
import time
import json
import rawpy
import scipy.io
import exifread 
import numpy as np 
from glob import glob
from sklearn.exceptions import DataDimensionalityWarning
from vidgear.gears import WriteGear 
from matplotlib import pyplot as plt
from natsort import natsorted as sort

from Utility.pyhdr import alignHdrplus
from Utility.merging import mergeHdrplus
from Utility.visUtils import addMotionField

from Utility.utils import ARPS_search_frame,MotionVectorClass, readTIFF,normalize,segmentImage,segmentImageHalfOverlap,getSAD,pthSAD, \
                            getSpatialL1Front,getDCTL1Front,calculateWeightFront,getDiffSpatialL1Front, \
                            readARW,TiffFastISP,ARWFastISP,ARWRawpyISP,writerVideo,getTime,weightedSAD
import torch
from torch.utils.data import Dataset

from einops import rearrange, reduce, repeat
from Utility.PSF import *
from Utility.wiener import inverse_filter


verbose = 0

def greenPixelReplace(raw_video,whiteLevel):
    """ set last channel(weakest) is NIR 
    green pixel replacement process: obtain new  grren channel for RGGB ISP
    Args:
        raw_video (shape : [videoNum,h,w,4]): raw video
        output (shape : [videoNum,h,w,4]): 
    """
    # get g channel by interpolating NIR channel [cfaH,cfaW,3] -> [cfaH,cfaW,4]
    NIRChannel = raw_video[:,:,:,-1:] # [seq_num,h,w,1]
    # NIR interpolation [n,cfaH,cfaW,1] -> [n,2cfaH,2cfaW,1]
    itplNIRChannel = bayer_bilinear(NIRChannel) # [seq_num,2h,2w,1]
    # get NIR subsample
    subNIRChannel = itplNIRChannel[:,0::2,1::2,0]
    # K = G - NIR subsample
    kChannel = raw_video[:,:,:,1] - subNIRChannel
    # G_2 = K + NIR
    g2Channel = kChannel + NIRChannel[...,0]
    # mean shift
    g2Channel = g2Channel - (np.mean(g2Channel) - np.mean(raw_video[:,:,:,1]))
    # clipping
    raw_video[:,:,:,-1] = np.clip(g2Channel,0,whiteLevel) 
    return raw_video

def bayer_bilinear(imagei, height = None, width= None):
    
    if len(imagei.shape) == 4:
        frames, img_height, img_width, channels = imagei.shape
    elif len(imagei.shape) == 3:
        img_height, img_width, channels = imagei.shape
        frames = 1
        imagei = imagei[np.newaxis]
    else:
        print('invalid image size')
        
    if height is None or width is None:
        height = img_height*2
        width = img_width*2

    new_image = np.empty((frames, height, width, channels))
    for k in range(0,frames):
        for i in range(0,channels):
            # channel-wise interpolation
            image = imagei[k,...,i].ravel()
            # print('>>> image.shape  ',image.shape) (691200,)
            x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
            y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0
            # print('>>> x_ratio  ',x_ratio)0.4997684113015285
            # print('>>> y_ratio  ',y_ratio)0.49960906958561374

            # 0 到 height的loop list，因为divmod取余数
            y, x = np.divmod(np.arange(height * width), width)
            # print('>>> y  ',y)  [   0    0    0 ... 1279 1279 1279]
            # print('>>> x  ',x)  [   0    1    2 ... 2157 2158 2159]
            
            #退一法 low
            x_l = np.floor(x_ratio * x).astype('int32') # 2159 * 0.49976841130285 = 1079
            y_l = np.floor(y_ratio * y).astype('int32') # 1279 * 0.49960906958561374 = 639
            # print('>>> x_l.shape  ',x_l.shape)(2764800,)
            # print('>>> y_l.shape  ',y_l.shape)(2764800,)
            #进一法 high
            x_h = np.ceil(x_ratio * x).astype('int32')
            y_h = np.ceil(y_ratio * y).astype('int32')
            # print('>>> x_h.shape  ',x_h.shape)(2764800,)
            # print('>>> y_h.shape  ',y_h.shape)(2764800,)

            x_weight = (x_ratio * x) - x_l
            y_weight = (y_ratio * y) - y_l
            # print('>>> x_weight.shape  ',x_weight.shape)(2764800,)
            # print('>>> y_weight.shape  ',y_weight.shape)(2764800,)
            
            # x_l_test = x_l.reshape(height,width)
            # y_l_test = y_l.reshape(height,width)
            # x_h_test = x_h.reshape(height,width)
            # y_h_test = y_h.reshape(height,width)
            # x_weight_test = x_weight.reshape(height,width)
            # y_weight_test = y_weight.reshape(height,width)
            # plt.subplot(3,4,1);
            # plt.title('x_l_test')
            # plt.imshow(x_l_test,cmap='gray')
            # plt.subplot(3,4,2);
            # plt.title('y_l_test')
            # plt.imshow(y_l_test,cmap='gray')
            # plt.subplot(3,4,3);
            # plt.title('x_h_test')
            # plt.imshow(x_h_test,cmap='gray')
            # plt.subplot(3,4,4);
            # plt.title('y_h_test')
            # plt.imshow(y_h_test,cmap='gray')
            # plt.subplot(3,4,5);
            # plt.title('x_weight_test')
            # plt.imshow(x_weight_test,cmap='gray')
            # plt.subplot(3,4,6);
            # plt.title('y_weight_test')
            # plt.imshow(y_weight_test,cmap='gray')
            # plt.subplot(3,4,7);
            # plt.title('x_h_test - x_l_test')
            # plt.plot(np.arange(len(x_l.ravel())),x_l.ravel())
            # plt.plot(np.arange(len(x_h.ravel())),x_h.ravel())
            # plt.imshow(x_h_test - x_l_test,cmap='gray')
            # plt.subplot(3,4,8);
            # plt.title('x_l_test - x_h_test')
            # plt.imshow(x_l_test - x_h_test,cmap='gray')
            # print('>>> x_h_test  ',x_h_test)
            # print('>>> x_l_test  ',x_l_test)
            # print('>>> x_h_test-x_l_test  ',x_h_test-x_l_test)
            # assert 1==2
            a = image[y_l * img_width + x_l]
            b = image[y_l * img_width + x_h]
            c = image[y_h * img_width + x_l]
            d = image[y_h * img_width + x_h]
            # a_test = a.reshape(height,width)
            # b_test = b.reshape(height,width)
            # c_test = c.reshape(height,width)
            # d_test = d.reshape(height,width)
            # print('>>> a.shape,b.shape,c.shape,d.shape  ',a.shape,b.shape,c.shape,d.shape)(2764800,)(2764800,)(2764800,)(2764800,)
            # plt.subplot(3,4,9);
            # plt.title('a_test')
            # plt.imshow(a_test,cmap='gray')
            # plt.subplot(3,4,10);
            # plt.title('b_test')
            # plt.imshow(b_test,cmap='gray')
            # plt.subplot(3,4,11);
            # plt.title('c_test')
            # plt.imshow(c_test,cmap='gray')
            # plt.subplot(3,4,12);
            # plt.title('d_test')
            # plt.imshow(d_test,cmap='gray')
            # plt.show()
            
            resized = a * (1 - x_weight) * (1 - y_weight) + \
                    b * x_weight * (1 - y_weight) + \
                    c * y_weight * (1 - x_weight) + \
                    d * x_weight * y_weight
            new_image[k,...,i] = resized.reshape(height, width)
            # print('>>> new_image[k,...,i].shape',new_image[k,...,i].shape)(1280, 2160)
            
    if frames == 1:
        new_image = new_image[0]
    return new_image

def write_mask(mask,points,rectangle_size=20): 
    half_rectangle_size = rectangle_size // 2 
    # input point 
    y,x = mask.shape 
    point_y, point_x = points
    
    # from 
    # for h in range(point_y-half_rectangle_size,point_y+half_rectangle_size): 
    #     for w in range(point_x-half_rectangle_size,point_x+half_rectangle_size): 
    #         if h<0 or h>=y: 
    #             continue 
    #         if w<0 or w>=x: 
    #             continue 
    #         mask[int(h),int(w)] = 0. 
    
    for h in range(point_y-half_rectangle_size,point_y+half_rectangle_size): 
        if h<0 or h>=y: 
            continue 
        if point_x<0 or point_x>=x: 
            continue 
        mask[int(h),point_x] = 0.
    for w in range(point_x-half_rectangle_size,point_x+half_rectangle_size): 
        if w<0 or w>=x: 
            continue 
        if point_y<0 or point_y>=y: 
            continue 
        mask[point_y,int(w)] = 0.
    return mask 

def fill_rectange_mask(h,w,num_patch_h,num_patch_w,rectangle_size): 
    mask = np.ones([h,w]) 
    for idx_y,point_y in enumerate(np.linspace(0,h,num_patch_h)): 
        for point_x in np.linspace(0,w,num_patch_w): 
            # pass center with 1
            if point_y == h//2 and point_x== w//2:
                continue 
                # make center has value
            mask = write_mask(mask,(point_y.astype(np.int),point_x.astype(np.int)),rectangle_size) 
        # horizontal line 
        # if idx_y == 1 or idx_y == 3:
        #     mask[int(point_y)-10:int(point_y)+10,:] = 0.
    
    # 中点的竖线
    # for i in range(h//2 - rectangle_size,h//2 - 40): 
    #     mask[i,w//2] = 0.
    # for i in range(h//2 + 40, h//2 + rectangle_size): 
    #     mask[i,w//2] = 0.
    # 中间的横线
    # for i in range(0,w//2-50): 
    #     mask[h//2,i] = 0.
    # for i in range(w//2+50,w): 
    #     mask[h//2,i] = 0.
    
    return mask.astype(np.float64) 

def WriteVideoGear(images,outputPath,filename,gamma=True,normalize=True,fps=10):
    """Usage:
    WriteVideoGear(self.raw_video,outputPath="/FPN/",filename="xx.mp4",gamma=True,normalize=True,fps=10) # no
    
    Args:
        images (any loopable array): stack of images
        gamma (bool, optional): Defaults to True.
        normalize (bool, optional): Defaults to True.
        fps (int, optional): Defaults to 10.
        mask_mode (bool, optional): Defaults to True.
    """
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    # MP4
    output_params = {"-input_framerate":fps}# "-r":30
    outputPath = outputPath + filename
    writer = WriteGear(output_filename=outputPath,compression_mode=True,logging=not True,**output_params) 
    
    for frame in images:
        if gamma:
            frame = frame**(1/2.2)    
        if normalize:
            frame = cv2.normalize(frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        writer.write(frame)
    writer.close() 

def cat2DPatches(patches):
    assert(len(patches.shape) == 4), "not a 2D array of 2D arrays"
    return np.concatenate(np.concatenate(patches, axis=1), axis=1)

def centeredCosineWindow(x, windowSize=16):
    '''1D version of the modified raised cosine window (Section 4.4 of the IPOL article).
    It is centered and nonzero at x=0 and x=windowSize-1'''
    y = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * (x + 1 / 2.) / windowSize)
    return y

def cosineWindow2Dpatches(patches):
    '''Apply a 2D version of the modified raised cosine window
    to a set of overlapped patches to avoid discontinuities and edge artifacts
    (Section 4.4 of the IPOL article).
    Args:
        patches: 2D array of 2D patches (overlapped by half in each dimension)'''
    assert(len(patches.shape) == 4), 'not a 2D array of image patches'
    windowSize = patches.shape[-1]  # Assumes patches are square
    # Compute the attenuation window on 1 patch dimension
    lineWeights = centeredCosineWindow(np.arange(windowSize), windowSize).reshape(-1, 1).repeat(windowSize, 1)
    columnWeights = lineWeights.T
    # the 2D window is the product of the 1D window in both patches dimensions
    window = np.multiply(lineWeights, columnWeights)
    # Apply the attenuation cosine weighting to all patches
    return np.multiply(patches, window)

def depatchifyOverlap(patches):
    '''recreates a single image out of a 2d arrangement
    of patches overlapped by half in each dimension
    '''
    assert(len(patches.shape) == 4), "not a 2D array of 2D patches"
    patchSize = patches.shape[-1]
    dp = patchSize // 2
    assert(patchSize == patches.shape[-2] and patchSize % 2 == 0), "function only supports square patches of even size"

    # separate the different groups of overlapped patches
    patchSet00 = patches[0::2, 0::2]  # original decomposition
    patchSet01 = patches[0::2, 1::2]  # straddled by patchSize/2 in horizontal axis
    patchSet10 = patches[1::2, 0::2]  # straddled by patchSize/2 in vertical axis
    patchSet11 = patches[1::2, 1::2]  # straddled by patchSize/2 half in both axes

    # recreate sub-images from the different patch groups
    imSet00 = cat2DPatches(patchSet00)
    imSet01 = cat2DPatches(patchSet01)
    imSet10 = cat2DPatches(patchSet10)
    imSet11 = cat2DPatches(patchSet11)

    # reconstruct final image by correctly adding sub-images
    reconstructedImage = np.zeros(((patches.shape[0] + 1) * dp, (patches.shape[1] + 1) * dp), dtype=imSet00.dtype)
    reconstructedImage[0 : imSet00.shape[0]     , 0 : imSet00.shape[1]     ]  = imSet00
    reconstructedImage[0 : imSet01.shape[0]     , dp: imSet01.shape[1] + dp] += imSet01
    reconstructedImage[dp: imSet10.shape[0] + dp, 0 : imSet10.shape[1]     ] += imSet10
    reconstructedImage[dp: imSet11.shape[0] + dp, dp: imSet11.shape[1] + dp] += imSet11
    return reconstructedImage

def getTiles(a, window, steps=None, axis=None):
    '''
    Create a windowed view over `n`-dimensional input that uses an
    `m`-dimensional window, with `m <= n`

    Parameters
    -------------
    a : Array-like
        The array to create the view on

    window : tuple or int
        If int, the size of the window in `axis`, or in all dimensions if
        `axis == None`

        If tuple, the shape of the desired window.  `window.size` must be:
            equal to `len(axis)` if `axis != None`, else
            equal to `len(a.shape)`, or
            1

    steps : tuple, int or None
        The offset between consecutive windows in desired dimension
        If None, offset is one in all dimensions
        If int, the offset for all windows over `axis`
        If tuple, the steps along each `axis`.
            `len(steps)` must me equal to `len(axis)`

    axis : tuple, int or None
        The axes over which to apply the window
        If None, apply over all dimensions
        if tuple or int, the dimensions over which to apply the window

    Returns
    -------

    a_view : ndarray
        A windowed view on the input array `a`, or a generator over the windows

    '''
    ashp = np.array(a.shape)

    if axis is not None:
        axs = np.array(axis, ndmin=1)
        assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin=1)
    assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), "Window is bigger than input array in axes"

    stp = np.ones_like(ashp)
    if steps:
        steps = np.array(steps, ndmin=1)
        assert np.all(steps > 0), "Only positive steps allowed"
        assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
        stp[axs] = steps

    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)

    return np.squeeze(np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides))

sys.path.append("Utility/fast-openISP")
from pipeline import Pipeline
from util.yacs import Config
def starlightFastISP(rawIn):
    """
    Usage:
        rgb = starFastISP(packing[0]) 
            input shape : (rawH,rawW)
    Keys of pipeline.execute output:
        (['bayer', 'rgb_image', 'y_image', 'cbcr_image', 'edge_map', 'output'])
    """
    cfg = Config('Utility/fast-openISP/configs/starlight.yaml')
    pipeline = Pipeline(cfg)
    
    rawIn = rawIn.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
    rawOut, _ = pipeline.execute(rawIn)
    
    return rawOut

def ycbcr2rgb(ycbcr_image):
    """convert ycbcr into rgb"""
    if len(ycbcr_image.shape)!=3 or ycbcr_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    ycbcr_image = ycbcr_image.astype(np.float32)
    transform_matrix = np.array([[0.257, 0.504, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    shift_matrix = np.array([16, 128, 128])
    rgb_image = np.zeros(shape=ycbcr_image.shape)
    w, h, _ = ycbcr_image.shape
    for i in range(w):
        for j in range(h):
            rgb_image[i, j, :] = np.dot(transform_matrix_inv, ycbcr_image[i, j, :]) - np.dot(transform_matrix_inv, shift_matrix)
    return rgb_image.astype(np.uint8)

def normalize(item): 
    vmin = np.min(item)
    vmax = np.max(item)
    assert vmin != vmax 
    return (item - vmin) / (vmax -vmin) 

class AutoGammaCorrection: # img in HSV Space 
    def __init__(self, img): 
        self.img = img 
    def execute(self): 
        img_h = self.img.shape[0] 
        img_w = self.img.shape[1] 
        Y = self.img[:,:,2].copy() 
        gc_img = np.zeros((img_h, img_w), np.float32) 
        Y = Y.astype(np.float32)/256.  # 255. is better right?
        Yavg = np.exp(np.log(Y+1e-20).mean()) 
        for y in range(self.img.shape[0]): 
            for x in range(self.img.shape[1]): 
                if Y[y, x]>0: 
                    gc_img[y, x] = np.log(Y[y,x]/Yavg + 1)/np.log(1/Yavg+1) 
                # if x==1500: 
                #     print(x, Y[y, x], gc_img[y, x]) 
        return gc_img 
#      _       _                 _   
#   __| | __ _| |_ __ _ ___  ___| |_ 
#  / _` |/ _` | __/ _` / __|/ _ \ __|
# | (_| | (_| | || (_| \__ \  __/ |_ 
#  \__,_|\__,_|\__\__,_|___/\___|\__|

class RawDataset(Dataset): #! ffcc accelarate 
    def __init__(self, burstPath, BUCKET_SIZE, seqID, whiteLevel):
        # initialization
        self.burst_path = burstPath
        self.bucket_size = BUCKET_SIZE
        self.seqID = seqID
        global verbose
        self.raw_video = self.load_video_seq(self.burst_path,seqID=seqID,start_ind=0,num_to_load=30) # num_to_load="all"  20
        self.num_samples = len(self.raw_video)
        
        # subtract FPN
        self.FPN = self.load_fpn()
        self.raw_video -= self.FPN # [seq_num,h,w,4]
        self.raw_video = np.clip(self.raw_video,0,whiteLevel) 
        
        # green CFA interpolation
        self.raw_video = greenPixelReplace(self.raw_video,whiteLevel)
        
        # making RGGB pattern
        videoNum,h,w,c = self.raw_video.shape
        self.bucket = np.empty([videoNum,h*2,w*2])
        self.bucket[:,::2,::2] = self.raw_video[...,0] #R
        self.bucket[:,::2,1::2] = self.raw_video[...,1] #G
        self.bucket[:,1::2,::2] = self.raw_video[...,3] #G
        self.bucket[:,1::2,1::2] = self.raw_video[...,2] #B

        # To torch
        # self.bucket = torch.from_numpy(self.bucket)
        
    def __len__(self):
        return self.num_samples - self.bucket_size + 1

    def __getitem__(self, idx): # idx 无限制
        return self.bucket[idx : idx + self.bucket_size]

    def load_video_seq(self,folder_name, seqID, start_ind, num_to_load):
        base_name_seq = folder_name + 'seq' + str(seqID) + '/'
        filepaths_all = glob(base_name_seq + '*.mat')
        total_num = len(filepaths_all)

        ind = []
        for i in range(0,len(filepaths_all)):
            ind.append(int(filepaths_all[i].split('/')[-1].split('.')[0]))
        ind = np.argsort(np.array(ind))
        filepaths_all_sorted = np.array(filepaths_all)[ind]
        
        if num_to_load == 'all':
            num_to_load = total_num
            print('loading ', num_to_load, 'frames')
        full_im = np.empty((num_to_load, 640, 1080, 4))
        for i in range(0,num_to_load):
            loaded = scipy.io.loadmat(filepaths_all_sorted[start_ind +i])
            full_im[i] = loaded['noisy_list'].astype('float32') # / 2**16
        if verbose:
            print('>>> filepaths_all_sorted  ',filepaths_all_sorted)
        return full_im

    def load_fpn(self,fpn_path="../starlight_denoising/data/fixed_pattern_noise.mat"):
        return scipy.io.loadmat(fpn_path)["mean_pattern"]

#                      _      _ 
#  _ __ ___   ___   __| | ___| |
# | '_ ` _ \ / _ \ / _` |/ _ \ |
# | | | | | | (_) | (_| |  __/ |
# |_| |_| |_|\___/ \__,_|\___|_|
                                                       
                                   
class LLR2V(torch.nn.Module):
    def __init__(self,blockSize,whiteLevel):
        super(LLR2V, self).__init__()
        self.mbSize = blockSize
        self.whiteLevel = whiteLevel
        self.count = 0
        
        self.padding = (0,0,0,0) # (paddingTop, paddingBottom, paddingLeft, paddingRight)
        print('Warning: The padding size are all zero! (Assuming image shape can be divided by the block size)')
        self.lambdaS = 25.239600000000003
        self.lambdaR = 26094.162999999997
        # keepAlternate pairedAverage DFTWiener
        self.params = {'patchSize': self.mbSize, 'method': 'DFTWiener', 'noiseCurve': 'exifNoiseProfile'}
        self.options = {'temporalFactor': 7500.0, 'spatialFactor': 10.0, 'ltmGain': -1, 'gtmContrast': 0.075, 'verbose': 2}
        self.rawpyParam = {
            'demosaic_algorithm' : rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
            'use_camera_wb' : False,
            'use_auto_wb' : True,
            'no_auto_bright': True,
            'bright': 0.03,
            'user_black' : 0,
            'exp_shift': 0.25,
            'gamma': (2.222, 1), 
            'chromatic_aberration':(3,1),
            'output_bps' : 16}
        
    def forward(self, input): # shape BS,bs,h,w
        singleBurst = input[0].numpy()
        burstSize,h,w = singleBurst.shape
        referenceImg = singleBurst[0,:,:] # [h,w] 
        alternateImgs = singleBurst[1:,:,:] # [bs-1,h,w]
        
        # Hdrplus pipeline
        motionVectors, alignedTiles = alignHdrplus(referenceImg,alternateImgs,self.mbSize)
        
        # Deblur
        motion_length, motion_angle = motion_vector_length(motionVectors)
        PSF = get_motion_blur(motion_length, motion_angle, alignedTiles)

        deb_img = np.zeros(alignedTiles.shape)
        deb_img[-1, ...] = alignedTiles[-1, ...]
        for j in range(PSF.shape[0]):
            for m in range(PSF.shape[1]):
                for n in range(PSF.shape[2]):
                    blurred = repeat(alignedTiles[j, m, n, ...], 'h w -> b h w c', b=1, c=1) 
                    PSF_trans = repeat(PSF[j, m, n, ...], 'h w -> h w c d', c=1, d=1)
                    # transfer tensor
                    blurred = torch.tensor(blurred)
                    PSF_trans = torch.tensor(PSF_trans)
                    deblur_img = inverse_filter(blurred, blurred, PSF_trans, init_gamma=1.5)
                    deblur_img = deblur_img.squeeze()
                    deblur_img = deblur_img.detach().numpy()

                    deb_img[j, m, n, ...] = deblur_img

        alignedTiles = deb_img
        
        mergedImage = mergeHdrplus(referenceImg, alignedTiles, self.padding, 
                                   self.lambdaS, self.lambdaR, self.params, self.options)
        mergedImage = np.clip(mergedImage,0,self.whiteLevel)
        
        # ISP process
        ISP_output = starlightFastISP(mergedImage)
        
        # To HSV and local tone mapping
        cfa = ISP_output['cfa']
        cfa = (cfa / 65535. * 255.).astype(np.float32) # scale and set type for cv2
        hsv = cv2.cvtColor(cfa, cv2.COLOR_RGB2HSV) 
        hsvOperator = AutoGammaCorrection(hsv)
        enhanceV = hsvOperator.execute()*255.
        hsv[...,-1] = enhanceV
        enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 
        return enhanceRGB,mergedImage

    def visualize_diff(self,subplotH,subplotW,ifAssert,TsaveFshowNone,ifWrite):
        self.count +=1
        vmax = np.max(self.diffFrames)
            
        figure = plt.figure(figsize=(subplotW*3,subplotH*2),dpi=400)
        for i in range(self.BucketSize):
            plt.subplot(subplotH,subplotW,i+1)
            plt.imshow(self.diffFrames[i],cmap='turbo',vmin=0,vmax=vmax/2)
            plt.tight_layout()
            plt.axis('off')
            
        if TsaveFshowNone:
            plt.savefig(f"/data/gravitychen/exp_data/star_light/alignment/color_diff_pth/mismatch{self.count}_vmax{vmax/2}.png")
        elif TsaveFshowNone == False:
            plt.show()
        elif TsaveFshowNone == None:
            pass
        
        if ifWrite:
            for i in range(self.BucketSize):
                plt.imsave(f"/data/gravitychen/exp_data/star_light/alignment/color_diff_patch/mismatch{self.count}_vmax{vmax/2}.png",self.diffFrames[i],cmap='turbo',vmax=vmax/2)

        if ifAssert:
            assert 1==2
            
    def visualize_overlap(self,overlap,index=0):
        tileSize = overlap.shape[-1]
        result = np.zeros([overlap.shape[1]*tileSize,overlap.shape[2]*tileSize])
        for heightNum in range(overlap.shape[1]):
            hindex = heightNum*tileSize
            for widthNum in range(overlap.shape[2]):
                windex = widthNum*tileSize
                result[hindex:hindex+tileSize,windex:windex+tileSize] = overlap[index][heightNum][widthNum] # self.diffFrames[0][hhh][www]
        return result
        
"""
figure = plt.figure(figsize=(20,10),dpi=400)
plt.subplot(2,2,1);
plt.axis('off')
# plt.title('R')
plt.imshow(self.raw_video[0,:,:,0],cmap='jet',vmin=0)
plt.subplot(2,2,2);
plt.axis('off')
# plt.title('G')
plt.imshow(self.raw_video[0,:,:,1],cmap='jet',vmin=0)
plt.subplot(2,2,3);
plt.axis('off')
# plt.title('B')
plt.imshow(self.raw_video[0,:,:,2],cmap='jet',vmin=0)
plt.subplot(2,2,4);
# plt.title('G - NIR')
plt.imshow(self.raw_video[0,:,:,3],cmap='jet',vmin=0)
plt.axis('off')
plt.tight_layout()
plt.show()
assert 1==2
"""
