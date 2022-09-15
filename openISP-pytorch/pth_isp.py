import sys
sys.path.append("../openISP/")
from model import dpc,blc,aaf,awb,cnf,cfa,gac,ccm,csc,bnf,eeh,fcs,bcc,hsc,nlm

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
# from einops.layers.torch import Rearrange
from einops import rearrange,repeat

def dead_pixel_correction(raw_img,thres=30,mode ='gradient',clip=1023):
    """ Dead Pixel Correction

    Args:
        raw_img (torch.FloatTensor): raw data with shape(w,h)
        thres (int, optional): dead pixel threshold for detection. Defaults to 30.
        mode (str, optional): pixel correction mode. Defaults to 'gradient'.
        clip (int, optional): set max values. Defaults to 1023.

    Returns:
        (torch.FloatTensor): processed_raw_img
    """
    raw_img = torch.unsqueeze(raw_img,dim=0)
    img_pad = F.pad(raw_img, pad=(2, 2, 2, 2), mode="reflect")
    img_pad  = torch.squeeze(img_pad)
    _,raw_h,raw_w = raw_img.shape
    dpc_img = torch.empty((raw_h, raw_w))
    for y in range(raw_h):
        for x in range(raw_w):
            p0 = img_pad[y + 2, x + 2]
            p1 = img_pad[y, x]
            p2 = img_pad[y, x + 2]
            p3 = img_pad[y, x + 4]
            p4 = img_pad[y + 2, x]
            p5 = img_pad[y + 2, x + 4]
            p6 = img_pad[y + 4, x]
            p7 = img_pad[y + 4, x + 2]
            p8 = img_pad[y + 4, x + 4]
            if (abs(p1 - p0) > thres) and (abs(p2 - p0) > thres) and (abs(p3 - p0) > thres) \
                and (abs(p4 - p0) > thres) and (abs(p5 - p0) > thres) and (abs(p6 - p0) > thres) \
                and (abs(p7 - p0) > thres) and (abs(p8 - p0) > thres):
                if mode == 'mean':
                    p0 = (p2 + p4 + p5 + p7) / 4
                elif mode == 'gradient':
                    dv = abs(2 * p0 - p2 - p7)
                    dh = abs(2 * p0 - p4 - p5)
                    ddl = abs(2 * p0 - p1 - p8)
                    ddr = abs(2 * p0 - p3 - p6)
                    if (min(dv, dh, ddl, ddr) == dv): # 边缘的梯度最小 沿着边缘求值
                        p0 = (p2 + p7 + 1) / 2
                    elif (min(dv, dh, ddl, ddr) == dh):
                        p0 = (p4 + p5 + 1) / 2
                    elif (min(dv, dh, ddl, ddr) == ddl):
                        p0 = (p1 + p8 + 1) / 2
                    else:
                        p0 = (p3 + p6 + 1) / 2
            dpc_img[y, x] = p0
    return torch.clip(dpc_img,min=0,max=clip)
    
def black_level_compensation(raw_img, r_offset=0, b_offset=0,gr_offset=0,alpha=0,gb_offset=0,beta=0, bayer_pattern='rggb', clip=1023):
    """Black Level Compensation

    Args:
        raw_img (torch.FloatTensor): [description]
        r_offset (int, optional): red offset. Defaults to 0.-*
        b_offset (int, optional): blue offset. Defaults to 0.
        gr_offset (int, optional): green offset. Defaults to 0.
        gb_offset (int, optional): Gb offset. Defaults to 0.
        alpha (int, optional): red coefficient. Defaults to 0.
        beta (int, optional): blue coefficient. Defaults to 0.
        bayer_pattern (str, optional): [bayer_pattern]. Defaults to 'rggb'.
        clip (int, optional): set max values. Defaults to 1023.

    Raises:
        NotImplementedError: Only works for RGGB pattern

    Returns:
        (torch.FloatTensor): processed_raw_img
    """

    if bayer_pattern not in ['rggb','RGGB']:
        raise NotImplementedError

    # R channel
    r = raw_img[::2,::2] + r_offset
    raw_img[::2,::2] = r

    # B channel
    b = raw_img[1::2,1::2] + b_offset
    raw_img[1::2,1::2] = b

    # Gr channel
    raw_img[::2,1::2] = raw_img[::2,1::2] + gr_offset + alpha * r / 256
    # Gb channel
    raw_img[1::2,::2] = raw_img[1::2,::2] + gb_offset + beta * b / 256

    blc_img = torch.clip(raw_img,min=0,max=clip)
    return blc_img

def lens_shaing_mask(res_h,res_w):
    """generate a lens shading mask

    Args:
        res_h (int): x resolution
        res_w (int): y resolution

    Usage:
        lens_shaing_mask(res_h = 12,res_w = 18)
    """
    range_h = torch.arange(-res_h,res_h,2)
    range_w = torch.arange(-res_w,res_w,2)
    # symmetric 
    h = torch.cat([range_h[1:len(range_h)//2+1],range_h[len(range_h)//2:]])
    w = torch.cat([range_w[1:len(range_w)//2+1],range_w[len(range_w)//2:]])

    [x, y] = torch.meshgrid(h,w) # -res_w // 2:res_w // 2
    print('>>>  x ',x);print('>>>  x ',x.shape)
    print('>>> y  ',y);print('>>> y  ',y.shape)

    # Assume distance to source is approx. constant over wave
    curvature = torch.sqrt(x ** 2 + y ** 2)
    mask = (curvature.max() - curvature)/curvature.max()
    # plt.imshow(mask,cmap='Greys_r')
    # plt.colorbar()
    # plt.show()

def anti_aliasing_filter_openISP(raw_img):
    raw_img = torch.unsqueeze(raw_img,dim=0)
    img_pad = F.pad(raw_img, pad=(2, 2, 2, 2), mode="reflect")
    img_pad  = torch.squeeze(img_pad)
    _,raw_h,raw_w = raw_img.shape

    aaf_img = torch.empty((raw_h, raw_w))
    for y in range(raw_h):
        for x in range(raw_w):
            p0 = img_pad[y + 2, x + 2]
            p1 = img_pad[y, x]
            p2 = img_pad[y, x + 2]
            p3 = img_pad[y, x + 4]
            p4 = img_pad[y + 2, x]
            p5 = img_pad[y + 2, x + 4]
            p6 = img_pad[y + 4, x]
            p7 = img_pad[y + 4, x + 2]
            p8 = img_pad[y + 4, x + 4]
            aaf_img[y, x] = (p0*8+p1+p2+p3+p4+p5+p6+p7+p8)/16
    return aaf_img

def anti_aliasing_filter_conv2D(raw_img):
    raw_h,raw_w = raw_img.shape
    raw_img = raw_img.view(1,1,raw_h,raw_w).float()
    img_pad = F.pad(raw_img, pad=(2,2,2,2), mode="reflect")

    kernel = torch.tensor([[1,1,1],[1,8,1],[1,1,1]]).view(1,1,3,3)/16 
    aaf_img = F.conv2d(img_pad,kernel,stride=1,padding=0,dilation=2)
    return aaf_img.view(raw_h,raw_w)

def tone_mapping(raw_img):
    # HDR to LDR
    # https://www.zhihu.com/search?type=content&q=%E8%89%B2%E8%B0%83%E6%98%A0%E5%B0%84
    pass

def auto_white_balance(raw_img,clip,r_gain,gr_gain,gb_gain,b_gain,bayer_pattern="rggb"):
    """auto white balance

    Args:
        raw_img (torch.FloatTensor): shape [w,h]
        clip (int): clip max number
        r_gain (int): gain for Red channel
        gr_gain (int): gain for Green_r channel
        gb_gain (int): gain for Green_b channel
        b_gain (int): gain for blue channel
        bayer_pattern (str, optional): Defaults to "rggb".

    Raises:
        NotImplementedError: when the bayer_pattern is not rggb

    Returns:
        torch.FloatTensor: white balanced raw image
    """
    # "Gray world" algorithm

    # https://github.com/guochengqian/TENet/blob/013608976e1c1f2b0e7a2d6cb832554ce081d61a/datasets/process.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L28
    if bayer_pattern.lower() != 'rggb':
        raise NotImplementedError
    
    # R channel
    raw_img[::2,::2] = raw_img[::2,::2] * r_gain
    # Gr channel
    raw_img[::2,1::2] = raw_img[::2,1::2] * gr_gain
    # Gb channel
    raw_img[1::2,::2] = raw_img[1::2,::2] * gb_gain
    # B channel
    raw_img[1::2,1::2] = raw_img[1::2,1::2] * b_gain

    awb_img = torch.clip(raw_img,min=0,max=clip)
    return awb_img

def demosaic(raw_img):
    """ refer to https://github.com/guochengqian/TENet/blob/013608976e1c1f2b0e7a2d6cb832554ce081d61a/datasets/processdnd.py#L54

    Args:
        raw_img (torch.FloatTensor,shape: [2w,2h])
        # bayer_images (torch.FloatTensor,shape: [1,4,w,h]) 
    
    Outputs:
        torch.FloatTensor,shape: [2w,2h,3]
    """
    def SpaceToDepth_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, C, H // bs, bs, W // bs, bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (bs ** 2), H // bs, W // bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def DepthToSpace_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, bs, bs, C // (bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (bs ** 2), H * bs, W * bs)  # (N, C//bs^2, H * bs, W * bs)
        return x

    r = torch.unsqueeze(raw_img[::2,::2],dim=0)
    gr = torch.unsqueeze(raw_img[::2,1::2],dim=0)
    gb = torch.unsqueeze(raw_img[1::2,::2],dim=0)
    b = torch.unsqueeze(raw_img[1::2,1::2],dim=0)
    bayer_images = torch.unsqueeze( torch.cat([r,gr,gb,b],dim=0),dim=0)

    """Bilinearly demosaics a batch of RGGB Bayer images."""
    bayer_images = bayer_images.permute(0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format

    shape = bayer_images.size()
    shape = [shape[1] * 2, shape[2] * 2]

    red = bayer_images[Ellipsis, 0:1]
    upsamplebyX = nn.Upsample(size=shape, mode='bilinear', align_corners=False)
    red = upsamplebyX(red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # [1,2w,2h,1]

    green_red = bayer_images[Ellipsis, 1:2] # [1,w,h,1]
    green_red = torch.flip(green_red, dims=[1])  # Flip left-right [1,w,h,1]
    green_red = upsamplebyX(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [1,2w,2h,1]
    green_red = torch.flip(green_red, dims=[1])  # Flip left-right # [1,2w,2h,1]
    green_red = SpaceToDepth_fact2(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # [1,w,h,4]

    #!=========
    #! bug
    green_blue = bayer_images[Ellipsis, 2:3] # [1,w,h,1]
    green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down # [1,w,h,1]
    green_blue = upsamplebyX(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # [1,2w,2h,1]
    green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down [1,2w,2h,1]
    green_blue = SpaceToDepth_fact2(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # [1,w,h,4]
    #!=========

    green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
    green_at_green_red = green_red[Ellipsis, 1]
    green_at_green_blue = green_blue[Ellipsis, 2]
    green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = DepthToSpace_fact2(torch.stack(green_planes, dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    blue = bayer_images[Ellipsis, 3:4]
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
    blue = upsamplebyX(blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

    rgb_images = torch.cat([red, green, blue], dim=-1)
    rgb_images = rgb_images.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return torch.squeeze(rgb_images.permute(0, 2, 3, 1))

def gamma_correction(rgb_img, gamma, gain = 1):
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

    # Clamps to prevent numerical instability of gradients near zero.
    rgb_img = torch.clamp(rgb_img, min=1e-8)
    result = gain * rgb_img ** gamma
    return result
    
def color_correction_matrix(rgb_img,ccm):
    """color space correction

    For XYZ conversion, you can refer to # https://github.com/guochengqian/TENet/blob/013608976e1c1f2b0e7a2d6cb832554ce081d61a/datasets/processdnd.py#L115 
    Args:
        rgb_img (torch.FloatTensor): shape [w,h,3]
        ccm (torch.FloatTensor): shape [3,4]

    Raises:
        TypeError: [description]
    """
    if len(rgb_img.shape) != 3:
        raise TypeError("Input should be a 3-channel image")

    ccm = torch.tensor([[1.,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0]])
    rgb_h,rgb_w,_ = rgb_img.shape
    homo_rgb = torch.cat([rgb_img,torch.ones([rgb_h,rgb_w,1])],dim=-1)
    cc_rgb = homo_rgb @ ccm.T
    return cc_rgb

def edge_enhancement(rgb_img):
    # unsharp masking
    # https://en.wikipedia.org/wiki/Unsharp_masking

    rgb_img = rearrange(rgb_img,'h w c -> 1 c h w')

    kernel = torch.tensor([[0,-1,0],[-1,5.,-1],[0,-1,0]]) 
    kernel = repeat(kernel, 'h w -> c 1 h w', c=3)

    ee_img = F.conv2d(rgb_img,kernel,padding=1,groups=3)
    return ee_img

DEAD_PIXEL_CORRECTION = 0
BLACK_LEVEL_COMPENSATION = 0
LENS_SHAING_MASK = 0
ANTI_ALIASING_FILTER_CONV2D = 1
AUTO_WHITE_BALANCE = 1
DEMOSAIC = 1
GAMMA_CORRECTION = 1

# raw_path = "test.raw" # 10 bits
# raw_h, raw_w, bits = 1080,1920,10
raw_path = "raw_images/DSC_1339_768x512_rggb.raw" # 14 bits
raw_h, raw_w, bits = 512,768,14
assert (not raw_h%2) and (not raw_w%2)


# read raw image
rawimg = np.fromfile(raw_path, dtype='int16', sep='')
rawimg = rawimg.reshape([raw_h, raw_w])
print('>>>  rawimg.max() ',rawimg.max())
print(50*'-' + '\nLoading RAW Image Done......')
print('>>>  rawimg.shape ',rawimg.shape)
pth_rawimg = torch.from_numpy(rawimg).float()

#=================
# dead pixel correction -- dpc
#=================
if DEAD_PIXEL_CORRECTION:
    # # openISP
    # open_dpc = dpc.DPC(rawimg, thres=30, mode="gradient", clip=2**bits-1).execute()
    # print(50*'-' + '\nDead Pixel Correction Done......')
    # print('>>>  open_dpc.shape ',open_dpc.shape)

    # pytorch
    pth_rawimg = dead_pixel_correction(pth_rawimg,clip=2**bits-1)
    print(50*'-' + '\nPytorch Dead Pixel Correction Done......')
    print('>>>  pth_dpc.shape ',pth_rawimg.shape)

    plt.subplot(1,2,1)
    plt.imshow(rawimg,cmap='gray',vmax=2**bits)
    plt.subplot(1,2,2)
    plt.imshow(pth_rawimg,cmap='gray',vmax=2**bits)
    plt.show()
#=================
# black level compensation -- blc
#=================
if BLACK_LEVEL_COMPENSATION:
    # openISP
    # open_blc = blc.BLC(rawimg, parameter=[0, 0, 0, 0, 0, 0], bayer_pattern="rggb", clip=2**bits-1).execute()
    # print(50*'-' + '\nDead Pixel Correction Done......')
    # print('>>>  open_blc.shape ',open_blc.shape)

    # pytorch
    pth_rawimg = black_level_compensation(pth_rawimg,r_offset=0, b_offset=0,gr_offset=0,alpha=0,gb_offset=0,beta=0,clip=2**bits-1)
    print(50*'-' + '\nPytorch Black Level Compensation Done......')
    print('>>>  pth_blc.shape ',pth_rawimg.shape)

    plt.subplot(1,2,1)
    plt.imshow(rawimg,cmap='gray',vmax=2**bits)
    plt.subplot(1,2,2)
    plt.imshow(pth_rawimg,cmap='gray',vmax=2**bits)
    plt.show()

#=================
# lens shading correction
#=================
if LENS_SHAING_MASK:
    # simulate a lens shading 
    # lens_shaing_mask(res_h = 12,res_w = 18)
    # print(50*'-' + '\nPytorch lens shading correction Done......')
    pass

#=================
# anti-aliasing filter 
#=================
# low pass filter
if ANTI_ALIASING_FILTER_CONV2D:
    pth_rawimg = anti_aliasing_filter_conv2D(pth_rawimg)
    pth_rawimg_aaf = pth_rawimg
    print(50*'-' + '\nPytorch anti-aliasing filter Done......')
    print('>>>  pth_aaf.shape ',pth_rawimg.shape)

    # plt.subplot(1,2,1)
    # plt.imshow(rawimg,cmap='gray',vmax=2**bits)
    # plt.subplot(1,2,2)
    # plt.imshow(pth_rawimg,cmap='gray',vmax=2**bits)
    # plt.show()
#=================
# auto white balance 
#=================
# B gain and R gain is calculated from RG and BG ratio
if AUTO_WHITE_BALANCE:
    r_gain = 1.5
    gr_gain = 1.0
    gb_gain = 1.0
    b_gain = 1.1
    pth_rawimg = auto_white_balance(pth_rawimg_aaf,clip=2**bits-1,
                                r_gain=r_gain,gr_gain=gr_gain,
                                gb_gain=gb_gain,b_gain=b_gain)
    pth_rawimg_awb = pth_rawimg
    print(50*'-' + '\nPytorch auto white balance Done......')
    print('>>>  pth_awb.shape ',pth_rawimg.shape)

    # plt.subplot(1,2,1)
    # plt.imshow(rawimg,cmap='gray',vmax=2**bits)
    # plt.subplot(1,2,2)
    # plt.imshow(pth_rawimg,cmap='gray',vmax=2**bits)
    # plt.show()
#=================
# demosaicking 
#=================
# bilinear interpolation
if DEMOSAIC:
    pth_rawimg = demosaic(pth_rawimg)
    pth_rawimg_dms = pth_rawimg
    print(50*'-' + '\nPytorch CFA demosaicking Done......')
    print('>>>  dms_rgb.shape ',pth_rawimg.shape)
    print('>>>  dms_rgb.max() ',pth_rawimg.max())

    # plt.subplot(1,2,1)
    # plt.imshow(rawimg,cmap='gray',vmax=2**bits)
    # plt.subplot(1,2,2)
    # plt.imshow(pth_rawimg_dms/pth_rawimg_dms.max())
    # plt.show()
#=================
# gamma correction
#=================
if GAMMA_CORRECTION:
    pth_rawimg = gamma_correction(pth_rawimg, gamma=0.9, gain = 1)
    pth_rawimg_gc = pth_rawimg
    print(50*'-' + '\nPytorch gamma correction Done......')
    print('>>>  gc_rgb.shape ',pth_rawimg.shape)

    plt.subplot(1,2,1)
    plt.imshow(pth_rawimg_dms/pth_rawimg_dms.max())
    plt.subplot(1,2,2)
    plt.imshow(pth_rawimg_gc/pth_rawimg_gc.max())
    plt.show()
# plt.imshow(pth_rawimg,vmax=2**bits)

