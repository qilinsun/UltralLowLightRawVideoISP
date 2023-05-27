import cv2
import rawpy
import numpy as np
from glob import glob
from blockmatching import patch_match, Locate_blk

raw_path = '/media/cuhksz-aci-03/数据/CUHK_SZ/motion/0.01/'
seq_id = "iso51200"
star_id = 0
num_load = 2

Step1_Blk_Size = 8                     # block_Size即块的大小，8*8
Step1_Blk_Step = 3                      # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
Step1_Search_Step = 3                   # 块的搜索step
Step1_Search_Window = 512                # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

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

def init(img):
    """该函数用于初始化，返回用于记录过滤后图像以及权重的数组,还有构造凯撒窗"""
    m_shape = img.shape
    m_img = np.matrix(np.zeros(m_shape, dtype=float))

    return m_img

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
input_seq_1 = load_video_seq(raw_path, seq_id, star_id, num_load)

cur_img = input_seq[0]
ref_img = input_seq[1:, ...]

# patch match using conv
for i in range(ref_img.shape[0]):
    refimg = ref_img[i]
    for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
        curimgchannel = cur_img[di::2, dj::2]
        refimgchannel = refimg[di::2, dj::2]
        # 初始化参数
        (height, width) = curimgchannel.shape  # 得到图像的长宽
        block_Size = Step1_Blk_Size  # 块大小
        blk_step = Step1_Blk_Step  # N块步长滑动
        Width_num = (width - block_Size) / blk_step
        Height_num = (height - block_Size) / blk_step
        Basic_img = init(curimgchannel)
        for k in range(int(Height_num + 1)):
            for j in range(int(Width_num + 1)):
                m_blockPoint = Locate_blk(k, j, blk_step, block_Size, height, width)
                # img to patch
                matching_patch, match_position = patch_match(curimgchannel, refimgchannel, m_blockPoint, Step1_Blk_Size, Step1_Blk_Step, Step1_Search_Window)
                shape = matching_patch.shape
                # 重建位置需要改
                Basic_img[(k*blk_step):(k*blk_step)+shape[0], (j*blk_step):(j*blk_step)+shape[1]] = matching_patch

        input_seq_1[i+1, di::2, dj::2] = Basic_img

raw = '/media/cuhksz-aci-03/数据/CUHK_SZ/motion/0.01/iso51200/03660.ARW'
raw = rawpy.imread(raw)

for i in range(len(input_seq_1)):
    raw.raw_image_visible[720:2000, 724:2884] = (input_seq_1[i])
    post_raw = raw.postprocess(**rawpyParam)
    cfa = (post_raw[720:2000, 724:2884] / 65535. * 255.).astype(np.float32)  # scale and set type for cv2
    hsv = cv2.cvtColor(cfa, cv2.COLOR_RGB2HSV)
    hsvOperator = AutoGammaCorrection(hsv)
    enhanceV = hsvOperator.execute()
    hsv[..., -1] = enhanceV * 255.
    enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    enhanceRGB = cv2.cvtColor(enhanceRGB, cv2.COLOR_RGB2BGR)
    img_write = cv2.normalize(((enhanceRGB)**1/2.2), dst=None, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/pm_conv_51200_' + str(i) + '.png', (img_write))