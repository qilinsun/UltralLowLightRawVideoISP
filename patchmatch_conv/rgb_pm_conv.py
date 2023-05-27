import cv2
import rawpy
import numpy as np
from glob import glob
from blockmatching import patch_match, Locate_blk
from scipy.signal import convolve2d

img_list = []

Step1_Blk_Size = 256 # 256 16                    # block_Size即块的大小，8*8
Step1_Blk_Step = 64 # 64 6                     # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
Step1_Search_Step = 64 # 64 6                  # 块的搜索step
Step1_Search_Window = 700 # 700 32               # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

def denoise_kernel(size=3):
    kernel = np.ones((size, size)) / size ** 2

    return kernel

def init(img):
    """该函数用于初始化，返回用于记录过滤后图像以及权重的数组,还有构造凯撒窗"""
    m_shape = img.shape
    m_img = np.matrix(np.zeros(m_shape, dtype=float))

    return m_img
path1 = '/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/ori_2500_0.png'
path2 = '/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/ori_2500_2.png'
img_1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2GRAY)
# img_512 = img_1[0:256, 0:256]
cv2.imwrite('/home/cuhksz-aci-03/Documents/PatchMatch-master/img1_gray' + '.png', (img_1))
img_list.append(img_1)
img_2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2GRAY)
cv2.imwrite('/home/cuhksz-aci-03/Documents/PatchMatch-master/img2_gray' + '.png', (img_2))
img_list.append(img_2)

input_seq = np.array(img_list)
input_seq_1 = np.array(img_list)

cur_img = input_seq[0]
ref_img = input_seq[1:, ...]

# patch match using conv
for i in range(ref_img.shape[0]):
    refimg = ref_img[i]
    # 初始化参数
    (height, width) = cur_img.shape  # 得到图像的长宽
    block_Size = Step1_Blk_Size  # 块大小
    blk_step = Step1_Blk_Step  # N块步长滑动
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step
    Basic_img = np.array(init(cur_img))
    for k in range(int(Height_num + 1)):
        for j in range(int(Width_num + 1)):
            m_blockPoint = Locate_blk(k, j, blk_step, block_Size, height, width)
            # img to patch
            matching_patch, match_position = patch_match(cur_img, refimg, m_blockPoint, Step1_Blk_Size, Step1_Blk_Step, Step1_Search_Window)
            cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/patch_conv/matchpatch256' + '.png',
                        (matching_patch))
            shape = matching_patch.shape
            Basic_img[(k*blk_step):(k*blk_step)+shape[0], (j*blk_step):(j*blk_step)+shape[1]] = matching_patch
            patch_img = cv2.cvtColor(Basic_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(
            #     '/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/patch_conv/pm_rgb_' + str(i) + '.png',
            #     (patch_img))
    input_seq_1[i+1, ...] = Basic_img


for i in range(len(input_seq_1)):

    enhanceRGB = cv2.cvtColor(input_seq_1[i], cv2.COLOR_GRAY2BGR)
    # enhanceRGB = cv2.cvtColor(enhanceRGB, cv2.COLOR_RGB2BGR)
    img_write = cv2.normalize(((enhanceRGB)), dst=None, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite('/home/cuhksz-aci-03/Documents/PatchMatch-master/pm_conv_cup_' + str(i) + '.png', (img_write))