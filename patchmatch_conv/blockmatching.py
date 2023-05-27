import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cosine
import cv2

def denoise_kernel(size=3):
    kernel = np.ones((size, size)) / size ** 2

    return kernel

def cosine_similarity(x, y):
    numrator = np.dot(x, y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    similarity = numrator / denominator

    return similarity

def Locate_blk(i, j, blk_step, block_Size, width, height):
    '''该函数用于保证当前的blk不超出图像范围'''
    if i*blk_step+block_Size < width:
        point_x = i*blk_step
    else:
        point_x = width - block_Size

    if j*blk_step+block_Size < height:
        point_y = j*blk_step
    else:
        point_y = height - block_Size

    m_blockPoint = np.array((point_x, point_y), dtype=int)  # 当前参考图像的顶点

    return m_blockPoint

def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    """该函数返回一个二元组（x,y）,用以界定_Search_Window顶点坐标"""
    point_x = _BlockPoint[0]  # 当前坐标
    point_y = _BlockPoint[1]  # 当前坐标

    # 获得SearchWindow四个顶点的坐标
    LX = point_x+Blk_Size/2-_WindowSize/2     # 左上x
    LY = point_y+Blk_Size/2-_WindowSize/2     # 左上y
    RX = LX+_WindowSize                       # 右下x
    RY = LY+_WindowSize                       # 右下y

    # 判断一下是否越界
    if LX < 0:   LX = 0
    elif RX > _noisyImg.shape[0]:   LX = _noisyImg.shape[0]-_WindowSize
    if LY < 0:   LY = 0
    elif RY > _noisyImg.shape[1]:   LY = _noisyImg.shape[1]-_WindowSize

    return np.array((LX, LY), dtype=int)

def patch_match(img1, img2, BlockPoint, mbsize, step, Search_Window):
    (present_x, present_y) = BlockPoint  # 当前坐标
    Blk_Size = mbsize
    Search_Step = step
    Window_size = Search_Window

    blk_positions = np.zeros((1, 2), dtype=int)  # 用于记录相似blk的位置
    Final_similar_blocks = np.zeros((Blk_Size, Blk_Size), dtype=float)

    img1_patch = img1[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]

    Window_location = Define_SearchWindow(img2, BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) / Search_Step  # 确定最多可以找到多少相似blk
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num ** 2, 2), dtype=int)
    similarity = np.zeros(blk_num ** 2, dtype=float)  # 记录各个blk与它的相似度

    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            img2_patch = img2[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            # 利用卷积响应找相似性
            # kernel = np.array([
            #     [-1, -1, -1],
            #     [-1, 8, -1],
            #     [-1, -1, -1]
            # ])
            # kernel = np.ones((3,3), np.float32)/9
            kernel = denoise_kernel(5)
            # kernel = np.array([
            #     [1, 0, -1],
            #     [1, 0, -1],
            #     [1, 0, -1]
            # ])
            resp_a = convolve2d(img1_patch, kernel, mode='same')
            # cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/patch_conv/feat_cur' + '.png',
            #             (resp_a))
            resp_b = convolve2d(img2_patch, kernel, mode='same')

            resp_norm1 = np.sqrt(np.sum(resp_a**2))
            resp_norm2 = np.sqrt(np.sum(resp_b ** 2))
            kernel_norm = np.sqrt(np.sum(kernel**2))
            m_Distance = np.dot(resp_a.ravel(), resp_b.ravel()) / (resp_norm1 * resp_norm2 * kernel_norm)
            # m_Distance = np.sum(resp_a * resp_b) / np.sqrt(np.sum(resp_a ** 2) * np.sum(resp_b ** 2))
            # m_Distance = np.mean(convolve2d(img2_patch, img1_patch, mode='same'))
            similar_blocks[matched_cnt, :, :] = img2_patch
            m_Blkpositions[matched_cnt, :] = (present_x, present_y)
            similarity[matched_cnt] = m_Distance
            matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    # similarity = similarity[:matched_cnt]
    Sort = similarity.argsort()
    Final_similar_blocks[:,:] = similar_blocks[Sort[-1], :, :]
    blk_positions[:] = m_Blkpositions[Sort[-1], :]
    cv2.imwrite('/home/cuhksz-aci-03/Documents/UltralLowLightRawVideoISP-main/results/patch_conv/feat_ref' + '.png',
                (Final_similar_blocks))

    return Final_similar_blocks, blk_positions