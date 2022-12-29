import numpy as np
import os
import rawpy
from glob import glob
from L_K import lucas_kanade
from motion_light import esti_motion
import cv2
from utils import *
from sklearn import mixture

raw_path = '/media/cuhksz-aci-03/数据/CUHK_SZ/'
seq_id = "17"
star_id = 0
num_load = 16
motion_list = []
motion_avers = []

def vevid_simple(rawimg, G, bais):
    out_img = np.arctan2(-G * (rawimg + bais), rawimg)

    return out_img

def predict_GMM(dataMat, components=3, tol= 0.001, iter = 10,cov_type="full"):
    clst = mixture.GaussianMixture(n_components=components, tol=tol, max_iter=iter,covariance_type=cov_type)
    clst.fit(dataMat)
    # predicted_labels =clst.predict(dataMat)
    # predict_pro = clst.predict_proba(dataMat)
    predict_samp = clst.sample(4)
    return predict_samp[0]


def pack_raw(raw):

    im = raw.raw_image_visible.astype(np.float32)
    im = im[792:2840, 732:2780]
    # im = np.expand_dims(im, axis=-1)
    # img_shape = im.shape
    # H = img_shape[0]
    # W = img_shape[1]
    #
    # out = np.concatenate((im[0:H:2, 0:W:2, :],
    #                       im[0:H:2, 1:W:2, :],
    #                       im[1:H:2, 1:W:2, :],
    #                       im[1:H:2, 0:W:2, :]), axis=-1)


    return im



def load_video_seq(folder_name, seqID, start_ind, num_to_load):
    base_name_seq = folder_name + 'seq' + str(seqID) + '/'  # starlight
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
    full_im = np.empty((num_to_load, 2048, 2048))
    for i in range(0, num_to_load):
        raw_img = rawpy.imread(filepaths_all_sorted[start_ind + i])
        full_im[i] = pack_raw(raw_img)

    return full_im

input_seq = load_video_seq(raw_path, seq_id, star_id, num_load)

for i in range(input_seq.shape[0]-2):
    motion = esti_motion(input_seq[i], input_seq[i+1], K=4)
    motion_list.append(motion*255)
    # motion_vector = lucas_kanade(input_seq[i]/16383, input_seq[i + 1]/16383)
    # motion = motion_vector[0] ** 2 + motion_vector[1] ** 2
    # motion_list.append(motion*255)
    # # 中心加权平均
    weight_map = gaussian_kernel(kernel_size=1024, sigma=300)
    motion_weight = np.mean(motion/255*16383*weight_map)
    motion_avers.append(motion_weight)

# fit a three-cluster GMM for before weighted average
# motion_array = (np.array(motion_avers)).reshape(-1, 1)
# pred_motion_min = predict_GMM(motion_array, components=3, tol= 0.001, iter = 100, cov_type="full")
out_path = '/home/cuhksz-aci-03/Desktop/handheld/output/'
filename = f"SEQ{seq_id}_motion_refine.mp4"
WriteVideoGear(motion_list, out_path, filename, gamma=True, normalize=True, fps=15)
for i in range(len(motion_list)):
    motion_img = cv2.normalize(motion_list[i], dst=None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    part_name = "/home/cuhksz-aci-03/Desktop/handheld/output/motion_" + str(i) + '.png'
    cv2.imwrite(part_name, (motion_img).astype(np.uint8))

