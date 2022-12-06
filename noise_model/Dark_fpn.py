import os
import numpy as np
import glob
import rawpy

dark_raw_path = '/Users/xuanzhu/Documents/CUHKSZ/噪声建模/dark_fpn/'
# load image
darkraw_all = glob.glob(dark_raw_path + '*.ARW')
total_num = len(darkraw_all)
ind = []
for i in range(0,len(darkraw_all)):
    ind.append(int(darkraw_all[i].split('/')[-1].split('.')[0]))
ind = np.argsort(np.array(ind))
filepaths_all_sorted = np.array(darkraw_all)[ind]
raw_avg = 0
for i in range(total_num):
    raw_image_read = rawpy.imread(filepaths_all_sorted[i])
    raw_image = raw_image_read.raw_image_visible.astype(np.float32)
    raw_avg = raw_image + raw_avg

raw_avg = raw_avg / 30
fpn_total = np.std(raw_avg)
print(fpn_total)