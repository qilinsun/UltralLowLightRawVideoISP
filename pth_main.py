import os
import cv2
import time
import numpy as np
from Utility.utils import getTime
from Utility import pth_utils 
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['font.family'] = 'serif' 

# parameter initialization
SEQ_ID = 17 # 9  17 23 35 37 
BATCH_SIZE = 1
BUCKET_SIZE = 8
WHITE_LEVEL = 2**16-1
isp_images = []
merge_images2 = []

# data loading
rootPath = "/data/gravitychen/dataset/starlight_denoising/submillilux_videos/submillilux_videos_mat/"
rawdataset = pth_utils.RawDataset(rootPath,BUCKET_SIZE=BUCKET_SIZE,seqID=SEQ_ID,whiteLevel=WHITE_LEVEL)
rawloader = DataLoader(dataset=rawdataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

# our pipeline
pipeline = pth_utils.LLR2V(blockSize=20,whiteLevel=WHITE_LEVEL)


#       ___           ___           ___     
#      /\  \         /\__\         /\__\    
#     /::\  \       /:/  /        /::|  |   
#    /:/\:\  \     /:/  /        /:|:|  |   
#   /::\~\:\  \   /:/  /  ___   /:/|:|  |__ 
#  /:/\:\ \:\__\ /:/__/  /\__\ /:/ |:| /\__\
#  \/_|::\/:/  / \:\  \ /:/  / \/__|:|/:/  /
#     |:|::/  /   \:\  /:/  /      |:/:/  / 
#     |:|\/__/     \:\/:/  /       |::/  /  
#     |:|  |        \::/  /        /:/  /   
#      \|__|         \/__/         \/__/    
modelTime = time.time()
for idx,i in enumerate(rawloader):
    isp,merge = pipeline(i)

    isp_images.append(isp)
    merge_images2.append(merge)
print('>>> Timing per burst',round((time.time() - modelTime), 4) / len(rawdataset))
    

#====================
#    write video
#====================
outputPath = f"/data/gravitychen/exp_data/star_light/ISP/local_tone_result/"
filename=f"SEQ{SEQ_ID}_hdrplus_LTM.mp4"
pth_utils.WriteVideoGear(isp_images,outputPath,filename,gamma=True,normalize=True,fps=8) #

# filename=f"{SEQ_ID}_hdrplus_tem{model.options['temporalFactor']}_spa{model.options['spatialFactor']}_FPNremoval__merge.mp4"
# pth_utils.WriteVideoGear(merge_images2,outputPath,filename,gamma=True,normalize=True,fps=8) #

#====================
#    write GIF
#====================
# import imageio
# outputPath = outputPath + "gamma**(2.2).gif"
# imageio.mimsave(outputPath, images,fps=10)





