import os
import sys
import cv2
import rawpy
import time
import numpy as np
np.set_printoptions(precision=3) 
from glob import glob 
from numba import njit
from natsort import natsorted as sort
from Utility.finish import finish
from Utility.params import getParams
from Utility.queue import Queue
from Utility.utils import ARPS_search_frame, readTIFF,normalize,segmentImage, \
                            getSpatialL1,getDCTL1,bayerDownsampling,calculateWeight, \
                            readARW,TiffFastISP,ARWFastISP,ARWRawpyISP,writerVideo,\
                            displayVideo,getTime
from vidgear.gears import WriteGear
import matplotlib
import matplotlib.animation as animation
from matplotlib import pyplot as plt
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10


# Initialization
WRITE_VIDEO = True

# params = getParams('finish')
BLC_OFFSET = 5
DOWNSAMPLE_FACTOR = 2
BUCKET_SIZE = 32  
WHITE_LEVEL = 2**14 - 1  # 12 for crvd, 14 for DRV
BLOCK_SIZE = 24 #! sensitive
MATCH_THRESH = BLOCK_SIZE * BLOCK_SIZE * WHITE_LEVEL * 0.005 #! sensitive,  0.005 OK 

# merging parameters
WEIGHT_PROP = 3e-1 #! sensitive
WEIGHT_COEF = WEIGHT_PROP * WHITE_LEVEL 

# todo merging low light images OK

burstPath = "/data/gravitychen/dataset/DRV/0001/"
rawPathList = sort(glob(os.path.join(burstPath, '*.ARW')))
print('>>> rawPathList  ',len(rawPathList))
assert rawPathList != [], 'At least one .ARW file must be present in the burst folder.'
with rawpy.imread(rawPathList[0]) as raw:
    height, width = raw.raw_image.shape

queue = Queue(bucketSize=BUCKET_SIZE)
downQueue = Queue(bucketSize=BUCKET_SIZE)

# * video
if WRITE_VIDEO:
    outputPath = f"/data/gravitychen/exp_data/merging/"
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    fullPath = outputPath + f"offset{BLC_OFFSET}_FIFO{BUCKET_SIZE}_patch{BLOCK_SIZE}_c{WEIGHT_PROP}_RGB.mp4"
    writer = WriteGear(output_filename=fullPath,logging=True) # ,**output_params

currentTime = time.time()
for rawPathIdx in range(len(rawPathList)):
    # raw = readTIFF(rawPathList[rawPathIdx])
    raw = readARW(rawPathList[rawPathIdx],normalize=False,offset=BLC_OFFSET)
    currentTime = getTime(currentTime,"--- read one image")
    print('>>> raw.shape  ',raw.shape)
    # raw   ___________
    #  |--> |||||||    
    #       ———————————
    queue.enqueue(raw) 
    
    downQueue.enqueue(bayerDownsampling(raw,DOWNSAMPLE_FACTOR))
    
    # we now have the full bucket. Algorithm starts to work.    
    if rawPathIdx >= queue.bucketSize: 
        assert 1==2       
        #  ___________
        #  ||||||||||| <-- refRawImg
        #  ———————————
        refRawImg = queue.getFront() 
        downRefRawImg = downQueue.getFront() 
        
        simFrames = np.empty([BUCKET_SIZE,*refRawImg.shape])
        simFrames[-1] = refRawImg
        #  --->loop--->
        #  ___________
        #  ||||||||||| 
        #  ———————————
        for i in range(BUCKET_SIZE-1): # the last is the ref
            movedPreviousImg,localP,neighborP = ARPS_search_frame(refRawImg,queue.memory[i],\
                                        mbSize=BLOCK_SIZE,matchThresh=MATCH_THRESH,debug=False)
            simFrames[i] = movedPreviousImg
        # assert 1==2
        mergedImg = np.empty(refRawImg.shape)
        # patchwise loop and save DCT spatial distance
        hSegments, wSegments = segmentImage(refRawImg, BLOCK_SIZE)
        for y in range(0, int(hSegments*BLOCK_SIZE), BLOCK_SIZE):
            for x in range(0, int(wSegments*BLOCK_SIZE), BLOCK_SIZE):
                sim3dGroup = simFrames[:,y:y+BLOCK_SIZE,x:x+BLOCK_SIZE]
                
                # downsample group                
                # sim3dGroup = np.array([bayerDownsampling(i,DOWNSAMPLE_FACTOR) for i in sim3dGroup])
                # print('>>> sim3dGroup.shape  ',sim3dGroup.shape)
                
                # obtain distance 
                spatialDist = getSpatialL1(sim3dGroup,BUCKET_SIZE)
                dctDist = getDCTL1(sim3dGroup,BUCKET_SIZE)
                spatialWeight = calculateWeight(spatialDist,WEIGHT_COEF)
                dctWeight = calculateWeight(dctDist,WEIGHT_COEF)
                assert len(dctWeight) == BUCKET_SIZE
                mergedPatch = np.array([ sim3dGroup[i] * spatialWeight[i] for i in range(BUCKET_SIZE)])
                mergedPatch = sum(mergedPatch) / sum(spatialWeight)

                # print('>>> mergedImg.shape  ',mergedImg.shape)
                # print('>>> mergedPatch.shape  ',mergedPatch.shape)
                mergedImg[y:y+BLOCK_SIZE,x:x+BLOCK_SIZE] = mergedPatch
        
        # plt.subplot(1,1,1);
        # plt.title(f'last spatial weight {spatialWeight}')
        # plt.imshow(mergedImg,cmap='gray',vmax=WHITE_LEVEL)
        # plt.savefig(f"Docs/Images/Merging/DRV_merge_result/FIFO{BUCKET_SIZE}_patch{BLOCK_SIZE}_c{WEIGHT_PROP}.png",dpi=500)
        # Raw to RGB
        
        # plt.subplot(2,1,1);
        # fastRGBOut = ARWFastISP(mergedImg,outKey='rgb_image')
        # plt.imshow(fastRGBOut)
        
        
        
        # plt.subplot(1,1,1);
        rawpyRGBOut = ARWRawpyISP(rawPathList[rawPathIdx],mergedImg,use_camera_wb=True,half_size=True) # no_auto_bright=True, bright=10
        # plt.imshow(rawpyRGBOut)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(f"Docs/Images/Merging/DRV_merge_result/offset{BLC_OFFSET}_FIFO{BUCKET_SIZE}_patch{BLOCK_SIZE}_c{WEIGHT_PROP}_RGB.png",dpi=500)
        # assert 1==2
        # plt.show(block=False)
        # plt.pause(0.001)
        # plt.show()
        
        # * video
        if WRITE_VIDEO:
            writer.write(frame=rawpyRGBOut,rgb_mode=True)
        
        """
        assert 1==2
        read image and make video and make ISP video
        displayVideo(rawPathList)

        # writerVideo(rawPathList,height, width)
        # assert 1==2 
        """

        if False:
            # visualize stacks 
            fig,ax = plt.subplots()
            ax.grid(False)
            for idx,i in enumerate(range(16,8,-1)):
                plt.subplot(4,4,idx+1);
                plt.axis('off')
                plt.title('index : %d'%(i-idx-1))
                plt.imshow(sim3dGroup[i-idx-1],cmap='gray',vmax=2**12-1)
                
            X = np.arange(BUCKET_SIZE-1)+1
            plt.subplot(2,2,3);
            plt.title('Spatial Distance')
            plt.xticks(X)
            plt.bar(X,spatialDist,facecolor='lightskyblue',edgecolor='white',label="Spatial",lw=1,width=0.8) 
            plt.subplot(2,2,4);
            plt.title('DCT Distance')
            plt.xticks(X)
            plt.bar(X,dctDist,facecolor='yellowgreen',edgecolor='white',label="DCT",lw=1,width=0.8) 
            plt.tight_layout()
            plt.savefig(f"Docs/Images/Merging/DCT_Spatial_Distance_downsample/patch{BLOCK_SIZE}_h{y//BLOCK_SIZE}_w{x//BLOCK_SIZE}.png",dpi=1000)
        
# * video
if WRITE_VIDEO:
    writer.close()
    
    
    
