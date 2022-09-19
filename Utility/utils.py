import os
import sys
import cv2
import time
import rawpy
import torch
import numpy as np
# import cupy as cp
from libtiff import TIFF
from timeit import default_timer as timer  
from matplotlib import image, pyplot as plt

sys.path.append("Utility/fast-openISP")
from pipeline import Pipeline
from util.yacs import Config


def normalize(item):
    assert item.min() == item.max()
    return (item - item.min()) / (item.max() -item.min())

def readTIFF(rawPath,normalize=False):
    """ 
    To read an image in the currect TIFF directory and return it as numpy array:
    
    ===== Information of CRVD dataset =====
    The Bayer pattern of raw data : GBRG, 
    black level : 240, 
    white level is 2^12-1
    """
    image = TIFF.open(rawPath, mode='r').read_image()
    
    if normalize:
        image = (image-240)/(2**12-1 - 240)
    return image

def readARW(rawPath,normalize=False,offset=10):
    """ 
    To read an image in the currect ARW directory and return it as numpy array:
    
    ===== Information of DRV dataset =====
    The Bayer pattern of raw data : RGGB
    """
    
    with rawpy.imread(rawPath) as raw:
        image = raw.raw_image_visible
        
        if normalize:
            white_level = int(raw.white_level)
            assert raw.black_level_per_channel[0]==raw.black_level_per_channel[1]==raw.black_level_per_channel[2]==raw.black_level_per_channel[3]
            black_level = raw.black_level_per_channel[0]
            image = (image + offset - black_level)/(white_level - black_level)
        else:
            image = image + offset
            
    return image

def getTime(currentTime, labelName, printTime=True, spaceSize=50):
    '''Print the elapsed time since currentTime. Return the new current time.'''
    if printTime:
        print(labelName, ' ' * (spaceSize - len(labelName)), ': ', round((time.time() - currentTime) * 1000, 2), 'milliseconds  ,  ',round((time.time() - currentTime), 4),"seconds")
    return time.time()

def getSigned(array):
    '''Return the same array, casted into a signed equivalent type.'''
    # Check if it's an unssigned dtype
    dt = array.dtype
    if dt == np.uint8:
        return array.astype(np.int16)
    if dt == np.uint16:
        return array.astype(np.int32)
    if dt == np.uint32:
        return array.astype(np.int64)
    if dt == np.uint64:
        return array.astype(np.int)

    # Otherwise, the array is already signed, no need to cast it
    return array

def isTypeInt(array):
    '''Check if the type of a numpy array is an int type.'''
    return array.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.uint, np.int]

def TiffFastISP(tiffIn,outKey='rgb_image'):
    """
    Keys of pipeline.execute output:
        (['bayer', 'rgb_image', 'y_image', 'cbcr_image', 'edge_map', 'output'])
    """
    cfg = Config('Utility/fast-openISP/configs/tiff.yaml')
    pipeline = Pipeline(cfg)
    
    tiffIn = tiffIn.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
    tiffOut, _ = pipeline.execute(tiffIn)
    
    return tiffOut[outKey]

def ARWFastISP(arwIn,outKey='rgb_image'):
    """
    Keys of pipeline.execute output:
        (['bayer', 'rgb_image', 'y_image', 'cbcr_image', 'edge_map', 'output'])
    """
    cfg = Config('Utility/fast-openISP/configs/arw.yaml')
    pipeline = Pipeline(cfg)
    
    arwIn = arwIn.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
    arwOut, _ = pipeline.execute(arwIn)
    
    return arwOut[outKey]

def ARWRawpyISP(arwIn, bayer_array=None, **kwargs):
    """using a raw file [and modified bayer pixels], get rgb pixels"""
    # open the raw image
    with rawpy.imread(arwIn) as raw:
        # overwrite the original bayer array
        if bayer_array is not None:
            raw.raw_image_visible[:] = bayer_array
            # raw.raw_image[:] = bayer_array 
        start = timer() 
        rgb = raw.postprocess(**kwargs)
        end = timer()  
        print("ARWRawpyISP",end - start)  
    return rgb

""" 
==============================
    for motion estimation
==============================
""" 

class MotionVectorClass:
    def __init__(self,h,w,blkSize,vx=0, vy=0, mag2=0):
        self.blkH = h // blkSize
        self.blkW = w // blkSize
        self.vx = vx
        self.vy = vy
        self.mag2 = mag2

        self.mvArray = torch.zeros([self.blkH * self.blkW,3])  #todo reshape can solve the 1D issue.
        self.mvArray_np = np.zeros([self.blkH * self.blkW,3])  #todo reshape can solve the 1D issue.
        # print('>>> class test self.mvArray[0]  ',self.mvArray[0])
        # print('>>> self.blkH  ',self.blkH)
        # print('>>> self.blkW  ',self.blkW)


def segmentImage(reference, blockSize=8):
    """
    Determines how many macroblocks an image is composed of
    :param reference: I-Frame
    :param blockSize: Size of macroblocks in pixels
    :return: number of rows and columns of macroblocks within
    """
    h, w = reference.shape
    hSegments = h // blockSize
    wSegments = w // blockSize
    totBlocks = hSegments * wSegments

    assert hSegments * blockSize == h
    assert wSegments * blockSize == w
    #if debug:
    #    print(f"Height: {h}, Width: {w}")
    #    print(f"Segments: Height: {hSegments}, Width: {wSegments}")
    #    print(f"Total Blocks: {totBlocks}")

    return hSegments, wSegments

def segmentImageHalfOverlap(reference, blockSize=8):
    """
    Determines how many overlap macroblocks an image is composed of
    :param reference: I-Frame
    :param blockSize: Size of macroblocks in pixels
    :return: number of rows and columns of macroblocks within
    """
    steps = blockSize//2
    h, w = reference.shape
    hSegments = h // steps
    wSegments = w // steps
    totBlocks = hSegments * wSegments

    assert hSegments * steps == h
    assert wSegments * steps == w
    #if debug:
    #    print(f"Height: {h}, Width: {w}")
    #    print(f"Segments: Height: {hSegments}, Width: {wSegments}")
    #    print(f"Total Blocks: {totBlocks}")

    return hSegments, wSegments

def getSAD(currentBlock, refBlock):
    """
    Returns Mean Absolute Difference between current frame macroblock (currentBlock) and anchor frame macroblock (refBlock)
    """
    return np.sum(np.abs(currentBlock - refBlock))/(currentBlock.shape[0]*currentBlock.shape[1])
def weightedSAD(currentBlock, refBlock):
    """
    Returns Mean Absolute Difference between current frame macroblock (currentBlock) and anchor frame macroblock (refBlock)
    """
    return np.sum(np.abs(currentBlock - refBlock) / (currentBlock.shape[0]*currentBlock.shape[1])) / np.sum(currentBlock+refBlock)

def pthSAD(currentBlock, refBlock):
    return torch.sum(torch.abs(currentBlock - refBlock))/(currentBlock.shape[0]*currentBlock.shape[1])

def getRealImage(refPath,curPath,cutting=True):
    """
    Args:
        cutting (bool, optional): return smaller region. Defaults to True.

    Returns:
        _type_: _description_
    """
    refImg = cv2.imread(refPath,cv2.IMREAD_GRAYSCALE)
    curImg = cv2.imread(curPath,cv2.IMREAD_GRAYSCALE)   
    
    if cutting:
        h, w = curImg.shape
        return  curImg[:h//4,:w//4],refImg[:h//4,:w//4]
    else:
        return curImg,refImg

def getRandomImage(height,width,downStep,leftStep,plot=False):
    """
    generate random mask image. 
    Randomly black or white.

    Args:
        height (int)
        width (int)
        downStep (int): set how many step to roll downward 
        leftStep (int): set how many step to roll leftward
        plot (bool, optional)

    Returns:
        currentImg , referenceImg(obtained by rolling current image)
    """
    
    currentImg = np.random.randint(2, size=width*height).reshape([height,width]) * 255
    referenceImg = np.roll(currentImg,downStep,axis=0) # down-rolling  2 rows
    referenceImg = np.roll(referenceImg,leftStep,axis=1) # left-rolling  2 rows
    if plot:
        plt.subplot(1,2,1);
        plt.title("currentImg");
        plt.imshow(currentImg,cmap='gray')
        plt.subplot(1,2,2);
        plt.title("referenceImg");
        plt.imshow(referenceImg,cmap='gray')
        plt.show()
        assert 1==2
    return  currentImg,referenceImg

def ARPS_search_encoding(referenceImg,currentImg,mbSize,matchThresh):
    """
    Adaptive Rood Pattern Search Algorithm
        checking similarity of close blocks, if similar, MV = 0;
        else search neighborhood by large pattern
        and small pattern repeatly
    
    return moved reference image (the terminalogy in video encoding) and difference image
    """
    assert referenceImg.shape == currentImg.shape
    h,w = currentImg.shape
    
    p = 6  # the boundary for small search pattern
    maxMag2 = 0 
    stepSize = 0
    maxIndex = -1
    
    costs = np.array([np.inf]*6)
    LDSP = np.zeros([6,2]) # The index points for Large Diamond Search pattern (LDSP)
    SSP = np.array([[-1,0],[0, -1],[0, 0],[0, 1],[1, 0],[1, 1]]) 
    diffImg = np.zeros([h,w])
    movedImg = np.zeros([h,w])
    motionVec = MotionVectorClass(h,w,mbSize)
    
    hSegments, wSegments = segmentImage(referenceImg, mbSize)

    for y in range(0, int(hSegments*mbSize), mbSize):
        for x in range(0, int(wSegments*mbSize), mbSize):
            motionIdx = y//mbSize * wSegments + x//mbSize
            
            currentBlock = currentImg[y:y+mbSize,x:x+mbSize]
            referenceBlock = referenceImg[y:y+mbSize,x:x+mbSize]
            
            #==================
            #      Step 1    
            #==================
            costs[2] = getSAD(currentBlock,referenceBlock)
            print('>>> getSAD  ',costs[2])
            # if similar to previous block, MV set 0
            if( costs[2] < matchThresh ):  
                motionVec.vy = 0; motionVec.vx = 0; motionVec.mag2 = 0; 
                motionInfo = np.array([motionVec.vy , motionVec.vx , motionVec.mag2])
                motionVec.mvArray[motionIdx,:] = motionInfo
                diffImg[y:y+mbSize,x:x+mbSize] = 128
                movedImg[y:y+mbSize,x:x+mbSize] = referenceBlock
                continue
            
            # if left side of image 
            if x==0:
                stepSize = 2
                maxIndex = 5
            else:
                lastMotionY = int(motionVec.mvArray[motionIdx-1,:][0])
                lastMotionX = int(motionVec.mvArray[motionIdx-1,:][1])
                stepSize = max(np.abs(lastMotionX),  np.abs(lastMotionY))
                
                if (np.abs(lastMotionX) == stepSize and lastMotionY == 0) or  \
                    (np.abs(lastMotionY) == stepSize and lastMotionX == 0):  # last MV is on XY-axis
                    maxIndex = 5
                else: # last MV is not on XY-axis
                    maxIndex = 6
                    LDSP[5] = (lastMotionY,lastMotionX)
            
            LDSP[0:5] = [(-stepSize,0),(0,-stepSize),(0,0),(0,stepSize),(stepSize,0)]
            
            #==================
            #      Step 2     
            #==================
            #  search using Large Diamond Search Pattern
            cost = costs[2]
            point = 2
            
            if stepSize == 0:
                pass
            else:
                for LDSPIdx in range(maxIndex):
                    
                    # center point already calculated
                    if (LDSPIdx == 2):
                        continue
                    
                    refBlkY = int(y + LDSP[LDSPIdx][0])
                    refBlkX = int(x + LDSP[LDSPIdx][1])
                    
                    # outside image boundary
                    if( refBlkY < 0 or refBlkX < 0 or 
                        refBlkY + mbSize > h or  refBlkX + mbSize > w):
                        continue
                    
                    referenceBlock = referenceImg[refBlkY:refBlkY+mbSize,refBlkX:refBlkX+mbSize]
                    costs[LDSPIdx] = getSAD(currentBlock,referenceBlock)
                    
                    if (costs[LDSPIdx] < cost):
                        # get smallest MME index of LDSP
                        cost = costs[LDSPIdx];
                        point = LDSPIdx; 
            
            #==================
            #      Step 3     然后再一次用 小P去repeat直到找到小P的中间
            #==================
            # relocate center point
            newY = int(y + LDSP[point][0])
            newX = int(x + LDSP[point][1])
            
            costs[:] = np.inf
            costs[2] = cost
            doneFlag = 0
            while not doneFlag:
                cost = costs[2]
                point = 2
                
                for SSPIdx in range(6):
                    if SSPIdx == 2:
                        continue
                    
                    newRefBlkY = int(y + SSP[SSPIdx][0])
                    newRefBlkX = int(x + SSP[SSPIdx][1])
                    if (newRefBlkY < 0 or newRefBlkX < 0 or 
                        newRefBlkY + mbSize > h or newRefBlkX + mbSize > w):
                        continue
                    
                    if newRefBlkX < x-p or newRefBlkX > x+p or newRefBlkY < y-p or newRefBlkY > y+p:
                        continue
                    
                    referenceBlock = referenceImg[newRefBlkY:newRefBlkY+mbSize,newRefBlkX:newRefBlkX+mbSize]
                    costs[SSPIdx] = getSAD(currentBlock,referenceBlock)
                    
                    if (costs[SSPIdx] < cost):
                        # get smallest MME index of LDSP
                        cost = costs[SSPIdx];
                        point = SSPIdx; 
                
                if point == 2: 
                    # Point incurred at the current URP
                    doneFlag = 1 
                else: 
                    # else align center with SSP
                    newY = int(newY + SSP[point][0])
                    newX = int(newX + SSP[point][1])
                    costs[:] = np.inf
                    costs[2] = cost
                
            # End of step3
            costs[:] = np.inf
            
            motionVec.vy = newY - y
            motionVec.vx = newX - x
            motionVec.mag2 = motionVec.vy**2 + motionVec.vx**2
            maxMag2 = max(maxMag2, motionVec.mag2);
            motionInfo = np.array([motionVec.vy , motionVec.vx , motionVec.mag2])
            motionVec.mvArray[motionIdx,:] = motionInfo

            referenceMVBlock = referenceImg[y+motionVec.vy:y+mbSize+motionVec.vy,x+motionVec.vx:x+mbSize+motionVec.vx]
            movedImg[y:y+mbSize,x:x+mbSize] = referenceMVBlock
            diffImg[y:y+mbSize,x:x+mbSize] = (currentBlock - referenceMVBlock) + 128
            
            
    return movedImg

def ARPS_search_frame(referenceImg,previousImg,mbSize,matchThresh,debug=True):
    """
    Adaptive Rood Pattern Search Algorithm
        checking similarity of close blocks, if similar, MV = 0;
        else search neighborhood by large pattern
        and small pattern repeatly
    
    return moved previous image (for our task)
    """
    assert referenceImg.shape == previousImg.shape
    h,w = previousImg.shape
    
    p = 6  # search boundary for current x y
    maxMag2 = 0 
    maxIndex = -1
    stepSize = 0
    local = 0
    neighbor = 0
    
    costs = np.array([np.inf]*6)
    LDSP = np.zeros([6,2]) # The index points for Large Diamond Search pattern (LDSP)
    SSP = np.array([[-1,0],[0, -1],[0, 0],[0, 1],[1, 0],[1, 1]]) 
    diffImg = np.zeros([h,w])
    movedPreviousImg = np.zeros([h,w])
    motionVec = MotionVectorClass(h,w,mbSize)
    
    hSegments, wSegments = segmentImage(referenceImg, mbSize)

    for y in range(0, int(hSegments*mbSize), mbSize):
        for x in range(0, int(wSegments*mbSize), mbSize):
            motionIdx = y//mbSize * wSegments + x//mbSize
            
            previousBlock = previousImg[y:y+mbSize,x:x+mbSize]
            referenceBlock = referenceImg[y:y+mbSize,x:x+mbSize]
            assert previousBlock.shape==referenceBlock.shape
            #==================
            #      Step 1    
            #==================
            costs[2] = getSAD(previousBlock,referenceBlock)
            # print('>>> getSAD  ',costs[2])
            # if similar to previous block, MV set 0
            if debug:
                # to finetune the matchThresh
                print('>>> costs[2]  ',costs[2])
                print('>>> matchThresh  ',matchThresh)
                
                assert 1==2
                
            if(costs[2] < matchThresh):  
                local += 1
                motionVec.vy = 0; motionVec.vx = 0; motionVec.mag2 = 0; 
                motionInfo = np.array([motionVec.vy , motionVec.vx , motionVec.mag2])
                motionVec.mvArray[motionIdx,:] = motionInfo
                diffImg[y:y+mbSize,x:x+mbSize] = 128
                movedPreviousImg[y:y+mbSize,x:x+mbSize] = previousBlock
                continue
            
            neighbor+=1
            # if left side of image 
            if x==0:
                stepSize = 2
                maxIndex = 5
            else:
                lastMotionY = int(motionVec.mvArray[motionIdx-1,:][0])
                lastMotionX = int(motionVec.mvArray[motionIdx-1,:][1])
                stepSize = max(np.abs(lastMotionX),  np.abs(lastMotionY))
                
                if (np.abs(lastMotionX) == stepSize and lastMotionY == 0) or  \
                    (np.abs(lastMotionY) == stepSize and lastMotionX == 0):  # last MV is on XY-axis
                    maxIndex = 5
                else: # last MV is not on XY-axis
                    maxIndex = 6
                    LDSP[5] = (lastMotionY,lastMotionX)
            
            LDSP[0:5] = [(-stepSize,0),(0,-stepSize),(0,0),(0,stepSize),(stepSize,0)]
            
            #==================
            #      Step 2     
            #==================
            #  search using Large Diamond Search Pattern
            cost = costs[2]
            point = 2
            
            if stepSize == 0:
                pass
            else:
                for LDSPIdx in range(maxIndex):
                    
                    # center point already calculated
                    if (LDSPIdx == 2):
                        continue
                    
                    prevBlkY = int(y + LDSP[LDSPIdx][0])
                    prevBlkX = int(x + LDSP[LDSPIdx][1])
                    
                    # outside image boundary
                    if( prevBlkY < 0 or prevBlkX < 0 or 
                        prevBlkY + mbSize > h or  prevBlkX + mbSize > w):
                        continue
                    
                    previousBlock = previousImg[prevBlkY:prevBlkY+mbSize,prevBlkX:prevBlkX+mbSize]
                    costs[LDSPIdx] = getSAD(previousBlock,referenceBlock)
                    
                    if (costs[LDSPIdx] < cost):
                        # get smallest MME index of LDSP
                        cost = costs[LDSPIdx];
                        point = LDSPIdx; 
            
            #==================
            #      Step 3     然后再一次用 小P去repeat直到找到小P的中间
            #==================
            # relocate center point
            newY = int(y + LDSP[point][0])
            newX = int(x + LDSP[point][1])
            
            costs[:] = np.inf
            costs[2] = cost
            doneFlag = 0
            while not doneFlag:
                cost = costs[2]
                point = 2
                
                for SSPIdx in range(6):
                    if SSPIdx == 2:
                        continue
                    
                    newPrevBlkY = int(y + SSP[SSPIdx][0])
                    newPrevBlkX = int(x + SSP[SSPIdx][1])
                    if (newPrevBlkY < 0 or newPrevBlkX < 0 or 
                        newPrevBlkY + mbSize > h or newPrevBlkX + mbSize > w):
                        continue
                    
                    if newPrevBlkX < x-p or newPrevBlkX > x+p or newPrevBlkY < y-p or newPrevBlkY > y+p:
                        continue
                    
                    previousBlock = previousImg[newPrevBlkY:newPrevBlkY+mbSize,newPrevBlkX:newPrevBlkX+mbSize]
                    costs[SSPIdx] = getSAD(previousBlock,referenceBlock)
                    
                    if (costs[SSPIdx] < cost):
                        # get smallest MME index of LDSP
                        cost = costs[SSPIdx];
                        point = SSPIdx; 
                
                if point == 2: 
                    # Point incurred at the previous URP
                    doneFlag = 1 
                else: 
                    # else align center with SSP
                    newY = int(newY + SSP[point][0])
                    newX = int(newX + SSP[point][1])
                    costs[:] = np.inf
                    costs[2] = cost
                
            # End of step3
            costs[:] = np.inf
            
            motionVec.vy = newY - y
            motionVec.vx = newX - x
            motionVec.mag2 = motionVec.vy**2 + motionVec.vx**2
            maxMag2 = max(maxMag2, motionVec.mag2);
            motionInfo = np.array([motionVec.vy , motionVec.vx , motionVec.mag2])
            motionVec.mvArray[motionIdx,:] = motionInfo
            # print('>>> motionVec.mvArray  ',motionVec.mvArray)
            
            movedPreviousBlock = previousImg[y+motionVec.vy:y+mbSize+motionVec.vy,x+motionVec.vx:x+mbSize+motionVec.vx]
            movedPreviousImg[y:y+mbSize,x:x+mbSize] = movedPreviousBlock
            diffImg[y:y+mbSize,x:x+mbSize] = abs(referenceBlock - movedPreviousBlock) + 128 
    return movedPreviousImg, local / hSegments*wSegments , neighbor / hSegments*wSegments

# ========= downsampling =========

def bayerDownsampling1(inputRAW,DOWN_FACTOR):
    """
    Downsampling raw image using average pooling
    """
    height, width = inputRAW.shape
    downsampledRAW = np.empty([height//DOWN_FACTOR, width//DOWN_FACTOR])
    
    # loop top-left bayer pattern 
    for y in range(height)[::DOWN_FACTOR]:
        for x in range(width)[::DOWN_FACTOR]:
            meanValue = (inputRAW[y,x] + inputRAW[y+1,x] + inputRAW[y,x+1] + inputRAW[y+1,x+1])//4
            downsampledRAW[y//DOWN_FACTOR,x//DOWN_FACTOR]  = meanValue
    
    return downsampledRAW

def bayerDownsampling2(inputRAW,DOWN_FACTOR):
    from skimage.measure import block_reduce
    return block_reduce(inputRAW,(2,2),np.mean)

def bayerDownsampling3(inputRAW,DOWN_FACTOR):
    H, W = inputRAW.shape
    HD = H // DOWN_FACTOR
    WD = W // DOWN_FACTOR
    return inputRAW.reshape(HD, DOWN_FACTOR, WD, DOWN_FACTOR).mean(axis=(1, 3))

# 1 timer 0.6538233080063947
# 2 timer 0.23859468504088
# 3 timer 0.017385435989126563
bayerDownsampling = bayerDownsampling3    

# ============== Distance
from scipy import fftpack
# implement 2D DCT
def dct2(a):
    return fftpack.dct(fftpack.dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return fftpack.idct(fftpack.idct(a.T, norm='ortho').T, norm='ortho')   

def getSpatialL1Back(group,length):
    return np.array([ np.mean(np.abs(group[i] - group[-1])) for i in range(length-1) ])

def getDCTL1Back(group,length):
    return np.array([ np.mean(np.abs(dct2(group[i]) - dct2(group[-1]))) for i in range(length-1) ])

def calculateWeightBack(x,weightCoef):
    return np.array((*[ weightCoef/(i+weightCoef) for i in x],1))
    


def getDCTL1Front(group,length):
    return np.array([ np.mean(np.abs(dct2(group[i]) - dct2(group[0]))) for i in range(1,length) ])
def getSpatialL1Front(group,length):
    return np.array([ np.mean(np.abs(group[i] - group[0])) for i in range(1,length) ])
def getDiffSpatialL1Front(group,length):
    return np.array([ np.mean(group[i]) for i in range(length-1) ])
def calculateWeightFront(x,weightCoef):
    return np.array((1,*[ weightCoef/(i+weightCoef) for i in x]))

# ============= IO =============

def writerVideo(imagePath,height,width):
    from vidgear.gears import WriteGear
    # define suitable tweak parameters for writer
    mode = "half_size" # use_camera_wb use_auto_wb  
    outputPath = "/data/gravitychen/exp_data/rawpy_rgb_use_auto_wb_" + mode
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    # output_params = {"-fourcc": "MJPG", "-fps": 15}
    writer1 = WriteGear(output_filename= outputPath + '/writer1.mp4',logging=True) # ,**output_params
    writer2 = WriteGear(output_filename= outputPath + '/writer2.mp4',logging=True) # ,**output_params
    writer3 = WriteGear(output_filename= outputPath + '/writer3.mp4',logging=True) # ,**output_params
    writer4 = WriteGear(output_filename= outputPath + '/writer4.mp4',logging=True) # ,**output_params
    
    for path in imagePath:
        frame = ARWRawpyISP(path,use_auto_wb=True,half_size=True) # no_auto_bright=True, bright=10 half_size=True
        writer1.write(frame[:height//2, :width//2],rgb_mode=True) # don't use rgb_mode 
        writer2.write(frame[:height//2, width//2:],rgb_mode=True) # 
        writer3.write(frame[height//2:, :width//2],rgb_mode=True) # 
        writer4.write(frame[height//2:, width//2:],rgb_mode=True) # 

    writer1.close()
    writer2.close()
    writer3.close()
    writer4.close()


def displayVideo(rawPathList):
    # opencv display video                                                                                          
    for path in rawPathList:
        # Display the resulting frame
        cv2.imshow('Frame',ARWRawpyISP(path,use_camera_wb=True)) # [:height//2, :width//2]
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


