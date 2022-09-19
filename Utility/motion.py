from utils import ARPS_search_encoding,ARPS_search_R2V,getRandomImage,getRealImage
from timeit import default_timer as timer 
 
imageMode = "real"
mbSize = 120
matchThresh = 100 #* you can modify this

if imageMode == "real":
    refPath = "../Utility/fast-openISP/output/test1.png"
    curPath = "../Utility/fast-openISP/output/test2.png"
    # currentImg,referenceImg = getRealImage(refPath,curPath,cutting=True)
    previousImg,referenceImg = getRealImage(refPath,curPath,cutting=False)
elif imageMode == "random":
    currentImg,referenceImg = getRandomImage(height=h,width=w,\
                                downStep=3,leftStep=0,plot=False)

# movedReferenceImg = ARPS_search_encoding(referenceImg,currentImg,mbSize,matchThresh)

sum = 0
for i in range(2):
    start = timer() 
    movedPreviousImg = ARPS_search_R2V(referenceImg,previousImg,mbSize,matchThresh)
    end = timer() 
    sum += (end - start) 
print('>>> sum/2  ',sum/2)


# from matplotlib import pyplot as plt
# import numpy as np
# plt.subplot(2,3,4);
# plt.title('referenceImg')
# plt.imshow(referenceImg,cmap='gray')
# plt.subplot(2,3,2);
# plt.title('previousImg')
# plt.imshow(previousImg,cmap='gray')
# plt.subplot(2,3,3);
# plt.title('movedPreviousImg')
# plt.imshow(movedPreviousImg,cmap='gray')

# # show diff
# plt.subplot(2,3,5);
# plt.title('referenceImg - previousImg')
# plt.imshow(abs(referenceImg - previousImg),cmap='gray',vmax = 300)
# plt.subplot(2,3,6);
# plt.title('referenceImg - movedPreviousImg')
# plt.imshow(abs(referenceImg - movedPreviousImg),cmap='gray',vmax = 300)
# plt.show()


import cv2
cv2.imwrite("../Docs/Images/MotionEstimation/R2V_example/blk_size_120/referenceImg.png",referenceImg)
cv2.imwrite("../Docs/Images/MotionEstimation/R2V_example/blk_size_120/previousImg.png",previousImg)
cv2.imwrite("../Docs/Images/MotionEstimation/R2V_example/blk_size_120/movedPreviousImg.png",movedPreviousImg)
cv2.imwrite("../Docs/Images/MotionEstimation/R2V_example/blk_size_120/diff_ref_prev.png",abs(referenceImg - previousImg))
cv2.imwrite("../Docs/Images/MotionEstimation/R2V_example/blk_size_120/diff_ref_movedPrev.png",abs(referenceImg - movedPreviousImg))
