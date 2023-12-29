Date: 19/12/2023
# Progress
1. Proposed a pipeline of image alignment and denoising
2. Verified block-matching method based on phase correlation with the video from Prof. Bihan Wen
3. Researched noise modeling and calibration, as well as the k-sigma transform to regularize the distribution of pixels

## The motion between adjacent frames
|MovedImage|ReferenceImage|
|:----:|:-------:|
|![the 30th frame](/Docs/Image_results/moved_frame_30.png)|![the 29th frame](/Docs/Image_results/reference_frame_29.png)|


## The motion during a relatively long period (Phase Correlate & L2-norm Matching)
|Moved Image|Reference Image (Phase Correlate)|Reference Image (L2-norm)|
|:-----:|:------:|:----:|
|![the 45th_frame](/Docs/Image_results/moved_frame_45.png)|![the 40th_frame](/Docs/Image_results/reference_frame_40.png)|![L2](/Docs/Image_results/results_1219/moved_frame_45_L2.png)|

## The pipeline of block-matching with phase correlation
![the pipeline of block-matching](/Docs/Image_results/blockMatching.svg)


 # Todo
 1. Capture raw data ourselves and perform noise calibration accordingly
 2. Test the block-matching method with the raw data
 3. Research sparse coding methods and dictionary learning for denoising and block merging



Date: 29/12/2023
# Progress
1. Verified gamma correction for block matching
2. Calibrated noise with a Sony camera
3. Researched sparse coding and low-rank denoising


## The motion detected between adjacent frames with and without gamma correction
|Raw Image|Corrected Image|Motion(corrected)|Motion(uncorrected)|
|:---:|:----:|:---:|:---:|


## The results of noise calibration



# Todo
1. Extend low-rank image denoising to video scenarios
2. Calibrate noise with mobile phones
3. Capture raw videos
