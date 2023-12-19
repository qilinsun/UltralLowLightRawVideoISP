Date: 19/12/2023
# Progress
1. Proposed a pipeline of image alignment and denoising
2. Verified block-matching method based on phase correlation with the video from Prof. Bihan Wen
3. Researched noise modeling and calibration, as well as the k-sigma transform to regularize the distribution of pixels

## The motion between adjacent frames
|MovedImage|ReferenceImage|
|:----:|:-------:|
|![the 30th frame](/Docs/Image_results/moved_frame_30.png)|![the 29th frame](/Docs/Image_results/reference_frame_29.png)|


## The motion during a relatively long period
|MovedImage|ReferenceImage|
|:-----:|:------:|
|![the 45th_frame](/Docs/Image_results/moved_frame_45.png)|![the 40th_frame](/Docs/Image_results/reference_frame_40.png)|

## The pipeline of block-matching with phase correlation
![the pipeline of block-matching](/Docs/Image_results/blockMatching.svg)


 # todo
 1. Capture raw data ourselves and perform noise calibration accordingly
 2. Test the block-matching method with the raw data
 3. Research sparse coding methods and dictionary learning for denoising and block merging
