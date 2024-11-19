# Pipeline

We read both images and event voxels as a burst. The central voxel and image are considered as the reference, and other voxels and images will be aligned to them.

## Event Branch
Both the current voxel and reference voxel will be fed into the feature extraction network, and we will conduct correlations with them to construct the correlation volume. We will also extract context information from 
the reference voxel. At last, the correlation volume and context volume will be concatenated to form the event feature volume.


## Image Alignment Branch
In this branch, we utilize the similarity constraints to supervise the alignment. Firstly, the RAW frame corresponding to the current voxel will be encoded and combined with the event feature volume to synthesize 
 features of the reference frame. The synthetic features are assumed to be close to the real features, therefore similarity loss can be derived and represented by the L2 norm currently. 

## Image Denoising
We perform alignment among feature maps, intuitively, if they are aligned perfectly, each layer in a certain feature volume should be the same as the counterparts in others. Thus, we can perform denoising by stacking 
the corresponding layers together and filtering. 

## Image Decoder
We need to decode the filtered feature maps to obtain the denoised frame. In the current stage, our image decoder is very simple and consists of three convolution layers and three deconvolution layers. 

## Training Scheme
While training the network, we will first train the main network and leave the decoder alone. After convergence, the parameters of the encoder will be fixed, and we train the decoder module to restore images from encoded feature maps. The decoder training process can be self-supervised, and it can be performed on clean frames.


The reason for adopting a two-stage training scheme is that frames captured in low-light environments contain considerable noise, degrading image quality significantly. With encoding, valid features lying in noisy images will be extracted, while noise will be suppressed. 

