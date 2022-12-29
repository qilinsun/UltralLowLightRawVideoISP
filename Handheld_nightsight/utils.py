import os
import math
import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from vidgear.gears import WriteGear
# package-specific imports (Package named 'package.algorithm')
from genericUtils import getSigned, isTypeInt
from numba import vectorize, guvectorize, uint8, uint16, float32, float64
from matplotlib import pyplot as plt

@vectorize([uint16(uint16, uint16, uint16, uint16)], target='parallel')
def umean4_(a, b, c, d):
    return np.right_shift(a + b + c + d + 2, 2) #????????????????


@vectorize([float32(float32, float32, float32, float32), float64(float64, float64, float64, float64)], target='parallel')
def fmean4_(a, b, c, d):
    return (a + b + c + d) * 0.25

def downsample(image, kernel='gaussian', factor=2):
    '''Apply a convolution by a kernel if required, then downsample an image.
    Args:
        image: the input image (WARNING: single channel only!)
        kernel: None / str ('gaussian' / 'bayer') / 2d numpy array
        factor: downsampling factor
    '''
    # Special case
    if factor == 1:
        return image

    # Filter the image before downsampling it
    if kernel is None:
        filteredImage = image
    elif kernel == 'gaussian':
        # gaussian kernel std is proportional to downsampling factor
        filteredImage = gaussian_filter(image, sigma=factor * 0.5, order=0, output=None, mode='reflect')
        #*print('>>> image.shape  ',image.shape)
        #*print('>>> filteredImage.shape  ',filteredImage.shape)
        #*plt.subplot(1,2,1);
        #*plt.title('image')
        #*plt.imshow(image,cmap='gray',vmin=780,vmax=850)
        #*plt.subplot(1,2,2);
        #*plt.title('filteredImage')
        #*plt.imshow(filteredImage,cmap='gray',vmin=780,vmax=850)
        #*plt.show()
    elif kernel == 'bayer':
        # Bayer means that a simple 2x2 aggregation is required
        if isTypeInt(image):
            # starlight
            return umean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
            # # sony
            # return umean4_(image[2::4, 0::4], image[4::4, 0::4], image[2::4, 2::4], image[4::4, 2::4])
        else:
            # starlight
            return fmean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
            # # sony
            # return umean4_(image[2::4, 0::4], image[4::4, 0::4], image[2::4, 2::4], image[4::4, 2::4])
    else:
        # filter by convoluting with the input kernel
        filteredImage = signal.convolve2d(image, kernel, boundary='symm', mode='valid')

    # Shape of the downsampled image
    h2, w2 = np.floor(np.array(filteredImage.shape) / float(factor)).astype(np.int)

    # Extract the pixels
    if isTypeInt(image):
        return np.rint(filteredImage[:h2 * factor:factor, :w2 * factor:factor]).astype(image.dtype)
    else:
        return filteredImage[:h2 * factor:factor, :w2 * factor:factor]

def patchesRMS(patches):
    '''Computes the Root-Mean-Square of a set of patches/tiles.
    Args:
        patches: nD array (n >= 3) of 2D patches
    '''
    assert len(patches.shape) >= 3, 'not an nD array of patches'
    # flatten each patch
    patches = np.reshape(patches, tuple(patches.shape[:-2] + (patches.shape[-2] * patches.shape[-1],)))
    return np.sqrt(np.mean(np.multiply(patches, patches), axis=-1))

def gaussian_kernel(kernel_size=1024, sigma=15):
    kernel = np.zeros(shape=(kernel_size, kernel_size), dtype=np.float)
    radius = kernel_size // 2
    for y in range(-radius, radius):  # [-r, r]
        for x in range(-radius, radius):
            # 二维高斯函数
            v = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
            kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
    kernel2 = kernel / np.sum(kernel)
    return kernel2


def WriteVideoGear(images, outputPath, filename, gamma=True, normalize=True, fps=10):
    """Usage:
    WriteVideoGear(self.raw_video,outputPath="/FPN/",filename="xx.mp4",gamma=True,normalize=True,fps=10) # no

    Args:
        images (any loopable array): stack of images
        gamma (bool, optional): Defaults to True.
        normalize (bool, optional): Defaults to True.
        fps (int, optional): Defaults to 10.
        mask_mode (bool, optional): Defaults to True.
    """
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    # MP4
    output_params = {"-input_framerate": fps}  # "-r":30
    outputPath = outputPath + filename
    writer = WriteGear(output_filename=outputPath, compression_mode=True, logging=not True, **output_params)

    for frame in images:
        if gamma:
            frame = frame ** (1 / 2.2)
        if normalize:
            frame = cv2.normalize(frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        writer.write(frame)
    writer.close()