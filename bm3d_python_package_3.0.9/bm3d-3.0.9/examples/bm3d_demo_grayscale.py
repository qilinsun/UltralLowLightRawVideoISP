"""
Grayscale BM3D denoising demo file, based on
Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.
"""


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bm3d import bm3d, BM3DProfile
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
import rawpy
import cv2
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, estimate_sigma, denoise_nl_means

class AutoGammaCorrection: # img in HSV Space
    def __init__(self, img):
        self.img = img
    def execute(self):
        img_h = self.img.shape[0]
        img_w = self.img.shape[1]
        Y = self.img[:,:,2].copy()
        gc_img = np.zeros((img_h, img_w), np.float32)
        Y = Y.astype(np.float32)/255.  # 255. is better right?
        Yavg = np.exp(np.log(Y+1e-20).mean())
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                if Y[y, x]>0:
                    gc_img[y, x] = np.log(Y[y,x]/Yavg + 1)/np.log(1/Yavg+1)
                # if x==1500:
                #     print(x, Y[y, x], gc_img[y, x])
        return gc_img

def main():
    rawpyParam1 = {
        'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
        'half_size': False,
        'use_camera_wb': True,
        'use_auto_wb': False,
        'no_auto_bright': True,
        'output_color': rawpy.ColorSpace.sRGB,
        'output_bps': 16}
    # Experiment specifications
    imagename = 'cameraman256.png'
    frame1_path = '/media/cuhksz-aci-03/数据/CUHK_SZ/0.1/iso6400/02527.ARW'
    # Load noise-free image
    y_img = rawpy.imread(frame1_path)
    y = (y_img.raw_image_visible.astype(np.float32))
    y = y[740:2020, 2096:4256]
    # y = np.array(Image.open(imagename)) / 255
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'g3'
    # noise_var = 0.02  # Noise variance
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    # noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, y.shape)
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    # z = np.atleast_3d(y) + np.atleast_3d(noise)

    # Call BM3D With the default settings.
    for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
        # noise_var = estimate_sigma(np.sqrt(y[di::2, dj::2] + (3 / 8)))
        noise_var = np.sqrt(0.001 * np.max(np.sqrt(y[di::2, dj::2] + (3 / 8))) ** 2)
        profile = BM3DProfile()
        profile.gamma = 6
        y_est = bm3d(np.sqrt(y[di::2, dj::2]+(3/8)), noise_var, profile)
        y[di::2, dj::2] = y_est
        # To include refiltering:
     # y_est = bm3d(z, psd, 'refilter')

        # For other settings, use BM3DProfile.
        # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
        # profile.gamma = 6;  # redefine value of gamma parameter
        # y_est = bm3d(z, psd, profile);

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm3d(z, sqrt(noise_var));

    # psnr = get_psnr(y, y_est)
    # print("PSNR:", psnr)

    # PSNR ignoring 16-pixel wide borders (as used in the paper), due to refiltering potentially leaving artifacts
    # on the pixels near the boundary of the image when noise is not circulant
    # psnr_cropped = get_cropped_psnr(y, y_est, [16, 16])
    # print("PSNR cropped:", psnr_cropped)



    # Ignore values outside range for display (or plt gives an error for multichannel input)
    # y_est = np.minimum(np.maximum(y_est, 0), 1)
    # z_rang = np.minimum(np.maximum(z, 0), 1)

    y_img.raw_image_visible[740:2020, 2096:4256] = (y**2-(3/8))
    y_post = y_img.postprocess(**rawpyParam1)
    y_post = (y_post[740:2020, 2096:4256]/65535*255.).astype(np.float32)
    hsv = cv2.cvtColor(y_post, cv2.COLOR_RGB2HSV)
    hsvOperator = AutoGammaCorrection(hsv)
    enhanceV = hsvOperator.execute()
    hsv[..., -1] = enhanceV * 255.
    enhanceRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    y_post = cv2.cvtColor(enhanceRGB, cv2.COLOR_RGB2BGR)
    isp_image_nor = cv2.normalize(y_post ** (1 / 2.2), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    path = '/home/cuhksz-aci-03/Desktop/handheld/output/bm3d' + '.png'
    cv2.imwrite(path, (isp_image_nor))
    # plt.title("y, z, y_est")
    # plt.imshow(y_post)
    # plt.show()


if __name__ == '__main__':
    main()

