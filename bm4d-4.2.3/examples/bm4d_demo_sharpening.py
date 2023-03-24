"""
BM4D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.
"""


import numpy as np
import matplotlib.pyplot as plt
from bm4d import bm4d, BM4DProfile, BM4DStages, BM4DProfileBM3D
from experiment_funcs import generate_noise, get_experiment_kernel, get_psnr, get_cropped_psnr_3d
from scipy.io import loadmat

def main():
    # Experiment specifications

    # The multichannel example data is acquired from: http://www.bic.mni.mcgill.ca/brainweb/
    # C.A. Cocosco, V. Kollokian, R.K.-S. Kwan, A.C. Evans,
    #  "BrainWeb: Online Interface to a 3D MRI Simulated Brain Database"
    # NeuroImage, vol.5, no.4, part 2/4, S425, 1997
    # -- Proceedings of 3rd International Conference on Functional Mapping of the Human Brain, Copenhagen, May 1997.
    data_name = 'brain_slices.mat'
    table_name = 'brain_slices'

    # Load noise-free volume
    y = loadmat(data_name)[table_name]
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'g0'
    noise_var = 0.02 * np.max(y) ** 2 # Noise variance
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    kernel = get_experiment_kernel(noise_type, noise_var)
    noise, psd, kernel = generate_noise(kernel, seed, y.shape)
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    z = np.atleast_3d(y) + np.atleast_3d(noise)

    # Call BM4D HT With the default settings.
    y_est = bm4d(z, psd, stage_arg=BM4DStages.HARD_THRESHOLDING)

    # Call BM4D HT With sharpening
    profile = BM4DProfile()
    profile.set_sharpen(1.2)
    y_est_s = bm4d(z, psd, profile, stage_arg=BM4DStages.HARD_THRESHOLDING)
    # Alternatively, set the parameters separately
    # profile.sharpen_alpha = 1.5
    # profile.sharpen_alpha_3d = 1.1

    # For 2-D filtering
    # profile = BM4DProfileBM3D()
    # profile.set_sharpen(1.2)
    # noise, psd, kernel = generate_noise(kernel, seed, (y.shape[0], y.shape[1], 1))
    # y_est_s = bm4d(z[:, :, 24], psd, profile, stage_arg=BM4DStages.HARD_THRESHOLDING)

    plt.title("y, z, y_est, y_est_sh")
    i = 5

    disp_mat = np.concatenate((y[:, :, i], np.squeeze(z[:, :, i]),
                               y_est[:, :, i], y_est_s[:, :, i]), axis=1)
    plt.imshow(np.squeeze(disp_mat))
    plt.show()


if __name__ == '__main__':
    main()
