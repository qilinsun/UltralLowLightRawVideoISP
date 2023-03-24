"""
BM4D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.
"""


import numpy as np
import matplotlib.pyplot as plt
from bm4d import bm4d, BM4DProfile
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
    noise_type = 'g1'
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

    # Call BM4D With the default settings.
    y_est = bm4d(z, psd)

    # For 8x8x1 blocks instead of 4x4x4/5x5x5
    # y_est = bm4d(z, psd, '8x8')

    # To include refiltering:
    # y_est = bm4d(z, psd, 'refilter')

    # For other settings, use BM4DProfile.
    # profile = BM4DProfile(); # equivalent to profile = BM4DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm4d(z, psd, profile)

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm4d(z, sqrt(noise_var))

    print("PSNR: ", get_psnr(y_est, y))

    plt.title("y, z, y_est")
    i = 0

    disp_mat = np.concatenate((y[:, :, i], np.squeeze(z[:, :, i]),
                               y_est[:, :, i]), axis=1)
    plt.imshow(np.squeeze(disp_mat))
    plt.show()


if __name__ == '__main__':
    main()
