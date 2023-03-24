"""
BM4D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.
"""


import numpy as np
import matplotlib.pyplot as plt
from bm4d import bm4d, bm4d_multichannel, BM4DProfile, BM4DStages
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

    # Weaker noise
    kernel = get_experiment_kernel(noise_type, noise_var / 4)
    noise2, psd2, kernel2 = generate_noise(kernel, seed, y.shape)

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    z = np.atleast_3d(y) + np.atleast_3d(noise)
    z2 = np.atleast_3d(y) + np.atleast_3d(noise2)

    y_est_z = bm4d(z, psd)
    y_est_z2 = bm4d(z2, psd2)

    # Call multi-channel BM4D for the two images, using the less noisy image as the first channel
    # Now, the block-matching will be performed on this channel, yielding less
    # noisy matches
    y_est2 = bm4d_multichannel([z2, z], [psd2, psd])

    # EQUIVALENT TO
    #z_arr = np.array([z2, z])
    #psd_arr = np.array([psd2, psd])
    #y_est2b = bm4d_multichannel(z_arr, psd_arr)

    # The PSNR of y_est_z2 and the first channel of y_est2 should be identical, as we are
    # denoising the same image with the same PSD and no further inputs.

    # However, PSNR of channel 2 of y_est2 is better than that of y_est_z,
    # because blockmatches were done on the less noisy channel 1
    print("PSNR (z): ", get_psnr(y_est_z, y))
    print("PSNR (z2): ", get_psnr(y_est_z2, y))

    print("PSNR multichannel 1 (=z2): ", get_psnr(y_est2[0], y))
    print("PSNR multichannel 2 (z with z2 blockmatches): ", get_psnr(y_est2[1], y))

    plt.title("y, z, y_est")
    i = 5

    disp_mat = np.concatenate((y[:, :, i], np.squeeze(z[:, :, i]),
                               y_est_z[:, :, i], y_est2[1][:, :, i]), axis=1)
    plt.imshow(np.squeeze(disp_mat))
    plt.show()


if __name__ == '__main__':
    main()
