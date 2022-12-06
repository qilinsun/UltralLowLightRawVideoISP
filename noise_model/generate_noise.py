import numpy as np
from scipy.stats import poisson

def generate_noisy_raw(gt_raw, a, b, fpn):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson(np.maximum(gt_raw-512, 0) / a).rvs() * a
    poisson_fpn = poisson(fpn).rvs()
    gaussian_noise = np.sqrt(gaussian_noise_var) * np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + poisson_fpn + 512
    noisy_img = np.minimum(np.maximum(noisy_img, 0), 2 ** 14 - 1)

    return noisy_img
