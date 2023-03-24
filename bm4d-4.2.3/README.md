# Python wrapper for BM4D denoising - from Tampere with love, again

Python wrapper for BM4D for stationary correlated noise (including white noise).

BM4D is an algorithm for attenuation of additive spatially correlated
stationary (aka colored) Gaussian noise for volumetric data.
This package provides a wrapper for the BM4D binaries for Python for the denoising of volumetric and volumetric multichannel data. For denoising of images/2-D multichannel data, see also the "bm3d" package.

This implementation is based on 
Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354, and
Y. Mäkinen, S. Marchesini, A. Foi, 2021,
"Ring Artifact and Poisson Noise Attenuation via Volumetric Multiscale Nonlocal Collaborative Filtering of Spatially Correlated Noise", submitted to Journal of Synchrotron Radiation.

The package contains the BM4D binaries compiled for:
- Windows (Win10, MinGW-32)
- Linux (CentOS 7, 64-bit)
- Mac OSX (El Capitan, 64-bit)

The package is available for non-commercial use only. For details, see LICENSE.

Basic usage:
```python
	y_hat = bm4d(z, sigma); # white noise: include noise std
	y_hat = bm4d(z, psd); # correlated noise: include noise PSD (size of z)
```

For usage examples, see the examples folder of the full source (bm4d-***.tar.gz) from https://pypi.org/project/bm4d/#files

Contact: Ymir Mäkinen <ymir.makinen@tuni.fi> 
