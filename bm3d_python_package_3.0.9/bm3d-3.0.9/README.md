# Python wrapper for BM3D denoising - from Tampere with love

Python wrapper for BM3D for stationary correlated noise (including white noise) for color,
grayscale and multichannel images and deblurring.

BM3D is an algorithm for attenuation of additive spatially correlated
stationary (aka colored) Gaussian noise. This package provides a wrapper
for the BM3D binaries for use for grayscale, color and other multichannel images
for denoising and deblurring.

This implementation is based on Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.

This package includes binaries which require an additional
installation of the OpenBLAS library (http://www.openblas.net/).
For Windows and Mac, a version of OpenBLAS is included in the binary.

The package contains the BM3D binaries compiled for:
- Windows (Win10, MinGW-32)
- Linux (Debian 10, 64-bit)
- Mac OSX (El Capitan, 64-bit)

The binaries are available for non-commercial use only. For details, see LICENSE.

For examples, see the examples folder of the full source zip, which also includes the example noise cases demonstrated in the paper.
Alternatively, you can download the examples from http://www.cs.tut.fi/~foi/GCF-BM3D/bm3d_py_demos.zip .

Authors: \
    Ymir Mäkinen <ymir.makinen@tuni.fi> \
    Lucio Azzari \
    Alessandro Foi



