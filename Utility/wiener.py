import torch
import torch as nn
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F


def psf2otf(input_filter, output_size):
    '''Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    '''
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1
        # pad修改
        padded = F.pad(input_filter, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0), "constant")
    else:
        padded = input_filter

    padded = padded.permute(2, 0, 1, 3)
    padded = nn.fft.ifftshift(padded)
    padded = padded.permute(1, 2, 0, 3)

    ## Take FFT
    tmp = padded.permute(2, 3, 0, 1)
    imag = torch.DoubleTensor(np.zeros(tmp.shape))
    tmp = nn.fft.fft2(nn.complex(tmp, imag))
    return tmp.permute(2, 3, 0, 1)

def inverse_filter(blurred, estimate, psf, gamma=None, init_gamma=2.):
     """Inverse filtering in the frequency domain.

     Args:
         blurred: image with shape (batch_size, height, width, num_img_channels)
         estimate: image with shape (batch_size, height, width, num_img_channels)
         psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
         gamma: Optional. Scalar that determines regularization (higher --> more regularization, output is closer to
                "estimate", lower --> less regularization, output is closer to straight inverse filtered-result). If
                not passed, a trainable variable will be created.
         init_gamma: Optional. Scalar that determines the square root of the initial value of gamma.
     """
     img_shape = blurred.shape

     if gamma is None: # Gamma (the regularization parameter) is also a trainable parameter.
        # 创建变量和初始化的方式
        gamma = torch.tensor(init_gamma, dtype=nn.float32, requires_grad=True)
        gamma = nn.pow(gamma, 2) # Enforces positivity of gamma.
        # tf.summary.scalar('gamma', gamma)

     a_tensor_transp = blurred.permute(0,3,1,2)
     estimate_transp = estimate.permute(0,3,1,2)

     # Everything has shape (batch_size, num_channels, height, width)
     imag = torch.DoubleTensor(np.zeros(a_tensor_transp.shape))
     img_fft = nn.fft.fft2(nn.complex(a_tensor_transp, imag))
     otf = psf2otf(psf, output_size=img_shape[1:3])
     otf = otf.permute(2,3,0,1)

     adj_conv = img_fft * nn.conj(otf)

     # This is a slight modification to standard inverse filtering - gamma not only regularizes the inverse filtering,
     # but also trades off between the regularized inverse filter and the unfiltered estimate_transp.
     numerator = adj_conv + nn.fft.fft2(nn.complex(gamma*estimate_transp, imag))

     kernel_mags = nn.pow(nn.abs(otf), 2) # Magnitudes of the blur kernel.

     denominator = nn.complex(kernel_mags + gamma, imag)
     filtered = nn.div(numerator, denominator)
     cplx_result = nn.fft.ifft2(filtered)
     real_result = nn.real(cplx_result) # Discard complex parts.

     # Get back to (batch_size, num_channels, height, width)
     result = real_result.permute(0,2,3,1)
     return result

