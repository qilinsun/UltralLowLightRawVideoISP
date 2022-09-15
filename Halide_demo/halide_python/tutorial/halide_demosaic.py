import rawpy
import imageio
import matplotlib.pylab as plt
import numpy as np
import os
import sys
sys.path.append("../build/src")
import halide as hl # https://halide-lang.org/docs/namespace_halide_1_1_boundary_conditions.html


image_path = "images/5k.dng"
with rawpy.imread(image_path) as raw:
    rawimg = raw.raw_image_visible.copy()
    print('>>> rawimg.shape  ',rawimg.shape)
    # 3821 2 -- 12bits
    white_balance = raw.camera_whitebalance
    white_balance_r = white_balance[0] / white_balance[1]
    white_balance_g0 = 1
    white_balance_g1 = 1
    white_balance_b = white_balance[2] / white_balance[1]
    cfa_pattern = raw.raw_pattern
    cfa_pattern = 3 # RGGB 1 'GRBG' 2 'BGGR' 3
    ccm = raw.color_matrix
    black_point = int(raw.black_level_per_channel[0])
    white_point = int(raw.white_level)
    ref_img = raw.postprocess(output_bps=16)


print('>>> np.max min(rawimg)  ',np.max(rawimg),np.min(rawimg))  
print('white balance', white_balance)
print('>>> cfa_pattern  ',cfa_pattern) # 0123 rgbg
print('>>> ccm  ',ccm)
print('>>> black_point  ',black_point)
print('>>> white_point  ',white_point)
print('>>> np.max min(ref_img)  ',np.max(ref_img),np.min(ref_img))  


print('Building image buffer...')
result = hl.Buffer(hl.UInt(16), [hl.Buffer(rawimg).width(), hl.Buffer(rawimg).height(), 1])

# plt.imshow(image,cmap="gray")
# plt.show()



#=============
# raw = rawpy.imread("images/5k.dng")
# # rgb = raw.postprocess() 
# im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
# rgb = np.float32(im / 65535.0*255.0)
# rgb = np.asarray(rgb,np.uint8)
# imageio.imsave("images/demo.jpg",rgb)
# 
# from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
# LIGHTHOUSE_IMAGE = colour.io.read_image(os.path.join('images', '5k.dng'))
# CFA = mosaicing_CFA_Bayer(LIGHTHOUSE_IMAGE)
# colour.plotting.plot_image(colour.cctf_encoding(demosaicing_CFA_Bayer_Menon2007(CFA)), text_kwargs={'text': 'Lighthouse - Demosaicing - Menon (2007)'});
#=============
