# File: fcs.py
# Description: False Color Suppression
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np
from matplotlib import pyplot as plt

from .basic_module import BasicModule, register_dependent_modules


@register_dependent_modules(('csc', 'eeh'))
class FCS(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        # 32 - 8 = 24
        threshold_delta = np.clip(self.params.delta_max - self.params.delta_min, 1E-6, None)
        
        # 65536(UV gain 1) - 0(UV gain 0)/(max edge - min edge) == -2730
        self.slope = -np.array(65536 / threshold_delta, dtype=np.int32)  # x65536

    def execute(self, data):
        cbcr_image = data['cbcr_image'].astype(np.int32)
        edge_map = data['edge_map']

        uv_gain_map = self.slope * (np.abs(edge_map) - self.params.delta_max)
        uv_gain_map = np.clip(uv_gain_map, 0, 65536)
        fcs_cbcr_image = np.right_shift(uv_gain_map[..., None] * (cbcr_image - 128), 16) + 128
        fcs_cbcr_image = np.clip(fcs_cbcr_image, 16, 240)
        
        data['cbcr_image'] = fcs_cbcr_image.astype(np.uint8)


"""
        # plt_w = 2
        # plt_h = 3
        # plt.rcParams['axes.labelsize'] = 24
        # plt.rcParams['axes.titlesize'] = 24
        # plt.figure(figsize=(plt_w*20,plt_h*10),dpi=100)
        # plt.subplot(plt_h,plt_w,1);
        # plt.title('cbcr_image')
        # plt.imshow(cbcr_image[..., 0],cmap='gray',vmin=16,vmax=240)
        # plt.subplot(plt_h,plt_w,2);
        # plt.title('cbcr_image')
        # plt.imshow(cbcr_image[..., 1],cmap='gray',vmin=16,vmax=240)
        # plt.subplot(plt_h,plt_w,3);
        # plt.title('fcs_cbcr_image')
        # plt.imshow(fcs_cbcr_image[..., 0],cmap='gray',vmin=16,vmax=240)
        # plt.subplot(plt_h,plt_w,4);
        # plt.title('fcs_cbcr_image')
        # plt.imshow(fcs_cbcr_image[..., 1],cmap='gray',vmin=16,vmax=240)
        # plt.subplot(plt_h,plt_w,5);
        # plt.title('fcs_cbcr_image')
        # plt.imshow(np.abs(fcs_cbcr_image[..., 0] - cbcr_image[..., 0]),cmap='gray')
        # plt.subplot(plt_h,plt_w,6);
        # plt.title('fcs_cbcr_image')
        # plt.imshow(np.abs(fcs_cbcr_image[..., 1] - cbcr_image[..., 1]),cmap='gray')
        # plt.show()
        
"""