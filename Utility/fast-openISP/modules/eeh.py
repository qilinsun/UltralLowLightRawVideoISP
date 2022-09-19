# File: eeh.py
# Description: Edge Enhancement
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from .helpers import generic_filter, gen_gaussian_kernel


@register_dependent_modules('csc')
class EEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        #sigma 小 没有边缘 0.2
        # kernel_size=3, sigma=2.2 ，星星点点 看得到
        kernel = gen_gaussian_kernel(kernel_size=5, sigma=1.2)
        self.gaussian = (1024 * kernel / kernel.max()).astype(np.int32)  # x1024

        t1, t2 = self.params.flat_threshold, self.params.edge_threshold # 4 8 
        threshold_delta = np.clip(t2 - t1, 1E-6, None) # 4
        # 384 *8 / 4 == 384 * 2 == 768
        self.middle_slope = np.array(self.params.edge_gain * t2 / threshold_delta, dtype=np.int32)  # x256
        # -1 * 384 * 8 * 4 / 4 = -3072
        self.middle_intercept = -np.array(self.params.edge_gain * t1 * t2 / threshold_delta, dtype=np.int32)  # x256
        # 384
        self.edge_gain = np.array(self.params.edge_gain, dtype=np.int32)  # x256

    def execute(self, data):
        """increasing the image contrast in the area immediately around the edge!"""
        y_image = data['y_image'].astype(np.int32)

        delta = y_image - generic_filter(y_image, self.gaussian)
        sign_map = np.sign(delta)
        # positive number means edge detected
        abs_delta = np.abs(delta) 
        # 8 bits == 256
        # 384  * abs_delta - 3072 
        # delta 在一半的地方放大？
        middle_delta = np.right_shift(self.middle_slope * abs_delta + self.middle_intercept, 8) 
        edge_delta = np.right_shift(self.edge_gain * abs_delta, 8)
        # 所以这个就是 edge map 的 look up table
        enhanced_delta = (
                # 4 < edge intersity < 8 * middle_delta +
                # edge intersity > 8 * edge_delta
                (abs_delta > self.params.flat_threshold) * (abs_delta <= self.params.edge_threshold) * middle_delta +
                (abs_delta > self.params.edge_threshold) * edge_delta
        )

        enhanced_delta = sign_map * np.clip(enhanced_delta, 0, self.params.delta_threshold)
        eeh_y_image = np.clip(y_image + enhanced_delta, 16, 235)

        data['y_eeh_image'] = eeh_y_image.astype(np.uint8)
        data['y_image'] = eeh_y_image.astype(np.uint8)
        data['edge_map'] = delta
