# File: gac.py
# Description: Gamma Correction
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np
from matplotlib import pyplot as plt

from .basic_module import BasicModule



def write_mask(mask,points,rectangle_size=20): 
    half_rectangle_size = rectangle_size // 2 
    # input point 
    y,x = mask.shape 
    point_y, point_x = points
    
    # if point_y-half_rectangle_size<0 or point_y+half_rectangle_size>=y:
    #     pass
    # if point_x-half_rectangle_size<0 or point_x+half_rectangle_size>=x:
    #     pass
    # else:
    #     # write black pixel around point 
    #     mask[point_y-half_rectangle_size:point_y+half_rectangle_size,point_x-half_rectangle_size:point_x+half_rectangle_size] = 0.
    for h in range(point_y-half_rectangle_size,point_y+half_rectangle_size): 
        for w in range(point_x-half_rectangle_size,point_x+half_rectangle_size): 
            if h<0 or h>=y: 
                continue 
            if w<0 or w>=x: 
                continue 
            mask[int(h),int(w)] = 1.
    return mask 

def fill_rectange_mask(h,w,num_patch_h,num_patch_w,rectangle_size): 
    mask = np.ones([h,w]) 
    for idx_y,point_y in enumerate(np.linspace(0,h,num_patch_h)): 
        for point_x in np.linspace(0,w,num_patch_w): 
            # pass center
            if point_y == 1280//2 and point_x== 2160//2: 
                continue 
            mask = write_mask(mask,(point_y.astype(np.int),point_x.astype(np.int)),rectangle_size) 

        if idx_y == 1 or idx_y == 3:
            mask[int(point_y)-10:int(point_y)+10,:] = 1.
    return mask.astype(np.float64) 


class GAC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.gain = np.array(self.params.gain,dtype=np.uint32)  # x256
        # self.gamma = np.array(self.params.gamma, dtype=np.float32)  # x1
        # original exp
        x = np.arange(self.cfg.saturation_values.hdr + 1)
        lut = ((x / self.cfg.saturation_values.hdr) ** self.params.gamma) * self.cfg.saturation_values.sdr
        self.lut = lut.astype(np.uint8)

    def execute(self, data):
        rgb_image = data['rgb_image'].astype(np.uint32)
        # original exp
        gac_rgb_image = np.right_shift(self.gain * rgb_image, 8)
        gac_rgb_image = np.clip(gac_rgb_image, 0, self.cfg.saturation_values.hdr)
        gac_rgb_image = self.lut[gac_rgb_image]
        
        # gac_rgb_image = self.gain * rgb_image
        # gac_rgb_image /= (self.gain * 255.)
        # gac_rgb_image = gac_rgb_image ** self.gamma
        # gac_rgb_image *= 255.
        
        data['gac'] = gac_rgb_image
        data['rgb_image'] = gac_rgb_image
