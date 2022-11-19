#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-21 22:14
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : SideWindowFilter.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SideWindowFilter(nn.Module):

    def __init__(self, radius, iteration, filter='box'):
        super(SideWindowFilter, self).__init__()
        self.radius = radius
        self.iteration = iteration
        self.kernel_size = 2 * self.radius + 1
        self.filter = filter

    def forward(self, im):
        b, c, h, w = im.size()

        d = torch.zeros(b, 8, h, w, dtype=torch.float, device='cuda')
        res = im.clone()

        if self.filter.lower() == 'box':
            filter = torch.ones(1, 1, self.kernel_size, self.kernel_size, device='cuda')
            L, R, U, D = [filter.clone() for _ in range(4)]

            L[:, :, :, self.radius + 1:] = 0
            R[:, :, :, 0: self.radius] = 0
            U[:, :, self.radius + 1:, :] = 0
            D[:, :, 0: self.radius, :] = 0

            NW, NE, SW, SE = U.clone(), U.clone(), D.clone(), D.clone()

            L, R, U, D = L / ((self.radius + 1) * self.kernel_size), R / ((self.radius + 1) * self.kernel_size), \
                         U / ((self.radius + 1) * self.kernel_size), D / ((self.radius + 1) * self.kernel_size)

            NW[:, :, :, self.radius + 1:] = 0
            NE[:, :, :, 0: self.radius] = 0
            SW[:, :, :, self.radius + 1:] = 0
            SE[:, :, :, 0: self.radius] = 0

            NW, NE, SW, SE = NW / ((self.radius + 1) ** 2), NE / ((self.radius + 1) ** 2), \
                             SW / ((self.radius + 1) ** 2), SE / ((self.radius + 1) ** 2)

            # sum = self.kernel_size * self.kernel_size
            # sum_L, sum_R, sum_U, sum_D, sum_NW, sum_NE, sum_SW, sum_SE = \
            #     (self.radius + 1) * self.kernel_size, (self.radius + 1) * self.kernel_size, \
            #     (self.radius + 1) * self.kernel_size, (self.radius + 1) * self.kernel_size, \
            #     (self.radius + 1) ** 2, (self.radius + 1) ** 2, (self.radius + 1) ** 2, (self.radius + 1) ** 2

            # print('L:', L)
            # print('R:', R)
            # print('U:', U)
            # print('D:', D)
            # print('NW:', NW)
            # print('NE:', NE)
            # print('SW:', SW)
            # print('SE:', SE)

        for ch in range(c):
            im_ch = im[:, ch, ::].clone().view(b, 1, h, w)
            # print('im size in each channel:', im_ch.size())

            for i in range(self.iteration):
                # print('###', (F.conv2d(input=im_ch, weight=L, padding=(self.radius, self.radius)) / sum_L -
                # im_ch).size(), d[:, 0,::].size())
                d[:, 0, ::] = F.conv2d(input=im_ch, weight=L, padding=(self.radius, self.radius)) - im_ch
                d[:, 1, ::] = F.conv2d(input=im_ch, weight=R, padding=(self.radius, self.radius)) - im_ch
                d[:, 2, ::] = F.conv2d(input=im_ch, weight=U, padding=(self.radius, self.radius)) - im_ch
                d[:, 3, ::] = F.conv2d(input=im_ch, weight=D, padding=(self.radius, self.radius)) - im_ch
                d[:, 4, ::] = F.conv2d(input=im_ch, weight=NW, padding=(self.radius, self.radius)) - im_ch
                d[:, 5, ::] = F.conv2d(input=im_ch, weight=NE, padding=(self.radius, self.radius)) - im_ch
                d[:, 6, ::] = F.conv2d(input=im_ch, weight=SW, padding=(self.radius, self.radius)) - im_ch
                d[:, 7, ::] = F.conv2d(input=im_ch, weight=SE, padding=(self.radius, self.radius)) - im_ch

                d_abs = torch.abs(d)
                # print('im_ch', im_ch)
                # print('dm = ', d_abs.shape, d_abs)
                mask_min = torch.argmin(d_abs, dim=1, keepdim=True)
                # print('mask min = ', mask_min.shape, mask_min)
                dm = torch.gather(input=d, dim=1, index=mask_min)
                im_ch = dm + im_ch

            res[:, ch, ::] = im_ch
        return res