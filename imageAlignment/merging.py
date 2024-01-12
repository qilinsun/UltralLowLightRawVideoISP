import numpy as np
import matplotlib.pyplot as plt
import cv2
import blockMatching as bm


def blockMerging(img_set: np.ndarray, sigma: np.float32) -> tuple:
    ksize = 64
    stride = 32
    groups, v = bm.velocityFieldPhaseV_5(img_set, ksize, stride)
    height, width = img_set.shape[1], img_set.shape[2]
    X = np.arange(ksize, height - 2 * ksize, stride)
    Y = np.arange(ksize, width - 2 * ksize, stride)
    centralIndex = int((img_set.shape[0] - 1) / 2)
    res = np.zeros([height, width]).astype(np.complex128)
    for j in range(len(groups)):
        group = groups[j]
        T_ = 0
        T0 = np.fft.fft2(group[centralIndex, :].reshape(ksize, ksize))
        for i in range(group.shape[0]):
            blocks = group[i, :].reshape(ksize, ksize)
            T = np.fft.fft2(blocks)
            D = np.abs(T - T0)
            A = D ** 2 / (D ** 2 + 8 * sigma ** 2)
            T_ += T + A * (T0 - T)
        newBlock = np.fft.ifft2(T_ / group.shape[0])
        x = int(j / Y.size)
        y = int(j % Y.size)
        if x == 0 and y == 0:
            res[X[x]:X[x] + 64, Y[y]:Y[y] + 64] = newBlock
        # top row
        elif x == 0:
            res[X[x]:X[x] + 64, Y[y]:Y[y] + 64] = (newBlock + res[X[x]:X[x] + 64, Y[y]:Y[y] + 64]) / 2
            res[X[x]:X[x] + 64, Y[y] + 32:Y[y] + 64] = newBlock[:, 32:64]
        # most left column
        elif y == 0:
            res[X[x]:X[x] + 64, Y[y]:Y[y] + 64] = (newBlock + res[X[x]:X[x] + 64, Y[y]:Y[y] + 64]) / 2
            res[X[x]:X[x] + 32, Y[y]:Y[y] + 64] = newBlock[:32, :]
        # common cases
        else:
            res[X[x]:X[x] + 64, Y[y]:Y[y] + 64] = (newBlock + res[X[x]:X[x] + 64, Y[y]:Y[y] + 64]) / 2
            res[X[x] + 32:X[x] + 64, Y[y] + 32:Y[y] + 64] = newBlock[32:, 32:]
    return res, v
