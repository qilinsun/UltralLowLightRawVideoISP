import copy

import numpy as np
import cv2
import canon_utils as cu
import matplotlib.pyplot as plt
import blockMatching as bm


# sigma estimation
def sigmaEstimation(frameNum: int):
    path = '/mnt/data/submillilux_videos/submillilux_videos_dng/seq8/'
    frameBlock = 0
    for i in range(frameNum):
        name = path + str(i) + '.dng'
        frame = cu.read_16bit_raw(name)
        if not isinstance(frameBlock, np.ndarray):
            frameBlock = frame[:400, :400].reshape([400**2, 1])
        else:
            frameBlock = np.hstack([frameBlock, frame[:400, :400].reshape([400**2, 1])])
    mean = np.mean(frameBlock, axis=0, keepdims=True)
    std = np.std(frameBlock - mean)
    # print(frameBlock.shape)
    return std


def NNM(groups: list, lam: float):
    denoised = []
    for group in groups:
        U, S, V = np.linalg.svd(group, full_matrices=False)
        S = np.where(S >= lam, S - lam, 0)
        S = np.diag(S)
        X = U.dot(S).dot(V)
        denoised.append(X)
    return denoised


def WNNM(img_set: np.ndarray, sigma, n_iter: int):
    ksize = 64
    stride = 32
    height, width = img_set.shape[1], img_set.shape[2]
    XX = np.arange(ksize, height - 2 * ksize, stride)
    YY = np.arange(ksize, width - 2 * ksize, stride)
    IMG = img_set
    centralIndex = int((img_set.shape[0] - 1) // 2)
    frameIndex = [i for i in range(centralIndex, img_set.shape[0])]
    frameIndex += [i for i in range(centralIndex-1, -1, -1)]
    for k in range(n_iter):
        groups, v = bm.velocityFieldPhaseV_4(IMG, 64)
        tmp = []
        img_set0 = np.zeros(img_set.shape)
        # WNNM
        for i in range(len(groups)):
            Y = groups[i]
            U, S, V = np.linalg.svd(Y, full_matrices=False)
            sigmaX_hat = np.where(S ** 2 >= S.shape[0] * sigma ** 2, np.sqrt(S ** 2 - S.shape[0] * sigma ** 2), 0)
            omega = 2.8 * np.sqrt(S.shape[0]) / (sigmaX_hat + 1e-16)
            S = np.where(S >= omega, S - omega, 0)
            S = np.diag(S)
            X = U.dot(S).dot(V)
            tmp.append(X)
        # Aggregate
        for i in range(XX.size):
            x = XX[i]
            for j in range(YY.size):
                y = YY[j]
                index = YY.size * i + j
                for frame in frameIndex:
                    newBlock = tmp[index][:, frame].reshape(ksize, ksize)
                    x0 = (x + v[frame, i, j, 0]).astype(int)
                    y0 = (y + v[frame, i, j, 1]).astype(int)
                    # left top corner
                    if x0 == 0 and y0 == 0:
                        img_set0[frame, x0:x0 + 64, y0:y0 + 64] = newBlock
                    # top row
                    elif x0 == 0:
                        img_set0[frame, x0:x0 + 64, y0:y0 + 64] = (newBlock + img_set0[frame, x0:x0 + 64, y0:y0 + 64]) / 2
                        img_set0[frame, x0:x0 + 64, y0 + 32:y0 + 64] = newBlock[:, 32:64]
                    # most left column
                    elif y0 == 0:
                        img_set0[frame, x0:x0 + 64, y0:y0 + 64] = (newBlock + img_set0[frame, x0:x0 + 64, y0:y0 + 64]) / 2
                        img_set0[frame, x0:x0 + 32, y0:y0 + 64] = newBlock[:32, :]
                    # common cases
                    else:
                        if img_set0[frame, x0:x0 + 64, y0:y0 + 64].shape != newBlock.shape:
                            print(1)
                        img_set0[frame, x0:x0 + 64, y0:y0 + 64] = (newBlock + img_set0[frame, x0:x0 + 64, y0:y0 + 64]) / 2
                        img_set0[frame, x0 + 32:x0 + 64, y0 + 32:y0 + 64] = newBlock[32:, 32:]
        IMG = img_set0 + 0.1 * (img_set - IMG)
    return IMG
