import typing

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import scipy as sc
import rawpy as rp


def complexMultipleConjugate(complex1: np.ndarray, complex2: np.ndarray) -> np.ndarray:
    re1 = complex1[:, :, 0]
    im1 = complex1[:, :, 1]
    re2 = complex2[:, :, 0]
    im2 = -complex2[:, :, 1]
    re = re1 * re2 - im1 * im2
    im = re1 * im2 + re2 * im1
    return np.array([re, im])


def complexModulus(complex1: np.ndarray) -> np.ndarray:
    re = complex1[:, :, 0]
    im = complex1[:, :, 1]
    return np.sqrt(re**2 + im**2)


def velocityFieldPhase(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 10)
    Y = np.arange(kSize, width - 3 * kSize, 10)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    window = cv2.createHanningWindow([2 * kSize + 1, 2 * kSize + 1], cv2.CV_32F)
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            # padding = cv2.copyMakeBorder(kernel, kSize, kSize, kSize, kSize, cv2.BORDER_DEFAULT)
            subImage = movedImage[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            translation, response = cv2.phaseCorrelate(kernel, subImage, window)
            y, x = translation
            if response >= 0 and (abs(y) <= kSize and abs(x) <= kSize):
                velocity[i, j, 0] = x
                velocity[i, j, 1] = y
            else:
                velocity[i, j, 0] = -100
                velocity[i, j, 1] = -100
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def velocityFieldPhaseV_2(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    optimalHeight = cv2.getOptimalDFTSize(height)
    optimalWidth = cv2.getOptimalDFTSize(width)
    paddedMovedImage = cv2.copyMakeBorder(movedImage, 0, optimalHeight - height, 0, optimalWidth - width,
                                          cv2.BORDER_DEFAULT)
    paddedReferenceImage = cv2.copyMakeBorder(reference, 0, optimalHeight - height, 0, optimalWidth - width,
                                              cv2.BORDER_DEFAULT)
    movedFFT = cv2.dft(paddedMovedImage, flags=cv2.DFT_COMPLEX_OUTPUT)
    referenceFFT = cv2.dft(paddedReferenceImage, flags=cv2.DFT_COMPLEX_OUTPUT)
    X = np.arange(kSize, height - 3 * kSize, 10)
    Y = np.arange(kSize, width - 3 * kSize, 10)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    for i in range(0, X.size):
        startX = X[i]
        for j in range(0, Y.size):
            startY = Y[i]
            Ga = referenceFFT[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1, :]
            Gb = movedFFT[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1, :]
            GaGb_ = complexMultipleConjugate(Ga, Gb)
            GaGb_abs = complexModulus(GaGb_)
            R = GaGb_ / GaGb_abs
            r = cv2.idft(R, flags=cv2.DFT_REAL_OUTPUT)


def velocityFieldBM4D(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 10)
    Y = np.arange(kSize, width - 3 * kSize, 10)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            subImage = movedImage[startX - kSize:startX + 3 * kSize + 1, startY - kSize:startY + 3 * kSize + 1]
            v, d = blockMatching(kernel, subImage, kSize)
            if d:
                y, x = v
                velocity[i, j, 0] = x
                velocity[i, j, 1] = y
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def blockMatching(kernel: np.ndarray, searchField: np.ndarray, kSize: int) -> tuple:
    height, width = searchField.shape
    minDistance = 1e4
    for i in range(0, height - 2 * kSize):
        for j in range(0, width - 2 * kSize):
            distance = np.linalg.norm(kernel - searchField[i:i + 2 * kSize + 1, j:j + 2 * kSize + 1])
            if distance < minDistance:
                minDistance = distance
                v = [i - kSize, j - kSize]
    return v, minDistance


data = pd.read_csv('gsalesman_sig10.csv', header=None)
kSize = 25
data = data.values.reshape((288, 50, 352)).astype(np.float32)
velocity = velocityFieldPhase(data[:, 43, :], data[:, 42, :], kSize)
validVelocity = velocity[velocity[:, :, 0] != -100, :]
invalidVelocity = velocity[velocity[:, :, 0] == -100, :]
plt.imshow(data[:, 43, :], cmap='gray')
plt.show()
plt.imshow(data[:, 42, :], cmap='gray')
plt.quiver(validVelocity[:, 3] + kSize, validVelocity[:, 2] + kSize, validVelocity[:, 1], validVelocity[:, 0],
           angles='xy', scale=1, color='yellow', units='xy')
plt.quiver(invalidVelocity[:, 3] + kSize, invalidVelocity[:, 2] + kSize, invalidVelocity[:, 1] * 0,
           invalidVelocity[:, 0] * 0,
           angles='xy', scale=1, color='red', units='xy')
plt.show()
