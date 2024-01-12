import numpy
import numpy as np
import cv2


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
    return np.sqrt(re ** 2 + im ** 2)


def velocityFieldPhase(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 15)
    Y = np.arange(kSize, width - 3 * kSize, 15)
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
            if response >= 0 and (abs(y) <= 2 * kSize and abs(x) <= 2 * kSize):
                velocity[i, j, 0] = x
                velocity[i, j, 1] = y
            else:
                velocity[i, j, 0] = -100
                velocity[i, j, 1] = -100
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def velocityFieldPhaseV_2(movedImage: np.ndarray, reference: np.ndarray, kSize: int, stride=32) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 2 * kSize, stride)
    Y = np.arange(kSize, width - 2 * kSize, stride)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    window = cv2.createHanningWindow([2 * kSize + 1, 2 * kSize + 1], cv2.CV_32F)
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            # padding = cv2.copyMakeBorder(kernel, kSize, kSize, kSize, kSize, cv2.BORDER_DEFAULT)
            bias = [[-8, -8], [-8, 0], [-8, 8], [0, -8], [0, 8], [8, -8], [8, 8], [8, 0], [0, 0]]
            max_response = 0
            # print(np.std(kernel))
            for k in range(9):
                subImage = movedImage[startX + bias[k][0]:startX + bias[k][0] + 2 * kSize + 1, startY + bias[k][1]:startY + bias[k][1] + 2 * kSize + 1]
                translation, response = cv2.phaseCorrelate(kernel, subImage, window)
                y, x = translation
                if response >= max_response and (abs(y) <= 2 * kSize and abs(x) <= 2 * kSize):
                    velocity[i, j, 0] = x
                    velocity[i, j, 1] = y
                    max_response = response
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def velocityFieldPhaseV_3(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    optimalHeight = cv2.getOptimalDFTSize(height)
    optimalWidth = cv2.getOptimalDFTSize(width)
    paddedMovedImage = cv2.copyMakeBorder(movedImage, 0, optimalHeight - height, 0, optimalWidth - width,
                                          cv2.BORDER_DEFAULT)
    paddedReferenceImage = cv2.copyMakeBorder(reference, 0, optimalHeight - height, 0, optimalWidth - width,
                                              cv2.BORDER_DEFAULT)
    windows = cv2.createHanningWindow(paddedReferenceImage.shape, cv2.CV_32F)
    movedFFT = np.fft.fft2(paddedMovedImage)
    referenceFFT = np.fft.fft2(paddedReferenceImage)
    X = np.arange(kSize, height - 3 * kSize, 25)
    Y = np.arange(kSize, width - 3 * kSize, 25)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    for i in range(0, X.size):
        startX = X[i]
        for j in range(0, Y.size):
            startY = Y[j]
            Ga = referenceFFT[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            bias = [[-10, -10], [-10, 0], [-10, 10], [0, -10], [0, 10], [10, -10], [10, 0], [10, 10], [0, 0]]
            max_response = 0
            for k in range(9):
                Gb = movedFFT[startX + bias[k][0]:startX + bias[k][0] + 2 * kSize + 1,
                     startY + bias[k][1]:startY + bias[k][1] + 2 * kSize + 1]
                GaGb_ = Ga * np.conjugate(Gb)
                GaGb_abs = np.absolute(GaGb_)
                R = GaGb_ / GaGb_abs
                r = np.fft.ifft2(R)
                res = np.fft.fftshift(r)
                res = np.real(res)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                y, x = max_loc
                if max_val >= max_response and (abs(y) <= 2 * kSize and abs(x) <= 2 * kSize):
                    velocity[i, j, 0] = x - kSize
                    velocity[i, j, 1] = y - kSize
                    max_response = max_val
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def blockMatchingVideo(kernel: np.ndarray, frame: np.ndarray, groups: np.ndarray, v: list, x, y):
    group0 = groups
    height, width = kernel.shape
    h, w = int((height-1)/4), int((width-1)/4)
    bias = [[-8, -8], [-8, 0], [-8, 8], [0, -8], [0, 8], [8, -8], [8, 8], [8, 0], [0, 0]]
    max_response = 0
    vx_tmp = 0
    vy_tmp = 0
    search_tmp = []
    window = cv2.createHanningWindow([height, width], cv2.CV_32F)
    for k in range(len(bias)):
        x1 = int(x + bias[k][0] + v[0])
        y1 = int(y + bias[k][1] + v[1])
        searchRegion = frame[x1:x1 + height, y1:y1 + width]
        translation, response = cv2.phaseCorrelate(kernel.astype(np.float32), searchRegion.astype(np.float32), window)
        ty, tx = translation
        if response >= max_response and (abs(y) <= width and abs(x) <= height):
            max_response = response
            vx_tmp = tx
            vy_tmp = ty
            search_tmp = searchRegion
    if isinstance(search_tmp, list):
        x1 = int(x + v[0])
        y1 = int(y + v[1])
        search_tmp = frame[x1:x1+height, y1:y1+width]
        v0 = v
    else:
        v0 = [vx_tmp, vy_tmp]
    group0 = np.hstack([group0, search_tmp[0:2*h, 0:w*2].reshape(4*h*w, 1)])
    return group0, v0


def blockMatchingVideoV_2(kernel: np.ndarray, frame: np.ndarray, v: np.ndarray, x, y):
    height, width = kernel.shape
    H, W = frame.shape
    h, w = int((height-1)/4), int((width-1)/4)
    bias = [[-8, -8], [-8, 0], [-8, 8], [0, -8], [0, 8], [8, -8], [8, 8], [8, 0], [0, 0]]
    # bias = [[0, 0]]
    max_response = 0
    vx_tmp = 0
    vy_tmp = 0
    search_tmp = []
    window = cv2.createHanningWindow([height, width], cv2.CV_32F)
    for k in range(len(bias)):
        x1 = int(x + bias[k][0] + v[0])
        y1 = int(y + bias[k][1] + v[1])
        if x1 + height > H:
            x1 = 1279 - height
        if y1 + width > W:
            y1 = 2159 - width
        searchRegion = frame[x1:x1 + height, y1:y1 + width]
        translation, response = cv2.phaseCorrelate(kernel.astype(np.float32), searchRegion.astype(np.float32), window)
        ty, tx = translation
        if response >= max_response and (abs(ty) <= width and abs(tx) <= height):
            max_response = response
            vx_tmp = tx
            vy_tmp = ty
            search_tmp = searchRegion
    if isinstance(search_tmp, list):
        v0 = np.array([-100, -100], dtype=object)
    else:
        v0 = np.array([vx_tmp, vy_tmp], dtype=object)
    return v0


def velocityFieldPhaseV_4(img_set: np.ndarray, ksize: int, stride=32) -> tuple:
    frame_num = img_set.shape[0]
    centerIndex = int(img_set.shape[0] // 2)
    centralFrame = img_set[centerIndex, ...]
    groups_all = []
    s = int(ksize/2)
    height, width = img_set.shape[1], img_set.shape[2]
    X = np.arange(ksize, height - 2 * ksize, stride)
    Y = np.arange(ksize, width - 2 * ksize, stride)
    v = np.zeros([frame_num, X.size, Y.size, 2])
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            velocity = [0, 0]
            kernel = centralFrame[startX:startX + 2 * ksize + 1, startY:startY + 2 * ksize + 1]
            groups = kernel[0:s*2, 0:s*2].reshape(ksize**2, 1)
            for frame in range(centerIndex+1, frame_num):
                groups, velocity = blockMatchingVideo(kernel, img_set[frame, :, :], groups, velocity, startX, startY)
                v[frame, i, j, 0] = velocity[0] + v[frame-1, i, j, 0]
                v[frame, i, j, 1] = velocity[1] + v[frame-1, i, j, 1]
            for frame in range(centerIndex-1, -1, -1):
                groups, velocity = blockMatchingVideo(kernel, img_set[frame, :, :], groups, velocity, startX, startY)
                v[frame, i, j, 0] = velocity[0] + v[frame+1, i, j, 0]
                v[frame, i, j, 1] = velocity[1] + v[frame+1, i, j, 1]
    groups_all.append(groups)
    return groups_all, v


def velocityFieldPhaseV_5(img_set: np.ndarray, ksize: int, stride=32) -> tuple:
    frame_num = img_set.shape[0]
    centralIndex = int(img_set.shape[0] // 2)
    centralFrame = img_set[centralIndex, ...]
    groups_all = []
    s = int(ksize/2)
    height, width = img_set.shape[1], img_set.shape[2]
    X = np.arange(ksize, height - 2 * ksize, stride)
    Y = np.arange(ksize, width - 2 * ksize, stride)
    v = np.zeros([frame_num, X.size, Y.size, 4])
    frameIndex = [i for i in range(centralIndex, img_set.shape[0])]
    frameIndex += [i for i in range(centralIndex - 1, -1, -1)]
    for frame in frameIndex[1:]:
        for i in range(X.size):
            startX = X[i]
            for j in range(Y.size):
                startY = Y[j]
                kernel = centralFrame[startX:startX + ksize * 2 + 1, startY:startY + ksize * 2 + 1]
                if frame > centralIndex:
                    v0 = blockMatchingVideoV_2(kernel, img_set[frame, ...], v[frame-1, i, j, :], startX, startY)
                    v[frame, i, j, 0] = v[frame-1, i, j, 0] + v0[0]
                    v[frame, i, j, 1] = v[frame-1, i, j, 1] + v0[1]
                else:
                    v0 = blockMatchingVideoV_2(kernel, img_set[frame, ...], v[frame+1, i, j, :], startX, startY)
                    v[frame, i, j, 0] = v[frame+1, i, j, 0] + v0[0]
                    v[frame, i, j, 1] = v[frame+1, i, j, 1] + v0[1]
        v[frame, ...] = interpolate(v[frame, ...])
        v[frame, :, :, 0] = cv2.medianBlur(v[frame, :, :, 0].astype(np.float32), 3)
        v[frame, :, :, 1] = cv2.medianBlur(v[frame, :, :, 1].astype(np.float32), 3)
    for i in range(X.size):
        for j in range(Y.size):
            groups = np.zeros([frame_num, ksize**2, 1])
            for frame in frameIndex:
                x = int(X[i] + v[frame, i, j, 0])
                y = int(Y[j] + v[frame, i, j, 1])
                block = img_set[frame, x:x+ksize, y:y+ksize]
                groups[frame, ...] = block.reshape(ksize**2, 1)
            groups_all.append(groups)
    return groups_all, v


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


def blockMatching(kernel: np.ndarray, searchField: np.ndarray, kSize: int, mode='l2') -> tuple:
    height, width = searchField.shape
    minDistance = 1e15
    for i in range(0, height - 2 * kSize):
        for j in range(0, width - 2 * kSize):
            if mode == 'l2':
                distance = np.linalg.norm(kernel - searchField[i:i + 2 * kSize + 1, j:j + 2 * kSize + 1], 2)
            elif mode == 'l1':
                distance = np.linalg.norm(kernel - searchField[i:i + 2 * kSize + 1, j:j + 2 * kSize + 1], 1)
            if distance < minDistance:
                minDistance = distance
                v = [i - kSize, j - kSize]
    return v, minDistance


def blockMatchingV_2(kernel: np.ndarray, searchField: np.ndarray, kSize=32):
    height, width = searchField.shape
    distances = []
    coordinates = [[], []]
    for i in range(0, height-kSize, 2):
        for j in range(0, width-kSize, 2):
            distance = np.linalg.norm(kernel - searchField[i:i+kSize, j:j+kSize], ord=2)
            distances.append(distance)
            coordinates[0].append(i)
            coordinates[1].append(j)
    distances = np.array(distances, dtype=object)
    coordinates = np.array(coordinates, dtype=object)
    indices = np.argsort(distances)
    sortedCoordinates = coordinates[:, indices]
    group = kernel.reshape(kSize**2, 1)
    for i in range(0, 10):
        x = sortedCoordinates[0, i]
        y = sortedCoordinates[1, i]
        group = np.hstack([group, searchField[x:x+kSize, y:y+kSize].reshape(kSize**2, 1)])
    return group, sortedCoordinates[:, :11]


def interpolate(velocityField: np.ndarray) -> np.ndarray:
    height, width = velocityField.shape[0], velocityField.shape[1]
    for i in range(height):
        for j in range(width):
            if velocityField[i, j, 0] == -100 and velocityField[i, j, 1] == -100:
                # upper left corner
                if i == 0 and j == 0:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i + 1, j, 0:2] + velocityField[i, j + 1, 0:2])
                # upper right corner
                elif i == 0 and j == width - 1:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i + 1, j, 0:2] + velocityField[i, j - 1, 0:2])
                # lower right corner
                elif i == height - 1 and j == width - 1:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i - 1, j, 0:2] + velocityField[i, j - 1, 0:2])
                # lower left corner
                elif i == height - 1 and j == 0:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i - 1, j, 0:2] + velocityField[i, j + 1, 0:2])
                # the top row
                elif i == 0 and j != 0 and j != width - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (
                            velocityField[i + 1, j, 0:2] + velocityField[i, j - 1, 0:2] + velocityField[i, j + 1,
                                                                                          0:2])
                # the bottom row
                elif i == height - 1 and j != 0 and j != width - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (
                            velocityField[i - 1, j, 0:2] + velocityField[i, j - 1, 0:2] + velocityField[i, j + 1,
                                                                                          0:2])
                # the first column
                elif j == 0 and i != 0 and i != height - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (
                            velocityField[i - 1, j, 0:2] + velocityField[i + 1, j, 0:2] + velocityField[i, j + 1,
                                                                                          0:2])
                # the most right column
                elif j == width - 1 and i != 0 and i != height - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (
                            velocityField[i - 1, j, 0:2] + velocityField[i + 1, j, 0:2] + velocityField[i, j - 1,
                                                                                          0:2])
                # common cases
                else:
                    velocityField[i, j, 0:2] = 1 / 4 * (
                            velocityField[i - 1, j, 0:2] + velocityField[i + 1, j, 0:2] + velocityField[i, j + 1,
                                                                                          0:2] + velocityField[i,
                                                                                                 j - 1, 0:2])
    return velocityField


def velocityFieldCorrelate(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 15)
    Y = np.arange(kSize, width - 3 * kSize, 15)
    [YY, XX] = np.meshgrid(Y, X)
    velocityField = np.zeros((X.size, Y.size, 4))
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            subImage = movedImage[startX - kSize:startX + 3 * kSize + 1, startY - kSize:startY + 3 * kSize + 1]
            res = cv2.matchTemplate(subImage, kernel, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            y, x = max_loc
            if max_val >= 0.2:
                velocityField[i, j, 0] = x - kSize
                velocityField[i, j, 1] = y - kSize
            else:
                velocityField[i, j, 0] = -100
                velocityField[i, j, 1] = -100
    velocityField[..., 2] = XX
    velocityField[..., 3] = YY
    return velocityField


def patchGroup(velocity: np.ndarray, moved: np.ndarray, reference: np.ndarray, kSize):
    validVelocity = velocity[velocity[:, :, 0] != -100, :]
    X = validVelocity[:, 2].astype(np.int32)
    Y = validVelocity[:, 3].astype(np.int32)
    X1 = (validVelocity[:, 2] + validVelocity[:, 0]).astype(np.int32)
    Y1 = (validVelocity[:, 3] + validVelocity[:, 1]).astype(np.int32)
    groups = []
    coordinates = []
    for i in range(validVelocity.shape[0]):
        bias = [[-16, -16], [-16, 0], [-16, 16], [0, -16], [0, 16], [16, -16], [16, 16], [16, 0],
                [-8, -8], [-8, 0], [-8, 8], [0, -8], [0, 8], [8, -8], [8, 8], [8, 0]]
        kernel = reference[X[i]:X[i] + 2 * kSize, Y[i]:Y[i] + 2 * kSize]
        group = moved[X1[i]:X1[i] + 2 * kSize, Y1[i]:Y1[i] + 2 * kSize].reshape(4*kSize**2, 1)
        _ = np.array([[X[i], Y[i]]]).reshape(2, 1)
        coordinate = np.array(_, dtype=object)
        window = cv2.createHanningWindow([2 * kSize, 2 * kSize], cv2.CV_32F)
        for j in range(len(bias)):
            block = moved[X1[i]+bias[j][0]:X1[i]+bias[j][0] + 2 * kSize, Y1[i]+bias[j][1]:Y1[i] + 2 * kSize + bias[j][1]]
            _, val = cv2.phaseCorrelate(block.astype(np.float32), kernel.astype(np.float32), window)
            if val >= 0.3:
                block = block.reshape(4*kSize**2, 1)
                group = np.hstack([group, block])
                __ = np.array([X1[i]+bias[j][0], Y1[i]+bias[j][1]]).reshape(2, 1)
                coordinate = np.hstack([coordinate, __])
        if len(groups) == 0:
            groups.append(group)
            coordinates.append(coordinate)
        else:
            max_correlation = 0
            max_idx = -1
            for k in range(len(groups)):
                correlate = np.corrcoef(groups[k][:, 0], group[:, 0])[0, 1]
                if correlate > max_correlation and correlate >= 0.5:
                    max_correlation = correlate
                    max_idx = k
            if max_idx >= 0:
                groups[max_idx] = np.hstack([groups[max_idx], group])
                coordinates[max_idx] = np.hstack([coordinates[max_idx], coordinate])
            else:
                groups.append(group)
                coordinates.append(coordinate)
    return groups, coordinates

