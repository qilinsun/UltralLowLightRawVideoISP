import cv2
import torch
import numpy as np
import os
import struct
import h5py
import cv2 as cv
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray


def raw_visualization(raw_path, raw_length):
    count = 0
    raw_image = np.zeros((raw_length,), dtype=np.uint16)
    with open(raw_path, 'rb') as f:
        while count < raw_length:
            buffer = f.read(2)
            data = struct.unpack('<H', buffer)
            raw_image[int(count / 2)] = data[0]
            count += 2
    return raw_image


def event_visualization(event_path, event_length, offset=0):
    with open(event_path, 'rb') as f:
        if offset > 0:
            f.seek(offset)
        buffer = f.read(event_length)
        event_image = np.frombuffer(buffer, dtype=np.uint8)
        # event_image = np.frombuffer(buffer).astype(np.uint8)
    # event_image = np.fromfile(event_path, dtype=np.uint8)
    return event_image


def voxel_visualization(voxel: torch.Tensor, layer: int):
    voxel_frame = voxel[:, layer, :, :]
    voxel_frame = voxel_frame.view(1156, 1632).numpy()
    cv.imwrite('voxel_test.png', voxel_frame * 200)


class Voxel(object):
    def __init__(self, event_stream_path, bin_num, timestamp, width, height):
        self.bin_num = bin_num
        self.timestamp = timestamp
        self.event_stream_path = event_stream_path
        self.frame_num = len(timestamp)
        self.width, self.height = width, height
        self.frame_length = self.width * self.height

    def _read_event_stream(self):
        with open(self.event_stream_path, 'rb') as f:
            buffer = f.read(self.frame_num * self.frame_length)
            self.event_stream = np.frombuffer(buffer, dtype=np.uint8)
        self.event_stream = self.event_stream.reshape(self.frame_num, self.height, self.width)

    def _timestamp_normalize(self):
        start = min(self.timestamp)
        end = max(self.timestamp)
        duration = end - start
        timestamp = np.array(self.timestamp)
        self.normed_timestamp = (timestamp - start) / duration

    def voxelConstruction(self, polarity=True) -> torch.Tensor:
        self._read_event_stream()
        bin_length = self.frame_num / self.bin_num
        for i in range(self.bin_num):
            event_frame = self.event_stream[int(i * bin_length): int((i + 1) * bin_length), ...]
            positive_mask = (event_frame == 1).astype(np.uint8)
            positive_frame = event_frame * positive_mask
            positive_frame = np.sum(positive_frame, axis=0)
            negative_mask = (event_frame == 2).astype(np.uint8)
            negative_frame = event_frame * negative_mask
            negative_frame = np.sum(negative_frame, axis=0)
            if i == 0:
                voxel = np.stack((positive_frame, negative_frame))
            else:
                voxel_frame = np.stack((positive_frame, negative_frame))
                voxel = np.concatenate((voxel, voxel_frame), axis=0)
        voxel = voxel.astype(np.float32)
        voxel /= bin_length
        voxel = torch.from_numpy(voxel).float()
        voxel = voxel.view((1, -1, self.height, self.width))
        return voxel


def generatePath(event_folder: str, image_folder: str, output_path: str) -> None:
    f = open("{}/{}.txt".format(output_path, 'path_event'), 'w')
    paths = []
    # walk through event folder
    for dir_path, dir_names, file_names in os.walk(event_folder):
        path = []
        for file_name in file_names:
            if 'left' in dir_path:
                path.append(os.path.join(dir_path, file_name))
                path.sort()
        if path != []:
            f.write(path[0] + ' ' + path[1] + '\n')  #
    f.close()
    # walk through image folder
    g = open("{}/{}.txt".format(output_path, 'path_image'), 'w')
    path = []
    for dir_path, dir_names, file_names in os.walk(image_folder):
        for file_name in file_names:
            if 'left' in dir_path:
                if 'exposure_time' in file_name:
                    path.append(os.path.join(dir_path, file_name))
                    path.append(os.path.join(dir_path, 'rectified'))
        if len(path) == 2:
            g.write(path[0] + ' ' + path[1] + '\n')
            path = []  # timestamps, exposure, rectified (images)
    g.close()


def opticalFlowGenerate(image_folder: str, output_path: str, burst_length: int) -> None:
    image_dir = os.path.join(image_folder, 'rectified')
    image_paths = []
    for dir_path, dir_names, file_names in os.walk(image_dir):
        for file_name in file_names:
            image_paths.append(os.path.join(dir_path, file_name))
    image_paths.sort()
    with h5py.File(os.path.join(output_path, 'optical_flows.h5'), 'w') as f:
        for i, image_path in enumerate(image_paths):
            if i < int(burst_length / 2) or i > len(image_paths) - int(burst_length / 2):
                continue
            center_image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
            optical_flows_burst = []
            for j in range(i - int(burst_length / 2), i + int(burst_length / 2)):
                img = cv2.imread(image_paths[j], cv2.IMREAD_GRAYSCALE)
                flow = cv.calcOpticalFlowFarneback(prev=center_image, next=img, flow=None,
                                                   pyr_scale=0.5,
                                                   levels=3, winsize=15, iterations=5, poly_n=5, poly_sigma=1.2,
                                                   flags=0)
                optical_flows_burst.append(flow)
            f.create_dataset(str(i), data=np.array(optical_flows_burst))



if __name__ == '__main__':
    # width, height = 3264, 2312
    # raw_image = raw_visualization('raw_test.bin', width * height)
    # raw_image = raw_image.reshape(height, width)
    # cv.imwrite('test.png', raw_image)
    # event_image = event_visualization('event_stream_test.bin', int(width * height / 4), offset=1886592)
    # event_image = event_image.reshape(int(height / 2), int(width / 2))
    # cv.imwrite('event_test.png', event_image * 50)
    # voxel_grid = Voxel(event_stream_path='event_stream_test.bin', bin_num=5, timestamp=[0] * 100, width=1632,
    #                    height=1156)
    # voxel = voxel_grid.voxelConstruction()
    # voxel_visualization(voxel, 5)
    # generatePath('/mnt/DSEC/test_events', '/mnt/DSEC/test_images', '/mnt/DSEC')
    for root, dirs, files in os.walk('/mnt/DSEC/test_images/'):
        for dir in dirs:
           path = os.path.join(root, dir)
           path += '/images/left'
           print(path)
           opticalFlowGenerate(image_folder=path, output_path=path, burst_length=7)
        break
    # rectify = h5py.File('/mnt/DSEC/test_images/interlaken_00_a/images/left/optical_flows.h5', 'r')
    # optical_flows = rectify['optical_flows'][:]
    # rectify.close()
