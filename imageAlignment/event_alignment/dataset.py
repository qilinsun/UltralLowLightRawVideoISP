import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import numpy as np
from numpy.random import binomial
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.data import Dataset, DataLoader
from representations import VoxelGrid
import h5py
import cv2 as cv
import os


class LEOFDataset(Dataset):
    def __init__(self, event_path: str, image_path: str, time_stamp_path: str, rectify_path: str, burst_length: int,
                 drop: bool = False):
        """
        :param event_path: event file path
        :param image_path: image file path
        :param time_stamp_path: time stamp file path
        :param burst_length: the length of a burst, expected to be odd
        :param drop: whether to drop out the events randomly
        """
        self.image_path = image_path
        self.rectify_path = rectify_path
        self.events = h5py.File(event_path, 'r')
        self.event_x = self.events['events']['x'][:]
        self.event_y = self.events['events']['y'][:]
        self.event_p = self.events['events']['p'][:]
        self.event_t = self.events['events']['t'][:].astype(np.int64)
        event_t_offset = self.events['t_offset'][()]
        self.events.close()
        self.event_t += event_t_offset
        self.image_list = list(os.listdir(self.image_path))
        self.image_list.sort()
        self.burst_length = burst_length
        self._read_time_stamps(time_stamp_path)
        if drop:
            mask = self._random_drop(self.event_x.shape, p=0.5)
            mask = mask.astype(bool)
            self.event_x = self.event_x[mask]
            self.event_y = self.event_y[mask]
            self.event_p = self.event_p[mask]
            self.event_t = self.event_t[mask]

    def _rectify(self, x, y):
        rectify = h5py.File(self.rectify_path, 'r')
        self.rectify_map = rectify['rectify_map'][:]
        rectify.close()
        return self.rectify_map[y, x, 0], self.rectify_map[y, x, 1]

    def _read_time_stamps(self, time_stamp_path):
        if 'exposure' in time_stamp_path:
            exposure_time_stamps = np.loadtxt(time_stamp_path, delimiter=',', dtype=np.int64)
            self.time_stamps = np.mean(exposure_time_stamps, axis=1)
        else:
            self.time_stamps = np.loadtxt(time_stamp_path, dtype=np.int64)

    def _determine_interval(self) -> int:
        time_end = self.time_stamps[-1]
        time_start = self.time_stamps[0]
        assert time_end >= time_start
        ave_interval = (time_end - time_start) / (self.time_stamps.shape[0] - 1)
        return int(ave_interval / 2)

    @staticmethod
    def _random_drop(shape: tuple, p: float) -> np.ndarray:
        mask = np.random.binomial(1, 1 - p, shape)
        return mask

    def __len__(self):
        return len(self.image_list) - self.burst_length

    def __getitem__(self, index_start: int):
        reference_index = int(self.burst_length / 2)
        index = list(range(index_start, index_start + self.burst_length))
        image_set = []
        # flow_gt = []
        # event_voxel = []
        dt = self._determine_interval()
        for i in range(len(index)):
            image_path = os.path.join(self.image_path, self.image_list[index[i]])
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            image_set.append(image)

        for i in range(len(index)):
            #  ground truth optical flow
            flow = cv.calcOpticalFlowFarneback(prev=image_set[reference_index], next=image_set[i], flow=None,
                                               pyr_scale=0.5,
                                               levels=3, winsize=15, iterations=5, poly_n=5, poly_sigma=1.1,
                                               flags=0)
            flow = torch.from_numpy(flow)
            flow = flow.unsqueeze(dim=0)
            # flow_gt.append(flow)
            # voxel construction for events
            # if i < reference_index:
            #     time_range = [self.time_stamps[index[i]] - dt, self.time_stamps[index[reference_index]] + dt]
            # else:
            #     time_range = [self.time_stamps[index[reference_index]] - dt, self.time_stamps[index[i]] + dt]

            time_range = [self.time_stamps[index[i]] - dt, self.time_stamps[index[i]] + dt]
            time_mask = (time_range[0] <= self.event_t) * (self.event_t <= time_range[1])
            voxel_x = self.event_x[time_mask]
            voxel_y = self.event_y[time_mask]
            rectified_x, rectified_y = self._rectify(voxel_x, voxel_y)
            rectified_x = torch.from_numpy(rectified_x).float()
            rectified_y = torch.from_numpy(rectified_y).float()
            voxel_p = self.event_p[time_mask]
            voxel_p = torch.from_numpy(voxel_p).float()
            voxel_t = self.event_t[time_mask]
            voxel_t = torch.from_numpy(voxel_t.astype(np.int32)).int()
            voxel = VoxelGrid(channels=10, height=480, width=640)
            voxel_grid = voxel.convert(rectified_x, rectified_y, voxel_p, voxel_t)
            if i == 0:
                flow_gt = flow
                event_voxel = voxel_grid
            else:
                flow_gt = torch.cat((flow_gt, flow), dim=0)
                event_voxel = torch.cat((event_voxel, voxel_grid), dim=0)
            # event_voxel.append(voxel_grid)
        return event_voxel, flow_gt


if __name__ == '__main__':
    event_txt = '/mnt/DSEC/path_event.txt'
    image_txt = '/mnt/DSEC/path_image.txt'
    event_paths, image_paths, time_paths, rectified_paths = [], [], [], []
    f = open(event_txt, 'r')
    for line in f.readlines():
        event_path, rectified_path = line.split(sep=' ')
        event_paths.append(event_path)
        rectified_paths.append(rectified_path[:-1])
    f.close()
    f = open(image_txt, 'r')
    for line in f.readlines():
        time_path, image_path = line.split(sep=' ')
        time_paths.append(time_path)
        image_paths.append(image_path[:-1])
    f.close()
    for i in range(len(event_paths)):
        dataset = LEOFDataset(event_path=event_paths[i], image_path=image_paths[i], time_stamp_path=time_paths[i],
                              rectify_path=rectified_paths[i], burst_length=7, drop=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for voxel, flow_gt in dataloader:
            hsv = np.zeros([1080, 1440, 3])
            hsv[..., 1] = 255
            print(voxel[0].shape)
            # visualization
            voxel_array = voxel[0].numpy()
            for j in range(10):
                image = voxel_array[0, j, :, :]
                plt.imshow(image, cmap='gray')
                plt.show()
            flow = flow_gt[0].numpy()
            mag, ang = cv.cartToPolar(flow[0, :, :, 0], flow[0, :, :, 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            # hsv[..., 2] = (hsv[..., 2] * 2.55).astype(np.uint8)
            hsv = hsv.astype(np.float32)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
            plt.imshow(bgr, cmap='gray')
            plt.show()
            break
        break
