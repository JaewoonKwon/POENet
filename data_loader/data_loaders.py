from base import BaseDataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
from utils.LieGroup import *
import os
from PIL import Image
import torchvision.transforms as transforms


class AmbidexWristDataset(Dataset):
    def __init__(self, data_dir):
        rawdata = np.loadtxt(data_dir)
        self.y = expSE3(torch.Tensor(rawdata[:, 2:]))
        self.x = torch.Tensor(rawdata[:, :2])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def getX(self):
        return self.x


class AmbidexWristDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = AmbidexWristDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


####################### UNIFIED ##########################

class UnifiedWristDataset(Dataset):
    def __init__(self, data_dir, dyn_dir):
        rawdata = np.loadtxt(data_dir)
        self.y = expSE3(torch.Tensor(rawdata[:, 2:]))
        self.x = torch.Tensor(rawdata[:, :2])
        dyndata = np.loadtxt(dyn_dir)
        self.motorPos = torch.Tensor(dyndata[:, :2])
        self.motorVel = torch.Tensor(dyndata[:, 2:4])
        self.motorAcc = torch.Tensor(dyndata[:, 4:6])
        self.motorTorque = torch.Tensor(dyndata[:, 6:8])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.motorPos[idx], self.motorVel[idx], self.motorAcc[idx], self.motorTorque[idx]

    def getX(self):
        return self.x


class UnifiedWristDataLoader(BaseDataLoader):
    def __init__(self, data_dir, dyn_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = UnifiedWristDataset(data_dir, dyn_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


##################### Euler spiral #######################

class OneDimDataset(Dataset):
    def __init__(self, data_dir):
        rawdata = np.loadtxt(data_dir)
        self.y = expSE3(torch.Tensor(rawdata[:, 1:]))
        self.x = torch.Tensor(rawdata[:, :1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def getX(self):
        return self.x


class OneDimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = OneDimDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


##################### ImageToPose #######################

class ImageToPoseDataset(Dataset):
    def __init__(self, data_dir, data_mean=None, data_std=None):
        assert data_mean is not None
        assert data_std is not None
        self.mean = data_mean
        self.std = data_std
        # Image directories
        dirlist = os.listdir(data_dir)
        image_names = [f for f in dirlist if f.endswith(".png")]
        temp_paths = [os.path.join(data_dir, img_name) for img_name in image_names]
        image_number = [int(img_name.split('_')[1].split('.')[0]) for img_name in image_names]
        self.image_paths = [path for _, path in sorted(zip(image_number, temp_paths))]
        assert len(self.image_paths) == len(image_number)
        assert min(image_number) == 0  # because the numbers are row indices.
        assert max(image_number) == len(self.image_paths) - 1
        self.tf = transforms.ToTensor()
        # SE3 pose label
        kine_names = [f for f in dirlist if f.endswith("kine.txt")]
        rawdata = np.loadtxt(os.path.join(data_dir, kine_names[0]))
        self.y = expSE3(torch.Tensor(rawdata[:, -6:]))
        assert len(kine_names) == 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Normalize the image inputs
        images = self.tf(Image.open(self.image_paths[idx]))
        for color_idx in range(3):
            images[color_idx, :, :] -= self.mean[color_idx]
            images[color_idx, :, :] /= self.std[color_idx]
        return images, self.y[idx]

    def getX(self):
        allimages = torch.stack([self.tf(Image.open(path)) for path in self.image_paths])
        for color_idx in range(3):
            allimages[:, color_idx, :, :] -= self.mean[color_idx]
            allimages[:, color_idx, :, :] /= self.std[color_idx]
        return allimages


class ImageToPoseDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, data_mean=None, data_std=None):
        self.dataset = ImageToPoseDataset(data_dir, data_mean, data_std)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
