import os
import torchvision.transforms as transforms
from PIL import Image
from copy import deepcopy
import torch


def calculate_rgb_mean_variance(data_dir):
    # Image directories
    dirlist = os.listdir(data_dir)
    image_names = [f for f in dirlist if f.endswith(".png")]
    temp_paths = [os.path.join(data_dir, img_name) for img_name in image_names]
    image_number = [int(img_name.split('_')[1].split('.')[0]) for img_name in image_names]
    image_paths = [path for _, path in sorted(zip(image_number, temp_paths))]

    tf = transforms.ToTensor()
    image = tf(Image.open(image_paths[0]))
    rgbDim, height, width = image.shape
    assert rgbDim == 3

    # assert mean, variance
    mean = torch.Tensor([0., 0., 0.])
    std = torch.Tensor([0., 0., 0.])

    # calculate mean
    for color_idx in range(3):
        for image_idx in range(len(image_paths)):
            image = tf(Image.open(image_paths[image_idx]))  # dim: height * width * channel
            mean_imgwise = torch.mean(image[color_idx, :, :])
            # update mean and variance
            mean[color_idx] += mean_imgwise
        mean[color_idx] /= len(image_paths)

    # calculate standard variation
    for color_idx in range(3):
        for image_idx in range(len(image_paths)):
            image = tf(Image.open(image_paths[image_idx]))  # dim: height * width * channel
            # update mean and variance
            std[color_idx] += ((image[color_idx, :, :] - mean[color_idx]) ** 2).sum()  # std[color_idx] + (mean_prev[color_idx]) ** 2 - (mean[color_idx]) ** 2 + ()
        std[color_idx] = (std[color_idx] / (len(image_paths) * height * width)) ** (1 / 2)

    assert torch.any(mean < 1.0) and torch.any(0.0 < mean)
    assert torch.any(0.0 < std) and torch.any(std < 1.0)
    return mean, std


def rgb_data_normalizer(data, mean, variance):
    new_data = np.ones(data.shape)
    for color_idx in range(3):
        new_data[color_idx, :, :] = data[color_idx, :, :] - mean[color_idx]
        new_data[color_idx, :, :] = new_data[color_idx, :, :] / std[color_idx]
    return new_data


if __name__ == '__main__':
    data_dir = "data/panda"
    mean, std = calculate_rgb_mean_variance(data_dir)
    print(mean, std)
