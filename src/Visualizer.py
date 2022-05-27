import torch
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torchvision.transforms as transforms
from torch.nn import Conv2d

from config import config


def read_image(image_filename: str) -> torch.Tensor:
    image_path = config.res_folder + image_filename
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    return transform(image)


# tensor is initially [rgb, height, width]
# matplotlib wants [height, width, rgb]
# so we transform it
def tensor_to_matplotlib(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(0, 2).transpose(0, 1)


class Visualizer(object):

    def __init__(self, image_filename):
        self.image_tensor = read_image(image_filename)

        self.fig = plt.figure()
        self.fig.set_size_inches(20, 20)

        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)

    def show(self, convolution: Conv2d):
        self.ax1.imshow(tensor_to_matplotlib(self.image_tensor), interpolation='none')

        convoluted_image_tensor = convolution(self.image_tensor).detach()

        self.ax2.imshow(tensor_to_matplotlib(convoluted_image_tensor), interpolation='none')

        self.fig.tight_layout()
        plt.show()
