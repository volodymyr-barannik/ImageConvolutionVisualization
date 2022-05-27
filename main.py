from torch.nn import Conv2d

from src.Visualizer import Visualizer

if __name__ == '__main__':
    v = Visualizer(image_filename='pathologic_polyhedron.jpg')

    # accepts rgb image (because we have 3 in_channels), outputs rgb image as well
    conv = Conv2d(in_channels=3, out_channels=3, kernel_size=15, stride=8, dilation=2)
    v.show(conv)
