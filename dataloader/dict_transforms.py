import torch

from parameters import *
from PIL import Image
import numpy as np

__all__ = ['DictNormalize', 'Dict2Tensor',
           'Resize', 'AlphaKill']


class DictNormalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.),
                 scale=255.0, gray=False):
        """
        Assumption
        image = [y, x, c] shape, RGB
        label = [y, x] shape, palette or grayscale
        """
        self.mean = mean
        self.std = std
        self.scale = scale
        self.gray_ = gray

    def __call__(self, input_dict : {str : Image.Image}):
        image = np.array(input_dict[tag_image]).astype(np.float32)
        label = np.array(input_dict[tag_label]).astype(np.float32)

        image /= self.scale
        image -= self.mean
        image /= self.std

        if self.gray_:
            label /= self.scale

        return {tag_image : image,
                tag_label : label}


class Dict2Tensor(object):
    def __init__(self, boundary_white=False, two_dim=False):
        pass
        """
        Assumption
        image = [y, x, c] shape, RGB
        label = [y, x] shape, palette or grayscale
        """
        self.boundary_white = boundary_white
        self.two_dim = two_dim

    def __call__(self, input_dict : {str : Image.Image}):
        # Height x Width x Channels -> Channels x Height x Width
        # divide dictionary
        image = np.array(input_dict[tag_image]).astype(np.float32)
        label = np.array(input_dict[tag_label]).astype(np.float32)

        # label 2D -> 3D
        if self.two_dim:
            label = np.expand_dims(label, axis=-1)

        # Boundary White -> Black
        if self.boundary_white:
            label[label == 255] = 0

        # transpose for torch.Tensor [H x W x C] -> [C x H x W]
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        torch_image = torch.from_numpy(image).float()
        torch_label = torch.from_numpy(label).float()

        return {tag_image : torch_image,
                tag_label : torch_label}


class Resize(object):
    def __init__(self, size:tuple, image_mode=Image.BILINEAR, label_mode=Image.BILINEAR):
        # (Width, Height) -> (Height, Width)
        #    X   ,   Y          Y   ,   X
        self.size = tuple(reversed(size))
        self.image_mode = image_mode
        self.label_mode = label_mode

    def __call__(self, input_dict : {str : Image.Image}):
        image = input_dict[tag_image]
        label = input_dict[tag_label]

        assert image.size == label.size, 'input and mask sizes must be equal.'

        image = image.resize(self.size, self.image_mode)
        label = label.resize(self.size, self.label_mode)

        return {tag_image : image,
                tag_label : label}

class AlphaKill(object):
    # kill alpha channel 4D -> 3D
    def __init__(self):
        pass

    def __call__(self, input_dict : {str : Image.Image}):
        image = input_dict[tag_image].convert('RGB')
        label = input_dict[tag_label]
        # label = input_dict[tag_label].convert('RGB')

        return {tag_image : image,
                tag_label : label}
