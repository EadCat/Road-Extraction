import torch

from parameters import *
from PIL import Image
import numpy as np

from typing import Union

__all__ = ['DictNormalize', 'Dict2Tensor',
           'DictResize', 'AlphaKill']


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.),
                 scale=255.0):
        """
        Assumption
        image = [y, x, c] shape, RGB
        """
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        image = input_dict[tag_image]
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32)
        elif isinstance(image, np.ndarray):
            image = image.astype(np.float32)
        image /= self.scale
        image -= self.mean
        image /= self.std

        return {tag_image : image,
                tag_label : None}


class DictNormalize(Normalize):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.),
                 scale=255.0, gray=False):
        super(DictNormalize, self).__init__(mean=mean, std=std, scale=scale)
        """
        Assumption
        image = [y, x, c] shape, RGB
        label = [y, x] shape, palette or grayscale
        """
        self.gray_ = gray

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        image = super().__call__(input_dict)[tag_image]
        label = input_dict[tag_label]

        if isinstance(label, Image.Image):
            label = np.array(label).astype(np.float32)
        elif isinstance(label, np.ndarray):
            label = label.astype(np.float32)

        if self.gray_:
            label /= self.scale

        return {tag_image : image,
                tag_label : label}


class ToTensor(object):
    def __init__(self):
        pass
        """
            Assumption
            image = [y, x, c] shape, RGB
        """

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        # Height x Width x Channels -> Channels x Height x Width
        image = input_dict[tag_image]
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32)
        elif isinstance(image, np.ndarray):
            image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        torch_image = torch.from_numpy(image).float()

        return {tag_image: torch_image,
                tag_label: None}


class Dict2Tensor(ToTensor):
    def __init__(self, boundary_white=False, two_dim=False):
        super(Dict2Tensor, self).__init__()
        """
        Assumption
        image = [y, x, c] shape, RGB
        label = [y, x] shape, palette or grayscale
        """
        self.boundary_white = boundary_white
        self.two_dim = two_dim

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        torch_image = super().__call__(input_dict)[tag_image]
        label = input_dict[tag_label]
        if isinstance(label, Image.Image):
            label = np.array(label).astype(np.float32)
        elif isinstance(label, np.ndarray):
            label = label.astype(np.float32)

        # label 2D -> 3D
        if self.two_dim:
            label = np.expand_dims(label, axis=-1)

        # Boundary White -> Black
        if self.boundary_white:
            label[label == 255] = 0

        # transpose for torch.Tensor [H x W x C] -> [C x H x W]
        label = label.transpose((2, 0, 1))

        torch_label = torch.from_numpy(label).float()

        return {tag_image : torch_image,
                tag_label : torch_label}


class Resize(object):
    def __init__(self, size:Union[tuple, list], image_mode=Image.BICUBIC):
        # (Width, Height) -> (Height, Width)
        #    X   ,   Y          Y   ,   X
        self.size = tuple(reversed(size))
        self.image_mode = image_mode

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        image = input_dict[tag_image]
        if isinstance(image, Image.Image):
            image = image.resize(self.size, self.image_mode)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image = image.resize(self.size, self.image_mode)

        return {tag_image : image,
                tag_label : None}


class DictResize(Resize):
    def __init__(self, size:Union[tuple, list], image_mode=Image.BICUBIC, label_mode=Image.BILINEAR):
        super(DictResize, self).__init__(size=size, image_mode=image_mode)
        self.label_mode = label_mode

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        image = super().__call__(input_dict=input_dict)[tag_image]
        label = input_dict[tag_label]
        if isinstance(label, Image.Image):
            pass
        elif isinstance(label, np.ndarray):
            label = Image.fromarray(label)

        label = label.resize(self.size, self.label_mode)

        return {tag_image : image,
                tag_label : label}


class AlphaKill(object):
    # kill alpha channel 4D -> 3D
    def __init__(self):
        pass

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        image = input_dict[tag_image]
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image = image.convert('RGB')

        return {tag_image : image,
                tag_label : None}


class DictAlphaKill(AlphaKill):
    def __init__(self):
        pass

    def __call__(self, input_dict : {str : Union[Image.Image, np.ndarray]}):
        image = super().__call__(input_dict=input_dict)[tag_image]
        label = input_dict[tag_label]
        if isinstance(label, np.ndarray):
            label = Image.fromarray(label).convert('L')

        return {tag_image : image,
                tag_label : label}
