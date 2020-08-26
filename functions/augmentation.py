import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from parameters import tag_image, tag_label, tag_name, label_folder_name

import random
import os


class AugManager(object):
    def __init__(self, iaalist=None):
        if iaalist is None:
            iaalist = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.ChannelShuffle(0.3)),
                iaa.Sometimes(0.5, iaa.MultiplyHue((0.5, 1.5))),
                iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),
                iaa.Sometimes(0.5, iaa.Fliplr(0.5)),
                iaa.Sometimes(0.5, iaa.Flipud(0.5)),
                iaa.Sometimes(0.5, iaa.Rotate((-50, 50)))
            ])
        self.transformSet = iaalist
        self.outscale = random.choice([0.8, 0.85, 0.9, 0.95])

    def __call__(self, input_dict : {str : Image.Image}) -> dict:
        image = np.array(input_dict[tag_image])
        label = np.array(input_dict[tag_label])

        # size measure
        y_max = image.shape[0]
        x_max = image.shape[1]

        # np.ndarray -> imgaug.augmentables.segmaps.SegmentationMapsOnImage
        label = SegmentationMapsOnImage(label, shape=label.shape)

        # augmentation
        zoomset = iaa.OneOf([
            iaa.Identity(),  # do nothing
            iaa.Affine(scale=self.outscale),  # zoom out
            RandomCrop(y_max, x_max).cut()  # zoom in
        ])
        image, label = zoomset(image=image, segmetation_maps=label)
        image, label = self.transformSet(image=image, segmentation_maps=label)

        # imgaug.augmentables.segmaps.SegmentationMapsOnImage -> np.ndarray
        label = label.get_arr()

        return {tag_image : image,
                tag_label : label}

    def augstore(self, src:dict, dst_base:str,
                 dataname_extension='.tiff', labelname_extension='.tif',
                 identifier=None):

        os.makedirs(dst_base, exist_ok=True)
        os.makedirs(os.path.join(dst_base, label_folder_name), exist_ok=True)
        # get image
        image = src[tag_image] # PIL.Image.Image
        label = src[tag_label] # PIL.Image.Image
        name = src[tag_name] # str

        # PIL -> numpy
        image = np.array(image)
        label = np.array(label)

        # size measure
        y_max = image.shape[0]
        x_max = image.shape[1]

        # np.ndarray -> imgaug.augmentables.segmaps.SegmentationMapsOnImage
        label = SegmentationMapsOnImage(label, shape=label.shape)

        # augmentation
        zoomset = iaa.OneOf([
            iaa.Identity(),  # do nothing
            iaa.Affine(scale=self.outscale),  # zoom out
            RandomCrop(y_max, x_max).cut()  # zoom in
        ])
        image, label = zoomset(image=image, segmentation_maps=label)
        image, label = self.transformSet(image=image, segmentation_maps=label)

        # imgaug.augmentables.segmaps.SegmentationMapsOnImage -> np.ndarray
        label = label.get_arr()

        if not identifier == None:
            name = name + '_' + str(identifier)

        # numpy -> PIL.Image.Image
        image = Image.fromarray(image)
        label = Image.fromarray(label)

        image.save(os.path.join(dst_base, name + dataname_extension))
        label.save(os.path.join(dst_base, label_folder_name, name + labelname_extension))

        return {tag_image : image,
                tag_label : label,
                tag_name : name}


class RandomCrop(object):
    def __init__(self, max_height, max_width):
        assert isinstance(max_height, int) and max_height >= 1, 'max_height must be positive integer type.'
        assert isinstance(max_width, int) and max_width >= 1, 'max_width must be positive integer type.'

        self.percent_limit = 0.15
        self.top, self.right, self.bottom, self.left = self.operate_location(max_height, max_width)

    def operate_location(self, max_height, max_width):
        import random
        max_height = max_height + 1
        max_width = max_width + 1

        min_height = int(self.percent_limit * max_height)
        min_width = int(self.percent_limit * max_width)

        fix_height = random.randint(min_height, max_height)
        fix_width = random.randint(min_width, max_width)

        left = random.randint(0, max_width - fix_width)
        up = random.randint(0, max_height - fix_height)

        right = max_width - fix_width - left
        down = max_height - fix_height - up

        return up, right, down, left

    def cut(self):
        return iaa.Crop(px=(self.top, self.right, self.bottom, self.left))





















