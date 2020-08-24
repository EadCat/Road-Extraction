import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from parameters import tag_image, tag_label, tag_name, label_folder_name

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

    def __call__(self, input_dict : {str : Image.Image}) -> dict:
        image = np.array(input_dict[tag_image])
        label = np.array(input_dict[tag_label])

        # np.ndarray -> imgaug.augmentables.segmaps.SegmentationMapsOnImage
        label = SegmentationMapsOnImage(label, shape=label.shape)

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

        # np.ndarray -> imgaug.augmentables.segmaps.SegmentationMapsOnImage
        label = SegmentationMapsOnImage(label, shape=label.shape)

        # augment
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