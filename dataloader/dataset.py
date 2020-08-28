from torch.utils.data import Dataset
import os, glob

from parameters import *

__all__ = ['MetaDataset', 'RoadDataset']


class MetaDataset(Dataset):
    def __init__(self, data_dir, transform=None, dataname_extension='*.jpg'):
        super(MetaDataset).__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.data_list = sorted(glob.glob(os.path.join(data_dir, dataname_extension), recursive=False))
        self.data_name = [os.path.splitext(os.path.basename(data))[0] for data in self.data_list]

    def __len__(self):
        return len(self.data_list)

    def __str__(self):
        print('instance of dataloader.dataset.MetaDataset.')
        for i, name in enumerate(self.data_name):
            print(f'{self.data_name[i]}, ', end='')
            if i % 5 == 4:
                print()

    def __repr__(self):
        for i, name in enumerate(self.data_name):
            print(f'{self.data_name[i]}, ', end='')
            if i % 5 == 4:
                print()

    def __getitem__(self, idx):
        img, name = self.make_pair(idx)
        sample = {tag_image: img}

        if self.transform is not None:
            sample = self.transform(sample)

        sample[tag_name] = name

        return sample

    def data_list(self):
        return self.data_name

    def make_pair(self, idx):
        from PIL import Image

        img = Image.open(self.data_list[idx])
        name = self.data_name[idx]

        return img, name


class RoadDataset(MetaDataset):
    def __init__(self, data_dir, label_dir, transform=None, dataname_extension='*.tiff', labelname_extension='*.tif',
                 label_gray=True):
        super().__init__(data_dir=data_dir, transform=transform, dataname_extension=dataname_extension)

        self.data_list = sorted(glob.glob(os.path.join(data_dir, dataname_extension), recursive=False))
        self.label_list = sorted(glob.glob(os.path.join(label_dir, labelname_extension), recursive=False))
        self.data_name = [os.path.splitext(os.path.basename(data))[0] for data in self.data_list]

        assert len(self.data_list) == len(self.label_list), 'The number of data and label must be equal.'

        self.label_dir = label_dir
        self.label_gray = label_gray

    def __getitem__(self, idx):
        img, label, name = self.make_triplet(idx)
        sample = {tag_image: img, tag_label: label}

        if self.transform is not None:
            sample = self.transform(sample)

        sample[tag_name] = name

        return sample

    def make_triplet(self, idx):
        from PIL import Image

        img = Image.open(self.data_list[idx])
        if self.label_gray:
            gt = Image.open(self.label_list[idx]).convert('L')
        else:
            gt = Image.open(self.label_list[idx])
        name = self.data_name[idx]

        return img, gt, name
