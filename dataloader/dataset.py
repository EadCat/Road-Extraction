from torch.utils.data import Dataset
import os, glob

from parameters import *

__all__ = ['RoadDataset']


class RoadDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None, dataname_extension='*.tiff', labelname_extension='*.tif',
                 label_gray=True):
        super(RoadDataset, self).__init__()

        ext_len = len(dataname_extension) - 1
        self.data_list = sorted(glob.glob(os.path.join(data_dir, dataname_extension), recursive=False))
        self.label_list = sorted(glob.glob(os.path.join(label_dir, labelname_extension), recursive=False))
        self.data_name = [os.path.basename(data)[:-ext_len] for data in self.data_list]

        assert len(self.data_list) == len(self.label_list), 'The number of data and label must be equal.'

        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_gray = label_gray

    def __len__(self):
        return len(self.data_list)

    def __str__(self):
        print('instance of dataloader.dataset.RoadDataset.')
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
        img, label, name = self.make_triplet(idx)
        sample = {tag_image: img, tag_label: label}

        if self.transform is not None:
            sample = self.transform(sample)

        sample[tag_name] = name

        return sample

    def data_list(self):
        return self.data_name

    def make_triplet(self, idx):
        from PIL import Image

        img = Image.open(self.data_list[idx])
        if self.label_gray:
            target = Image.open(self.label_list[idx]).convert('L')
        else:
            target = Image.open(self.label_list[idx])
        name = self.data_name[idx]

        return img, target, name
