if __name__ == '__main__':
    # main code for saving augmented images.
    import os
    from functions.augmentation import AugManager
    from parameters import *
    from dataloader.dataset import RoadDataset


    train_dir = os.path.join(os.getcwd(), train_directory)
    label_dir = os.path.join(train_dir, label_folder_name)
    augsave_dir = os.path.join(os.getcwd(), r'data/aug')
    dataset_toronto = RoadDataset(data_dir=train_dir, label_dir=label_dir)
    dataset_pix = RoadDataset(data_dir=train_dir, label_dir=label_dir,
                              dataname_extension='*.jpg', labelname_extension='*.jpg')

    num_aug = 5
    augset = AugManager()

    for i, data in enumerate(dataset_toronto):
        for j in range(num_aug):
            augset.augstore(data, augsave_dir, identifier=j+1)
        print(f'toronto: {(i+1)/len(dataset_toronto) * 100 : .2f} % done.')

    for i, data in enumerate(dataset_pix):
        for j in range(num_aug):
            augset.augstore(data, augsave_dir, identifier=j+1,
                            dataname_extension='.jpg', labelname_extension='.jpg')
        print(f'pix2pix: {(i+1)/len(dataset_pix) * 100 : .2f} % done.')