# Pytorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Parameter Setup
from parameters import *

# My Libraries
from path import *
from networks import nets as net
from dataloader import dataset
from dataloader import dict_transforms
from functions import loss_F
from functions import utils
from functions import augmentation
from functions.plot import PlotGenerator
from networks.nets import NetEnd, ResNet101_DeeplabV3, ResNet101_FCN, ResNet50_DeeplabV3

# Basic Utility
import os
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    environment = {}
    # ======================================= Directory Panel =============================================
    data_man = DataManager(os.getcwd())
    mode = 'new'  # please set the mode. ['new', 'load', 'overlay']
    load_branch = None  # need 'load' or 'overlay' mode. you can set integer here.
    load_num = None  # need 'load' or 'overlay' mode. you can set integer here.
    dir_man = DirectoryManager(model_name=model_name, mode=mode, branch_num=load_branch,
                               load_num=load_num)
    # =====================================================================================================

    # ============================================== model definition ==============================================
    if mode is 'new':
        print('start training from epoch 1.')
        params['resume_epoch'] = 1
        print('constructing network...')
        netend = net.NetEnd(num_classes=params['num_classes'])
        network = net.ResNet101_DeeplabV3(netend, pretrain=permission['pretrain']) # <- model definition

    elif mode is 'external_train':
        print('start training from epoch 1.')
        params['resume_epoch'] = 1
        print('constructing network...')
        netend = net.NetEnd(num_classes=params['num_classes'])
        network = net.ResNet50_DeeplabV3(netend, pretrain=False)
        path = dir_man.load()
        print(f'{dir_man.load()} loading...')
        network.load_state_dict((torch.load(path)))

    elif mode is 'load' or mode is 'overlay':
        assert load_branch is not None, 'check load_branch.'
        assert load_num is not None, 'check load_num.'
        print('loading weights file...')
        params['resume_epoch'] = load_num + 1
        # model load
        print('constructing network...')
        netend = NetEnd(num_classes=params['num_classes'])
        path = dir_man.load()
        network = ResNet101_DeeplabV3(end_module=netend, pretrain=permission['pretrain']) # <- model definition
        print(f'{dir_man.load()} loading...')
        network.load_state_dict((torch.load(path)))

    else:
        print("please modify 'mode' variable.")
        import sys
        sys.exit(1)

    assert params['total_epochs'] >= params['resume_epoch'], 'please check resume start point.'
    # ==============================================================================================================
    print('network has been constructed.')
    print(f'branch number : {dir_man.branch_info()}')
    print(f'model name: {dir_man.name_info()}')

    # ================================= GPU setting =================================
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GPU {torch.cuda.get_device_name()} available.')
        network.cuda()
        environment['gpu'] = True
    else:
        device = torch.device("cpu")
        print(f'GPU unable.')
        environment['gpu'] = False
    # ===============================================================================

    # ========================== optimizer & loss function ==========================
    optimizer = torch.optim.Adam(network.parameters(), lr=params['learning_rate'])
    params['optimizer'] = 'Adam' # <- It must be set manually. (snapshot record)
    params['loss_function'] = 'BinaryCrossEntropyLoss'  # <- It must be set manually. (snapshot record)
    # ===============================================================================

    # =========================================== image pre-processing & load ==========================================
    toronto_setting = transforms.Compose([dict_transforms.Resize(params['resized']),
                                          augmentation.AugManager(),
                                          dict_transforms.DictNormalize(gray=True,
                                                                        mean=params['mean'],
                                                                        std=params['std']),
                                          dict_transforms.Dict2Tensor(two_dim=True)])
    pix2pix_setting = transforms.Compose([dict_transforms.Resize(params['resized']),
                                          augmentation.AugManager(),
                                          dict_transforms.DictNormalize(gray=True,
                                                                        mean=params['mean'],
                                                                        std=params['std']),
                                          dict_transforms.Dict2Tensor(two_dim=True)])

    # training set
    train_set_Toronto = dataset.RoadDataset(data_dir=data_man.train(), label_dir=data_man.label('train'), transform=toronto_setting)
    train_set_Pix = dataset.RoadDataset(data_dir=data_man.train(), label_dir=data_man.label('train'), transform=pix2pix_setting,
                                        dataname_extension='*.jpg', labelname_extension='*.jpg')
    train_set = train_set_Toronto + train_set_Pix
    print(f'train data : {len(train_set)} files detected.')
    train_loader = DataLoader(dataset=train_set, batch_size=params['train_batch'],
                              shuffle=permission['shuffle'], num_workers=user_setting['train_processes'])

    # validating set
    if permission['validation']:
        valid_set_Toronto = dataset.RoadDataset(data_dir=data_man.validation(), label_dir=data_man.label('valid'), transform=toronto_setting)
        valid_set_Pix = dataset.RoadDataset(data_dir=data_man.validation(), label_dir=data_man.label('valid'), transform=pix2pix_setting,
                                            dataname_extension='*.jpg', labelname_extension='*.jpg')
        valid_set = valid_set_Toronto + valid_set_Pix
        print(f'validation data : {len(valid_set)} files detected.')
        valid_loader = DataLoader(dataset=valid_set, batch_size=params['valid_batch'],
                                  shuffle=False, num_workers=user_setting['valid_processes'])
    # ==================================================================================================================
    epoch_loss = 0.0
    if permission['validation']:
        print(f'your validation epoch setting : {user_setting["validation_intervals"]}')
    else:
        print(f'validation disabled.')
    train_loss_dict = {}
    iter_loss_dict = {}
    val_loss_dict = {}
    # =========================== training epoch ===============================
    # training
    print(f'training start.')
    for epoch in range(params['resume_epoch'], params['total_epochs'] + 1):
        network.train()
        iter_loss = 0.0
        epoch_start = time.perf_counter()
        print(f'epoch: {epoch}')
        # ========================================= training ==============================================
        for i, data in enumerate(train_loader):
            # load
            image, label = data[tag_image], data[tag_label]

            name = data[tag_name]

            # gpu copy
            if environment['gpu']:
                image, label = image.cuda(), label.cuda()

            # optimizer initialization
            optimizer.zero_grad()

            # Forwarding
            output = network.forward(image)

            # loss operation
            loss = loss_F.binary_entropy_2d(output, label)
            epoch_loss += loss.item()
            iter_loss += loss.item()
            loss.backward()
            optimizer.step()  # update

            # iter loss operation & print
            if i % user_setting['iter_print_intervals'] == user_setting['iter_print_intervals']-1:
                iter_loss /= user_setting['iter_print_intervals']
                if permission['iter_print']:
                    print(f'[{epoch}, {i+1}] iteration loss : {iter_loss : .3f}')
                iter_loss_dict[str(epoch)+':'+str(i+1)] = iter_loss
                if permission['loss_save']:
                    utils.write_line({str(epoch)+':'+str(i+1): iter_loss_dict[str(epoch)+':'+str(i+1)]},
                                     os.path.join(dir_man.branch(), 'history', model_name+'_iter'+'.txt'))
                iter_loss = 0.0  # reset
        # ========================================= training ==============================================
        # model prediction image store.
        if permission['train_predict_store']:
            if epoch % user_setting['img_save_intervals'] == 0:
                utils.imgstore(output*255.0, nums=2, save_dir=dir_man.train_predict_store(), epoch=epoch,
                               cls='pred', filename=name)
                utils.imgstore(label*255.0, nums=2, save_dir=dir_man.train_predict_store(), epoch=epoch,
                               cls='label', filename=name)


        # epoch loss print
        if epoch % user_setting['epoch_store_intervals'] == 0:
            epoch_loss /= len(train_loader)
            epoch_loss /= user_setting['epoch_store_intervals']
            if permission['epoch_print']:
                print(f'{epoch} epoch train loss : {epoch_loss : .3f}')
            train_loss_dict[str(epoch)] = epoch_loss
            if permission['loss_save']:
                utils.write_line({str(epoch): train_loss_dict[str(epoch)]},
                                 os.path.join(dir_man.branch(), 'history', model_name+'_epoch'+'.txt'))
            epoch_loss = 0

        # save the model weights
        if epoch % user_setting['model_store_intervals'] == 0:
            if permission['snapshot_save']:
                utils.snapshot_maker(params, dir_man.save_dir()+str(epoch)+'.txt')
            if permission['weight_save']:
                torch.save(network.state_dict(), dir_man.save_dir()+str(epoch)+'.pth')
                print(f'The model weights saved at epoch {epoch}.')
                print(f'save directory : {dir_man.save_dir()+str(epoch)+".pth"}')

        # ========================================= validating ==============================================
        if permission['validation']:
            if epoch % user_setting['validation_intervals'] == 0:
                network.eval()
                iter_loss = 0.0
                # ==================================== validation iter =========================================
                for i, val in enumerate(valid_loader):
                    image, label = val[tag_image], val[tag_label]
                    if environment['gpu']:  # gpu copy
                        image, label = image.cuda(), label.cuda()
                    with torch.no_grad():
                        output = network.forward(image)

                    loss = loss_F.binary_entropy_2d(output, label)
                    iter_loss += loss.item()
                # ==============================================================================================
                iter_loss /= len(valid_loader)
                if permission['valid_print']:
                    print(f'{epoch} epoch validation loss : {iter_loss:.3f}')
                val_loss_dict[str(epoch)] = iter_loss
                if permission['loss_save']:
                    utils.write_line({str(epoch): val_loss_dict[str(epoch)]},
                                     os.path.join(dir_man.branch(), 'history', model_name+'_valid'+'.txt'))
                iter_loss = 0.0
        # ========================================= validating ==============================================
        print(f'eta : {time.perf_counter()-epoch_start:.3f}s')
    # =========================== training epoch ===============================

    # =================================== plot =======================================
    # train loss
    plot1 = PlotGenerator(1, 'train', (20, 15), xlabel='epochs', ylabel='BCELoss')
    plot1.add_data(train_loss_dict)
    plot1.add_set(name='train loss', color='r')
    # plot1.interval_remove(interval=3, idx=0, update=True)
    plot1.plot()
    plot1.save(os.path.join(dir_man.graph_store(), 'train.png'))

    # validation loss
    if permission['validation']:
        plot2 = PlotGenerator(2, 'Validation', (20, 15), xlabel='epochs', ylabel='BCELoss')
        plot2.add_data(val_loss_dict)
        plot2.add_set(name='validation loss', color='g')
        # plot2.interval_remove(interval=3, idx=0, update=True)
        plot2.plot()
        plot2.save(os.path.join(dir_man.graph_store(), 'validation.png'))

        # train & validation loss
        plot3 = PlotGenerator(3, 'Training', (20, 15), xlabel='epochs', ylabel='BCELoss')
        plot3.add_data(plot1.data(0))
        plot3.add_data(plot2.data(0))
        plot3.add_set(data=plot1.set(0))
        plot3.add_set(data=plot2.set(0))
        plot3.plot()
        plot3.save(os.path.join(dir_man.graph_store(), 'Training.png'))

    print('training ends.')



































