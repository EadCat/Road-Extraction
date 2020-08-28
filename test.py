import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# My Libraries
from parameters import *
from path import DataManager, DirectoryManager
from networks.nets import *
from dataloader import dict_transforms
from dataloader import dataset
from functions.loss_F import binary_entropy_2d
import functions.utils as util
from eval.evaluator import Evaluator
from functions.plot import PlotGenerator, iter2dict

# basic utils
import os
import time
import numpy as np

if __name__ == "__main__":
    # directory
    data_man = DataManager(os.getcwd())  # Get test data directory
    # ===================================== load weights targeting panel =====================================
    # if mode is 'test', must be set.
    epoch = 197  # epoch number of model you want to test.
    branch_num = 4
    # =========================================================================================================
    # if mode is 'external_test', DirectoryManager(external_weight= set here!!)
    dir_man = DirectoryManager(model_name, mode='test', branch_num=branch_num, load_num=epoch)  # model_name is defined in <parameters.py>.

    # load trained model
    print(f'loading network...')
    netend = NetEnd(num_classes=params['num_classes'])  # set the number of classification.
    path = dir_man.load()
    model = ResNet101_DeeplabV3(end_module=netend, pretrain=permission['pretrain'])
    model.load_state_dict((torch.load(path)))

    # GPU setting
    environment = {}
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GPU {torch.cuda.get_device_name()} available.')
        model.cuda()
        environment['gpu'] = True

    else:
        device = torch.device('cpu')
        print(f'GPU unable.')
        environment['gpu'] = False

    # dataloader
    # handle with care, always keep the dimensions of your images in mind.
    print('test data processing...')
    toronto_setting = transforms.Compose([dict_transforms.DictResize(params['resized']),
                                          dict_transforms.DictNormalize(gray=True,
                                                                        mean=params['mean'],
                                                                        std=params['std']),
                                          dict_transforms.Dict2Tensor(two_dim=True)])
    pix2pix_setting = transforms.Compose([dict_transforms.DictResize(params['resized']),
                                          dict_transforms.DictNormalize(gray=True,
                                                                        mean=params['mean'],
                                                                        std=params['std']),
                                          dict_transforms.Dict2Tensor(two_dim=True)])
    test_set_Toronto = dataset.RoadDataset(data_dir=data_man.test(), label_dir=data_man.label('test'),
                                           transform=toronto_setting,
                                           dataname_extension='*.tiff', labelname_extension='*.tif')
    test_set_Pix = dataset.RoadDataset(data_dir=data_man.test(), label_dir=data_man.label('test'),
                                       transform=pix2pix_setting,
                                       dataname_extension='*.jpg', labelname_extension='*.jpg')
    test_set = test_set_Toronto + test_set_Pix
    print(f'test data : {len(test_set)} files detected.')
    test_loader = DataLoader(dataset=test_set, batch_size=test_params['test_batch'],
                             shuffle=False, num_workers=user_setting['test_processes'])

    # ========================================================================================================================
    # test
    # network test mode
    model.eval()
    test_start = time.perf_counter()
    loss_dict = {}

    mae = 0
    evaluator = Evaluator(mode='binary')

    print(f'test start.')
    for i, data in enumerate(test_loader):
        # load
        image, label = data[tag_image], data[tag_label]
        name = data[tag_name]

        # gpu copy
        if environment['gpu']:
            image, label = image.cuda(), label.cuda()

        # Forwarding
        with torch.no_grad():
            output = model.forward(image)

        # sample
        util.imgstore(output*255.0, nums=1, save_dir=dir_man.test_sample(), epoch=epoch, cls='test', filename=name)

        # loss
        loss = binary_entropy_2d(output, label)
        print(f'iter : {i+1} / loss : {loss:.3f}')
        loss_dict[i] = loss.item()
        util.write_line({i+1:loss.item()}, os.path.join(dir_man.test(), 'model_loss.txt'))

        # evaluator recording
        mae += torch.mean(torch.abs(output - label))
        evaluator.add(pred=output, mask=label)

    # test parameter record
    util.snapshot_maker(test_params, os.path.join(dir_man.test(), 'test_model_snapshot.txt'))

    precision, recall, accuracy, f1_score, confidence = evaluator.view()

    # print test spending time.
    print(f'{time.perf_counter() - test_start:.3f} s spended.')

    # plot data processing : iterable -> dictionary
    prec_Data = iter2dict(confidence, precision)
    reca_Data = iter2dict(confidence, recall)
    PR_Data = iter2dict(recall, precision)
    loss_Data = loss_dict
    accu_Data = iter2dict(confidence, accuracy)
    f1_Data = iter2dict(confidence, f1_score)

    # record
    util.write_line(prec_Data, os.path.join(dir_man.test(), 'Precision.txt'))
    util.write_line(reca_Data, os.path.join(dir_man.test(), 'Recall.txt'))
    util.write_line(PR_Data, os.path.join(dir_man.test(), 'PR Curve.txt'))
    util.write_line(accu_Data, os.path.join(dir_man.test(), 'Accuracy.txt'))
    util.write_line(f1_Data, os.path.join(dir_man.test(), 'F1 Score.txt'))

    # plot
    prec_plot = PlotGenerator(1, 'precision', (20, 15), xlabel='confidence', ylabel='precision')
    prec_plot.add_data(prec_Data)
    prec_plot.add_set(name='precision', color='r')
    prec_plot.plot()
    prec_plot.save(os.path.join(dir_man.test_graph(), 'precision.jpg'))

    reca_plot = PlotGenerator(2, 'recall', (20, 15), xlabel='confidence', ylabel='recall')
    reca_plot.add_data(reca_Data)
    reca_plot.add_set(name='recall', color='b')
    reca_plot.plot()
    reca_plot.save(os.path.join(dir_man.test_graph(), 'recall.jpg'))

    overlay_pr = PlotGenerator(3, 'Precision & Recall', (20, 15), xlabel='confidence', ylabel='value')
    overlay_pr.add_data(prec_plot.data(0))
    overlay_pr.add_data(reca_plot.data(0))
    overlay_pr.add_set(data=prec_plot.set(0))
    overlay_pr.add_set(data=reca_plot.set(0))
    overlay_pr.plot()
    overlay_pr.save(os.path.join(dir_man.test_graph(), 'PR_Overlay.jpg'))

    PR_plot = PlotGenerator(4, 'PR Curve', (20, 15), xlabel='Recall', ylabel='Precision')
    PR_plot.add_data(PR_Data)
    PR_plot.add_set(name='PR Curve', color='g')
    PR_plot.plot()
    PR_plot.save(os.path.join(dir_man.test_graph(), 'PR_Curve.jpg'))

    loss_plot = PlotGenerator(5, 'BCELoss', (20, 15), xlabel='Epochs', ylabel='loss')
    loss_plot.add_data(loss_Data)
    loss_plot.add_set(name='BCELoss', color='y')
    loss_plot.plot()
    loss_plot.save(os.path.join(dir_man.test_graph(), 'BCELoss.jpg'))

    f1_plot = PlotGenerator(6, 'F1 Score', (20, 15), xlabel='confidence', ylabel='F1 Score')
    f1_plot.add_data(f1_Data)
    f1_plot.add_set(name='F1 Score', color='m')
    f1_plot.plot()
    f1_plot.save(os.path.join(dir_man.test_graph(), 'F1 Score.jpg'))

    accu_plot = PlotGenerator(7, 'Accuracy', (20, 15), xlabel='confidence', ylabel='accuracy')
    accu_plot.add_data(accu_Data)
    accu_plot.add_set(name='Accuracy', color='k')
    accu_plot.plot()
    accu_plot.save(os.path.join(dir_man.test_graph(), 'Accuracy.jpg'))


























