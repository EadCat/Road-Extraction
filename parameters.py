from collections import OrderedDict
import os

__all__ = ['model_name', 'tag_image', 'tag_label', 'tag_name',
           'train_directory', 'test_directory', 'valid_directory', 'label_folder_name',
           'weight_save_directory', 'params', 'test_params', 'user_setting', 'permission'
           ]

# model name information for save and load
model_name = 'ResNet101-DeepLabV3'

# dictionary key
tag_image = 'image'
tag_label = 'label'
tag_name = 'name'

# data directory
train_directory = r'data/train'
test_directory = r'data/test'
valid_directory = r'data/validation'
label_folder_name = r'labels'

# directory for saving results of model running
weight_save_directory = r'save'

# ============================= model parameter control panel =============================
# =============================== in training environment =================================
params = OrderedDict()
params['total_epochs'] = 40
params['train_batch'] = 10
params['valid_batch'] = 10
params['learning_rate'] = 1e-5
params['num_classes'] = 1
params['resized'] = (500, 500) # tuple
params['mean'] = [0.485, 0.456, 0.406]
params['std'] = [0.229, 0.224, 0.225]
params['pretrain'] = 'COCO 2017'  # only for writing record
# =========================================================================================

# ============================= model parameter control panel =============================
# ================================= in test environment ===================================
test_params = OrderedDict()
test_params['test_batch'] = 10
test_params['num_classes'] = 1
test_params['resized'] = (500, 500)
test_params['mean'] = [0.485, 0.456, 0.406]
test_params['std'] = [0.229, 0.224, 0.225]
test_params['pretrain'] = 'COCO 2017'
# =========================================================================================

# ======================================== setting =========================================
user_setting = {}
user_setting['gpu_id'] = 0
user_setting['train_processes'] = 8 # data loader num_process
user_setting['valid_processes'] = 8
user_setting['test_processes'] = 8
user_setting['validation_intervals'] = 1 # validating period epoch
user_setting['model_store_intervals'] = 1 # storing period of model's info snapshot
user_setting['epoch_store_intervals'] = 1
user_setting['iter_print_intervals'] = 50
user_setting['img_save_intervals'] = 1 # epoch
# =========================================================================================

# ============================= permission control panel =============================
permission = {}
permission['validation'] = True
permission['weight_save'] = True
permission['snapshot_save'] = True
permission['loss_save'] = True
permission['shuffle'] = True
permission['epoch_print'] = True
permission['iter_print'] = True
permission['valid_print'] = True
permission['pretrain'] = True
permission['train_predict_store'] = True
# =========================================================================================