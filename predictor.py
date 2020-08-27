import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from parameters import *
from networks.nets import *
from dataloader.dataset import MetaDataset
from dataloader import dict_transforms
import functions.utils as util

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------

    # ========================================== dir & param ==========================================
    data_dir = r'/home/user/Desktop/test_dataset'
    weight_dir = r'./save/branch_3/ResNet101-DeepLabV3_epoch_117.pth'
    dst_dir = os.path.join(data_dir, 'predicton')
    store_num = 10
    the_name = os.path.splitext(os.path.basename(weight_dir))[0]
    assert test_params['test_batch'] >= store_num, 'batch size must be bigger than the number of storing image.'
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== transform ===========================================
    transform_set1 = transforms.Compose([dict_transforms.Resize(params['resized']),
                                         dict_transforms.Normalize(mean=params['mean'], std=params['std']),
                                         dict_transforms.ToTensor()])
    transform_set2 = transforms.Compose([dict_transforms.AlphaKill(),
                                         dict_transforms.Resize(params['resized']),
                                         dict_transforms.Normalize(mean=params['mean'], std=params['std']),
                                         dict_transforms.ToTensor()])
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ============================================ Dataset ============================================
    pred_data1 = MetaDataset(data_dir=data_dir, transform=transform_set1, dataname_extension='*.jpg')
    pred_data2 = MetaDataset(data_dir=data_dir, transform=transform_set2, dataname_extension='*.png')
    pred_data = pred_data1 + pred_data2
    data_loader = DataLoader(dataset=pred_data, batch_size=test_params['test_batch'],
                             shuffle=False, num_workers=user_setting['test_processes'])
    print(f'{len(pred_data)} data detected.')
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== Model Load ==========================================
    netend = NetEnd(1)
    model = ResNet101_DeeplabV3(end_module=netend, pretrain=permission['pretrain'])
    model.load_state_dict(torch.load(weight_dir))
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ========================================== GPU setting ==========================================
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
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------
    os.makedirs(dst_dir, exist_ok=True)
    print(f'save directory : {dst_dir}')
    # ============================================ run ================================================
    model.eval()
    for i, data in enumerate(data_loader):
        image = data[tag_image]
        name = data[tag_name]

        if environment['gpu']:
            image = image.cuda()

        with torch.no_grad():
            output = model.forward(image)

        util.imgstore(output*255.0, nums=store_num, save_dir=dst_dir, epoch=the_name, filename=name)
        print(f'{(i / len(data_loader)) * 100:.2f} % done.')
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

