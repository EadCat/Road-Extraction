import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from parameters import *
from networks.nets import *
from dataloader.dataset import MetaDataset
from dataloader.dict_transforms import *
import functions.utils as util

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------

    # ========================================== dir & param ==========================================
    data_dir = './'
    weight_dir = './'
    dst_dir = './'
    store_num = 10
    the_name = os.path.splitext(os.path.basename(weight_dir))[0]
    assert params['test_batch'] >= store_num, 'batch size must be bigger than the number of storing image.'
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== transform ===========================================
    transform_set = transforms.Compose([Resize(params['resized']),
                                        Normalize(mean=params['mean'], std=params['std']),
                                        ToTensor()])
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ============================================ Dataset ============================================
    pred_data = MetaDataset(data_dir=data_dir, transform=transform_set)
    data_loader = DataLoader(dataset=pred_data, batch_size=test_params['test_batch'],
                             shuffle=False, num_workers=user_setting['test_processes'])
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

