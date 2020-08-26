import os, glob

from parameters import *

__all__ = ['DataManager', 'DirectoryManager']


class DataManager:
    # Data Managing Class
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_data = os.path.join(self.root_dir, train_directory)
        self.train_labels = os.path.join(self.train_data, label_folder_name)
        self.valid_data = os.path.join(self.root_dir, valid_directory)
        self.valid_labels = os.path.join(self.valid_data, label_folder_name)
        self.test_data = os.path.join(self.root_dir, test_directory)
        self.test_labels = os.path.join(self.test_data, label_folder_name)

    def __repr__(self):
        self.__str__()

    def __str__(self):
        printer = 'path.DataManager class instance'
        print(f'train directory ==> : {self.train_data}')
        print(f'valid directory ==> : {self.valid_data}')
        print(f'test  directory ==> : {self.test_data}')
        return printer

    def train(self):
        return self.train_data

    def validation(self):
        return self.valid_data

    def test(self):
        return self.test_data

    def label(self, target: str):
        if target is 'train':
            return self.train_labels
        elif target is 'validation' or target is 'valid':
            return self.valid_labels
        elif target is 'test':
            return self.test_labels
        else:
            print('Illegal input parameter. check [target] input.')

    def mkdir(self):
        # make all folders
        os.makedirs(self.train_labels, exist_ok=True)
        os.makedirs(self.test_labels, exist_ok=True)
        os.makedirs(self.valid_labels, exist_ok=True)


class DirectoryManager:
    # Saving Directory Manager
    def __init__(self, model_name, mode='new', branch_num=None, load_num=None):

        # mode : 'new', 'load', 'overlay', 'test'
        # branch_num : target branch folder number for 'load' or 'overlay' or 'test'
        # load_num : target epoch number for 'load' or 'overlay' or 'test'
        self.root_dir = os.getcwd()
        self.weight_dir = os.path.join(self.root_dir, weight_save_directory)
        self.model_name = model_name
        self.mode = mode

        os.makedirs(self.weight_dir, exist_ok=True)  #./save
        self.load_num = None  # epoch of model for loading
        self.branch_num = None  # directory branch number of loading pth model file.

        self.branch_list = []
        self.branch_last = None
        self.branch_new = None
        self.load_root = None
        self.load_dir = None
        self.save_name = None

        self.update(branch_num, load_num)

        self.save_name = None

    def __repr__(self):
        self.__str__()

    def __str__(self):
        print('path.DirectoryManager class instance')
        print(f'model name =========> : {self.model_name}')
        print(f'directory mode =====> : {self.mode}')
        print(f'new branch number ==> : {self.branch_num}')
        print(f'load number ========> : {self.load_num}')
        print(f'save directory =====> : {self.save_dir(make=False)}')

    def mode(self, mode):
        # fix model mode
        self.mode = mode

    def name(self, name):
        # fix model name
        self.model_name = name

    def load(self):
        # return : load .pth model weights directory.
        if self.load_dir is not None:
            return self.load_dir
        else:
            print('load directory empty.')

    def branch(self):
        # return : store branch directory
        if self.mode is 'new' or self.mode is 'load':
            return self.branch_new
        elif self.mode is 'overlay' or self.mode is 'test':
            return self.load_root
        else:
            print('Illegal mode.')

    def branch_info(self):
        return self.branch_num

    def name_info(self):
        return self.model_name

    def update(self, branch_num=None, load_num=None):  # no return
        # update the manager corresponding to the mode.
        self.branch_list = sorted(glob.glob(os.path.join(self.weight_dir, 'branch_*')))
        self.branch_last = len(self.branch_list)

        if self.mode is 'new':
            self.branch_new = os.path.join(self.weight_dir, 'branch_' + str(self.branch_last + 1))
            self.branch_num = self.branch_last + 1
        else:
            assert branch_num is not None, 'check load branch number.'
            assert load_num is not None, 'check load epoch number.'
            if self.mode is 'load':
                self.branch_num = branch_num
                self.load_num = load_num
                self.branch_new = os.path.join(self.weight_dir, 'branch_' + str(self.branch_last + 1))
                self.load_root = os.path.join(self.weight_dir, 'branch_' + str(self.branch_num))
                self.load_dir = os.path.join(self.load_root,
                                             self.model_name + '_' + 'epoch_' + str(self.load_num) + '.pth')
            elif self.mode is 'overlay' or self.mode is 'test':
                self.branch_num = branch_num
                self.load_num = load_num
                self.load_root = os.path.join(self.weight_dir, 'branch_' + str(self.branch_num))
                self.load_dir = os.path.join(self.load_root,
                                             self.model_name + '_' + 'epoch_' + str(self.load_num) + '.pth')
                # branch_xx/model_epoch_yy.pth
            else:
                print(f'Illegal mode.')

    def save_dir(self, make=True):  # using save pth file. (only)
        if self.mode is 'new':
            if make:
                os.makedirs(self.branch_new, exist_ok=True)
            self.save_name = os.path.join(self.branch_new, self.model_name+'_'+'epoch_')
            return self.save_name
        else:
            if self.mode is 'load' and self.load_num is not None:
                assert os.path.isfile(self.load_dir), "no weights file."
                if make:
                    os.makedirs(self.branch_new, exist_ok=True)
                self.save_name = os.path.join(self.branch_new, self.model_name+'_'+'epoch_')
                return self.save_name
            elif self.mode is 'overlay' and self.load_num is not None:
                assert os.path.isfile(self.load_dir), "no weights file."
                self.save_name = os.path.join(self.load_root, self.model_name+'_'+'epoch_')
                return self.save_name
            elif self.mode is 'test':
                print("Directory manager's mode is 'test' now.")
                print("you don't need to save .pth weight file.")
            else:
                print(f'Illegal directory selection inputs. check mode and load_num.')

    def last_branch(self):
        if self.branch_num is not None:
            return self.branch_num

    def graph_store(self): # graph store directory
        self.hist = os.path.join(self.branch(), 'history', 'graphs')
        os.makedirs(self.hist, exist_ok=True)
        return self.hist

    def train_predict_store(self): # training prediction store directory
        self.imgdir = os.path.join(self.branch(), 'history', 'predictions')
        os.makedirs(self.imgdir, exist_ok=True)
        return self.imgdir

    def test(self): # test base folder
        test_base = os.path.join(self.branch(), 'test')
        os.makedirs(test_base, exist_ok=True)
        return test_base

    def test_sample(self): # test prediction store directory
        os.makedirs(os.path.join(self.test(), 'sample'), exist_ok=True)
        return os.path.join(self.test(), 'sample')

    def test_graph(self): # test graph store directory
        os.makedirs(os.path.join(self.test(), 'graph'), exist_ok=True)
        return os.path.join(self.test(), 'graph')
