import os
from data.base_dataset import BaseDataset
import numpy as np
import torch


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        A_path = '/home2/zwang/pix2pix/dataset/xcat128/trainA/%d_40KeV.npy' %(index)
        B_path = '/home2/zwang/pix2pix/dataset/xcat128/trainB/%d_80KeV.npy' %(index)
        A_img = np.load(A_path)
        B_img = np.load(B_path)
        #A = (A_img - A_img.min()) / (A_img.max() - A_img.min()) * 2 - 1
        A = A_img / 0.2319 * 2 - 1
        #B = B_img / 0.1216 * 2 - 1  # 60eV
        B = B_img / 0.0917 * 2 - 1  # 80eV
        #B = B_img / 0.0790 * 2 - 1  # 100eV
        #B = B_img / 0.0719 * 2 - 1  # 120eV
        #B = B_img / 0.0671 * 2 - 1  # 140eV

        A = torch.from_numpy(A)
        B = torch.from_numpy(B)

        A = A.unsqueeze(0).float()
        B = B.unsqueeze(0).float()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return 140
