from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
from torchvision import transforms


# No data augmentation

def make_dataset(input_path, gt_path):
    """
    input: sparse-view CBCT FDK reconstruction image (2D)
    gt: full-view CBCT FDK reconstruction image (2D)
    """
    imgs = []
    case_list = os.listdir(input_path)
    for case in case_list:
        slice_list = os.listdir('%s/%s' % (input_path, case))
        for slice in slice_list:
            sv_img_2D_path = '%s/%s/%s' % (input_path, case, slice)
            fv_img_2D_path = '%s/%s/%s' % (gt_path, case, slice)
            imgs.append((sv_img_2D_path, fv_img_2D_path))

    return imgs


class CBCTDataset(Dataset):
    def __init__(self, input_path, gt_path, input_transform=None, gt_transform=None, mode='train'):
        imgs = make_dataset(input_path, gt_path)
        self.imgs = imgs
        self.input_transform = input_transform  # input
        self.gt_transform = gt_transform  # ground truth
        self.mode = mode  # training process or test process

    def __getitem__(self, index):
        sv_img_2D_path, fv_img_2D_path = self.imgs[index]

        with open(sv_img_2D_path, 'rb') as f:
            sv_img_2D = np.load(f)  # input   float32

        with open(fv_img_2D_path, 'rb') as f:
            fv_img_2D = np.load(f)  # gt

        sv_img_2D = self.input_transform(sv_img_2D)
        fv_img_2D = self.gt_transform(fv_img_2D)

        if self.mode == 'train':
            return sv_img_2D, fv_img_2D
        else:
            return sv_img_2D, fv_img_2D, sv_img_2D_path, fv_img_2D_path  # return paths too

    def __len__(self):
        return len(self.imgs)
