import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from unet import Unet
from dataset import CBCTDataset
import numpy as np
import glob
import os
import pickle

# 模型训练完毕后，输入验证集与测试集的FV CBCT 2D图像，得到抑制伪影、噪声后的2D图像，保存并服务于后续图像质量评价

# 是否使用cuda
device = torch.device('cuda')

# input_image_transform
input_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# ground truth_image_transform
gt_transforms = transforms.Compose([
    transforms.ToTensor()
])


def model_output_image(args):
    model = Unet(1, 1).to(device)
    model.load_state_dict(torch.load('../FBPConvNet/result/weight/best_model_epoch_29.pth'))
    dataset = CBCTDataset('../data/2_fdk_clinical_dose_2D_slice/%s' % args.dataset,
                          '../data/1_fdk_clean_2D_slice/%s' % args.dataset,
                          input_transform=input_transforms, gt_transform=gt_transforms, mode='test')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=10)

    model.eval()

    with torch.no_grad():
        for batch_num, (x, y, x_path, y_path) in enumerate(dataloader):  # x: image preprocessing
            print(batch_num)
            # model prediction
            inputs = x.to(device)  # input: SV_CBCT_img_2D
            outputs = model(inputs)  # residual
            final_outputs = inputs + outputs  # SV_CBCT_img_2D + residual
            final_outputs = torch.squeeze(final_outputs).cpu().numpy()
            # save reduction streak artifacts & noise 2D CBCT image
            for i, path in enumerate(x_path):
                case_slice = path.split(args.dataset)[-1]  # '/2021-03-16_134545_FINISHED_Head/30.pkl'
                os.makedirs('./result/validation/model_output%s'%case_slice[:5], exist_ok=True)
                np.save('./result/%s/model_output%s' % (args.dataset.split('_')[0], case_slice), final_outputs[i])

    return None


if __name__ == '__main__':
    # ---Validation & test----
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='validation_set')
    parse.add_argument('--batch_size', type=int, default=100)
    parse.add_argument('--gpu', type=str, default='0')
    args = parse.parse_args()

    # GPU Number
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_output_image(args)
