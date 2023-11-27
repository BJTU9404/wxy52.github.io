import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import REDCNN
from dataset import CBCTDataset
import numpy as np
import os
import random
from tensorboardX import SummaryWriter
import sys

sys.path.append('../others/program')
from Evaluation_metrics import Image_Quality_Evaluation

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

# Start TensorBoard Writer
writer = SummaryWriter()


# set the random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, args):
    num_epochs = args.epoch
    early_stopping_patience = args.early_stopping_patience
    logfile_train = open('./result/train/train_loss_log.txt', 'w')
    logfile_val = open('./result/validation/val_loss_mse_log.txt', 'w')

    # The early stopping and Learning Rate Scheduling are determined with the average MSE on the validation set.
    best_MSE = float('inf')
    epochs_no_improve = 0
    prev_best_model_path = None  # to store the path of the previously best model

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logfile_train.write('Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n')
        # After the model is trained for each epoch, use the validation set to verify the model performance
        # 1. model_train
        print('-' * 10)
        print('model_train')
        logfile_train.write('-' * 10 + '\n')
        logfile_train.write('model_train' + '\n')

        data_size_train = len(dataloader_train.dataset)
        step_train = 0
        if data_size_train % dataloader_train.batch_size != 0:
            total_step_train = data_size_train // dataloader_train.batch_size
        else:
            total_step_train = data_size_train // dataloader_train.batch_size - 1
        total_loss_train = 0

        model.train()
        for x, y in dataloader_train:
            inputs = x.to(device)  # input: SV_CBCT_img_2D
            gts = y.to(device)  # gt: FV_CBCT_img_2D
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(inputs + outputs, gts)  # MSE loss
            # backward
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()

            print('%d/%d,train_loss:%0.15f' % (step_train, total_step_train, loss.item()))
            logfile_train.write('%d/%d,train_loss:%0.15f' % (step_train, total_step_train, loss.item()) + '\n')
            step_train += 1

        avg_loss_train = total_loss_train / step_train
        print('epoch %d mean_loss_train: %0.15f\n' % (epoch, avg_loss_train))
        logfile_train.write('epoch %d mean_loss_train:%0.15f\n' % (epoch, avg_loss_train) + '\n')
        logfile_train.flush()
        writer.add_scalar('Train/Mean_Loss', avg_loss_train, epoch)  # TensorBoard

        # 2. model_validation
        print('-' * 10)
        print('model_validation')
        logfile_val.write('Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n')
        logfile_val.write('-' * 10 + '\n')
        logfile_val.write('model_validation' + '\n')

        data_size_val = len(dataloader_val.dataset)
        step_val = 0
        if data_size_val % dataloader_val.batch_size != 0:
            total_step_val = data_size_val // dataloader_val.batch_size
        else:
            total_step_val = data_size_val // dataloader_val.batch_size - 1
        total_loss_val = 0
        total_MSE_val = 0

        model.eval()
        with torch.no_grad():
            for x, y in dataloader_val:
                batch_MSE_val = 0

                inputs = x.to(device)  # input: SV_CBCT_img_2D
                gts = y.to(device)  # gt: FV_CBCT_img_2D
                # forward
                outputs = model(inputs)
                loss = criterion(inputs + outputs, gts)
                total_loss_val += loss.item()
                # *** Calculate MSE ***
                final_outputs = torch.squeeze(inputs + outputs, dim=-3).cpu().numpy()  # b,1,512,512 -> b,512,512
                gts = torch.squeeze(gts, dim=-3).cpu().numpy()
                for num in range(final_outputs.shape[0]):
                    final_output = final_outputs[num]
                    gt = gts[num]  # FV_CBCT_img_2D

                    evaluation_metrics = Image_Quality_Evaluation(gt, final_output)
                    mse = evaluation_metrics.MSE()
                    batch_MSE_val += mse
                    total_MSE_val += mse

                batch_avg_MSE_val = batch_MSE_val / final_outputs.shape[0]
                # *******

                print('%d/%d,val_loss:%0.15f' % (step_val, total_step_val, loss.item()))
                logfile_val.write('%d/%d,val_loss:%0.15f' % (step_val, total_step_val, loss.item()) + '\n')
                print('%d/%d,mean_val_MSE:%0.15f' % (step_val, total_step_val, batch_avg_MSE_val))
                logfile_val.write('%d/%d,mean_val_MSE:%0.15f' % (step_val, total_step_val, batch_avg_MSE_val) + '\n')
                step_val += 1

        avg_loss_val = total_loss_val / step_val
        print('epoch %d mean_loss_val: %0.15f' % (epoch, avg_loss_val))
        logfile_val.write('epoch %d mean_loss_val:%0.15f' % (epoch, avg_loss_val) + '\n')
        avg_MSE_val = total_MSE_val / data_size_val
        print('epoch %d mean_MSE_val: %0.15f\n' % (epoch, avg_MSE_val))
        logfile_val.write('epoch %d mean_MSE_val:%0.15f\n' % (epoch, avg_MSE_val) + '\n')
        logfile_val.flush()
        writer.add_scalar('Val/Mean_Loss', avg_loss_val, epoch)  # TensorBoard
        writer.add_scalar('Val/Mean_MSE', avg_MSE_val, epoch)  # TensorBoard

        # Learning Rate Scheduling
        scheduler.step(avg_MSE_val)

        # Early stopping
        if avg_MSE_val < best_MSE:
            best_MSE = avg_MSE_val
            epochs_no_improve = 0

            # Delete the previous best model file
            if prev_best_model_path is not None and os.path.isfile(prev_best_model_path):
                os.remove(prev_best_model_path)

            # Save the new best model and update the file path
            prev_best_model_path = './result/weight/best_model_epoch_%d.pth' % epoch
            torch.save(model.state_dict(), prev_best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print('Early stopping!')
                logfile_train.write('Early stopping!' + '\n')
                logfile_val.write('Early stopping!' + '\n')
                logfile_train.close()
                logfile_val.close()
                break

        ## ---save the weight file of the newest training epoch and delete the weight file of the previous epoch---
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_MSE': best_MSE},
                   './result/weight/model_epoch_%03d.pth' % epoch)
        if epoch > 0:
            os.remove('./result/weight/model_epoch_%03d.pth' % (epoch - 1))
        ## ------

    logfile_train.close()
    logfile_val.close()

    return None


def main(args):
    set_seed(0)

    model = REDCNN().to(device)
    model = nn.parallel.DataParallel(model)  # GPU parallel

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # default lr, beta1, beta2
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.scheduler_patience, threshold=0,
                                  verbose=True)

    dataset_train = CBCTDataset('../data/11_proj_low_split_360/training_set',
                                '../data/12_proj_clinical_split_360/training_set',
                                input_transform=input_transforms, gt_transform=gt_transforms)
    dataset_val = CBCTDataset('../data/11_proj_low_split_360/validation_set',
                              '../data/12_proj_clinical_split_360/validation_set/',
                              input_transform=input_transforms, gt_transform=gt_transforms)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=12)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=12)

    train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, args)

    return None


if __name__ == '__main__':
    # # ***complexity of the model***
    # from thop import profile
    #
    # model = Unet(1, 1)
    # input = torch.randn(1, 1, 512, 512)
    # flops, params = profile(model, inputs=(input,))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))
    # print('GFLOPs: %.1f' % (flops / 10 ** 9))
    # print('Paramsx10^6: %.1f' % (params / 10 ** 6))
    #
    # print('Total TFLOPs: %.1f' % (flops * 93 / 10 ** 12))
    # print('Total Paramsx10^6: %.1f' % (params / 10 ** 6))
    # # ******

    # ----Train----
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', type=int, default=92)
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument("--scheduler_patience", type=int, default=10)
    parse.add_argument("--early_stopping_patience", type=int, default=20)
    parse.add_argument('--gpu', type=str, default='0,1')
    args = parse.parse_args()

    # GPU Number
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
