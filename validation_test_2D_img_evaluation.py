import pickle
import os
import numpy as np
import sys
import argparse

sys.path.append('../others/program/')
from Evaluation_metrics import Image_Quality_Evaluation


def img_quality_evaluation(args):
    logfile = open('./result/%s/%s_2D_img_evaluation.txt' % (args.dataset.split('_')[0], args.dataset), 'w')
    print("Evaluation in Image domain:")
    logfile.write("Evaluation in Image domain:" + '\n')

    metrics = {'MSE': []}
    case_list = os.listdir('../data/1_fdk_clean_2D_slice/%s' % args.dataset)
    slice_num = len(os.listdir(
        '../data/1_fdk_clean_2D_slice/%s/%s' % (args.dataset, case_list[0])))  # same length for each case

    for k, case in enumerate(case_list):
        print('%d-%s' % (k, case))
        logfile.write('%d-%s' % (k, case) + '\n')

        for i in range(slice_num):
            with open('../data/1_fdk_clean_2D_slice/%s/%s/%03d.npy' % (args.dataset, case, i), 'rb') as f:
                reference = np.load(f)
            with open('./result/%s/model_output/%s/%03d.npy' % (args.dataset.split('_')[0], case, i), 'rb') as f:
                predict = np.load(f)

            evaluation_metrics = Image_Quality_Evaluation(reference, predict)
            MSE = evaluation_metrics.MSE()

            metrics['MSE'].append(MSE)

            print('slice_%02d' % i)
            print('MSE = %.10f' % MSE)
            logfile.write('slice_%02d' % i + '\n')
            logfile.write('MSE = %.10f' % MSE + '\n')
            logfile.flush()

        print()
        logfile.write('\n')

    print("Statistical result")
    logfile.write('Statistical result' + '\n')

    print('MSE %.10f±%.10f' % (np.array(metrics['MSE']).mean(), np.array(metrics['MSE']).std(ddof=1)))
    logfile.write('MSE %.10f±%.10f' % (np.array(metrics['MSE']).mean(), np.array(metrics['MSE']).std(ddof=1)) + '\n')



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default='validation_set')
    args = parse.parse_args()

    img_quality_evaluation(args)
