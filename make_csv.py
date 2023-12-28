import numpy as np
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time
import copy
import glob
import gc
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_dataset, get_network, get_eval_pool, evaluate_model, get_daparam, DiffAugment, ParamDiffAug


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--num_eval', type=int, default=25, help='the number of evaluating randomly initialized models')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--cond_path', type=str, default='result', help='path to save results')
    parser.add_argument('--attributes', type=str, default='Blond_Hair')
    parser.add_argument('--sensitive_feature', type=str, default='Male')
    parser.add_argument('--ipc', type=int, default=10)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True
    
    if args.dsa:
        if args.dataset.startswith('MNIST'):
            args.dsa_strategy ='color_crop_cutout_scale_rotate'
        elif args.dataset.startswith('HAM'):
            args.dsa_strategy ='crop_cutout_flip_scale_rotate'
        else:
            args.dsa_strategy ='color_crop_cutout_flip_scale_rotate'
            
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "", sf=True, args=args, train=False)
    model_eval_pool = get_eval_pool('M', None, None, im_size=im_size)

    metrics = {
        'accuracy': accuracy_score,
    }
    print(args.cond_path)
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    model_weights = torch.load(os.path.join(args.cond_path, f'eval_{args.ipc}ipc.pt'))['weights']
    sens_names = np.array(['Not Male', 'Male'])

    cols = ['Model', 'Acc Male', 'Acc Not Male']

    df_accs = pd.DataFrame(columns=cols)
    df_all = pd.DataFrame(columns=['Model', 'Prediction', 'Target', 'Male'])
    
    for model_eval in model_eval_pool:
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
            net_eval.load_state_dict(model_weights[model_eval][it_eval])
            pred, true, sf = evaluate_model(net_eval, testloader, args)
            sf_nums = sf
            sf = sens_names[sf.astype('int')]
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=true,
                y_pred=pred,
                sensitive_features=sf
            )


            res_grouped = metric_frame.by_group
            print(res_grouped)
            acc_male, acc_female = res_grouped['accuracy']

            row = [model_eval+f'_{it_eval}', acc_male, acc_female]
            
            df_accs.loc[len(df_accs.index)] = row

            combined_array = np.array([np.array([model_eval+f'_{it_eval}']*pred.shape[0]), pred, true, sf]).T
            new_rows = pd.DataFrame(combined_array, columns=df_all.columns)
            df_all = df_all.append(new_rows, ignore_index=True)

            
            net_eval=None
            gc.collect()
            torch.cuda.empty_cache()
            
    df_accs.to_csv(os.path.join('results_all', args.cond_path.split('/')[-1]+'_accs.csv'))
    df_all.to_csv(os.path.join('results_all', args.cond_path.split('/')[-1]+'_all.csv'))
    
            






if __name__ == '__main__':
    main()

