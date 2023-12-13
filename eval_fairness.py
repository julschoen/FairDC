import numpy as np
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio
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
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
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
    args.dsa_strategy ='color_crop_cutout_flip_scale_rotate'
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "", sf=True, args=args)
    model_eval_pool = get_eval_pool('M', None, None, im_size=im_size)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    model_weights = torch.load(os.path.join(args.cond_path, f'eval_{args.ipc}ipc.pt'))['weights']

    metrics = {
        'accuracy': accuracy_score,
    }

    if args.dataset.startswith('MNIST'):
        sens_names = ['Blue', 'Red']
    else:
        sens_names = ['Not '+args.sensitive_feature, args.sensitive_feature]

    sens_names = np.array(sens_names)

    cols = ['Model', 'Sensitive'] + list(metrics.keys())

    df = pd.DataFrame(columns=cols)

    results= dict()
    for model in model_eval_pool:
        d = {}
        for key in metrics.keys():

            d[key] = {
                    True: [],
                    False: []
                }
            
        results[model] = d

    eors = []
    eods = []
    dprs = []
    dpds = []
    
    for model_eval in model_eval_pool:
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
            net_eval.load_state_dict(model_weights[model_eval][it_eval])
            pred, true, sf = evaluate_model(net_eval, testloader, args)

            #if args.dataset.startswith('MNIST'):
            #    sf = np.minimum(true//3,2) == sf
            sf = sens_names[sf.astype('int')]
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=true,
                y_pred=pred,
                sensitive_features=sf
            )
            if args.dataset.startswith('CelebA'):
                eod = equalized_odds_difference(true, pred, sensitive_features=sf)
                eods.append(eod)
                eor = equalized_odds_ratio(true, pred, sensitive_features=sf)
                eors.append(eor)

                dpd = demographic_parity_difference(true, pred, sensitive_features=sf)
                dpds.append(dpd)
                dpr = demographic_parity_ratio(true, pred, sensitive_features=sf)
                dprs.append(dpr)

            # Print the results
            res_grouped = metric_frame.by_group
            for key in res_grouped.keys():
                major, minor = res_grouped[key]
                results[model_eval][key][True].append(major)
                results[model_eval][key][False].append(minor)

            df.loc[len(df.index)] = [model_eval, 'Blue', major]
            df.loc[len(df.index)] = [model_eval, 'Red', minor]
            net_eval=None
            gc.collect()
            torch.cuda.empty_cache()

    sns.violinplot(data=df, x='Model', y='accuracy', hue='Sensitive', split=True, inner="quart")
    plt.savefig('sns.png')
    plt.close()

    for i, model_eval in enumerate(model_eval_pool):
        print(model_eval)
        r = results[model_eval]
        if args.dataset.startswith('CelebA'):
            eor = np.array(eors[i*args.num_eval:(i+1)*args.num_eval])
            eod = np.array(eods[i*args.num_eval:(i+1)*args.num_eval])
            dpr = np.array(dprs[i*args.num_eval:(i+1)*args.num_eval])
            dpd = np.array(dpds[i*args.num_eval:(i+1)*args.num_eval])
            print('EOR %.2f\\pm%.2f'%(np.mean(eor), np.std(eor)))
            print('EOD %.2f\\pm%.2f'%(np.mean(eod), np.std(eod)))
            print('DPR %.2f\\pm%.2f'%(np.mean(dpr), np.std(dpr)))
            print('DPD %.2f\\pm%.2f'%(np.mean(dpd), np.std(dpd)))
       
        for key in r.keys():
            
            gap = np.abs(np.array(r[key][True]) - np.array(r[key][False]))
            print('%s gap of %.2f\\pm%.2f'%(key, np.mean(gap)*100, np.std(gap)*100))






if __name__ == '__main__':
    main()

