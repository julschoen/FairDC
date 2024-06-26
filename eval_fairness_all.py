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
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
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
            
    methods = [
        'coreset/random_'+args.dataset.lower(),
        'dsa_dm/dm_'+args.dataset.lower(),
        'dsa_dm/dc_'+args.dataset.lower(),
        'dsa_dm/dsa_'+args.dataset.lower(),
        'mtt/mtt_'+args.dataset.lower(),
        args.dataset.lower()+'_full'
    ]
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "", sf=True, args=args)
    model_eval_pool = ['ConvNet']

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    metrics = {
        'accuracy': accuracy_score,
    }

    if args.dataset.startswith('MNIST'):
        sens_names = ['Red', 'Blue']
    else:
        sens_names = ['Not '+args.sensitive_feature, args.sensitive_feature]

    sens_names = np.array(sens_names)

    cols = ['Method', 'Sensitive'] + list(metrics.keys())

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
    
    for method in methods:
        model_weights = torch.load(os.path.join(method, f'eval_{args.ipc}ipc.pt'))['weights']
        num_eval = 5 if method.startswith(args.dataset.lower()) else 25
        for model_eval in model_eval_pool:
            for it_eval in range(num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                net_eval.load_state_dict(model_weights[model_eval][it_eval])
                pred, true, sf = evaluate_model(net_eval, testloader, args)

                sf = sens_names[sf.astype('int')]
                metric_frame = MetricFrame(
                    metrics=metrics,
                    y_true=true,
                    y_pred=pred,
                    sensitive_features=sf
                )
                
                # Print the results

                res_grouped = metric_frame.by_group

                if method.startswith(args.dataset.lower()):
                    row_major = ['Full', sens_names[0]]
                    row_minor = ['Full', sens_names[1]]
                else:
                    method_name = method.split('/')[-1].split('_')[0].upper()
                    if method_name == 'DC':
                        method_name = 'GM'

                    row_major = [method_name, sens_names[0]]
                    row_minor = [method_name, sens_names[1]]


                for key in res_grouped.keys():
                    minor, major = res_grouped[key]
                    results[model_eval][key][True].append(major)
                    results[model_eval][key][False].append(minor)
                    row_major.append(major)
                    row_minor.append(minor)

                df.loc[len(df.index)] = row_major
                df.loc[len(df.index)] = row_minor
                net_eval=None
                gc.collect()
                torch.cuda.empty_cache()

    title= 'Accuracy Gap of Different Methods'
    if title.startswith('DC'):
        title='GM'

    plt.title(title)
    plt.figure(figsize=(12,24))
    sns.violinplot(data=df, x='Method', y='accuracy', hue='Sensitive', split=True, inner="quart")

    # Set the y-axis limits
    #plt.ylim(bottom=0., top=1.)

    plt.savefig(args.dataset.lower()+'_all.pdf', bbox_inches='tight')
    plt.close()

    for i, model_eval in enumerate(model_eval_pool):
        print(model_eval)
        r = results[model_eval]
        
       
        for key in r.keys():
            
            gap = np.abs(np.array(r[key][True]) - np.array(r[key][False]))
            print('%s gap of %.2f\\pm%.2f'%(key, np.mean(gap)*100, np.std(gap)*100))






if __name__ == '__main__':
    main()

