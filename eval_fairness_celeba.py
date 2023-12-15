import numpy as np
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
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
    parser.add_argument('--dataset', type=str, default='CelebA', help='dataset')
    parser.add_argument('--num_eval', type=int, default=25, help='the number of evaluating randomly initialized models')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--cond_path', type=str, default='result', help='path to save results')
    parser.add_argument('--attributes', type=str, default='Blond_Hair')
    parser.add_argument('--sensitive_feature', type=str, default='Male')
    parser.add_argument('--ipc', type=int, default=50)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True

    print('eval fairness')
    
    if args.dataset.startswith('MNIST'):
        args.dsa_strategy ='color_crop_cutout_scale_rotate'
    else:
        args.dsa_strategy ='color_crop_cutout_flip_scale_rotate'
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "", sf=True, args=args, train=False)
    model_eval_pool = ['ConvNet']

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    model_weights = torch.load(os.path.join(args.cond_path, f'eval_{args.ipc}ipc.pt'))['weights']

    metrics = {
        'accuracy': accuracy_score,
    }

    if args.dataset.startswith('MNIST'):
        sens_names = ['Red', 'Blue']
    else:
        sens_names = ['Not '+args.sensitive_feature, args.sensitive_feature]

    sens_names = np.array(sens_names)

    cols = ['Model', 'Sensitive'] + list(metrics.keys()) + ['TPR', 'TNR', 'FPR', 'FNR']

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

            sf = sens_names[sf.astype('int')]
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=true,
                y_pred=pred,
                sensitive_features=sf
            )
            
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
            row_major = [model_eval, sens_names[0]]
            row_minor = [model_eval, sens_names[1]]

            for key in res_grouped.keys():
                minor, major = res_grouped[key]
                results[model_eval][key][True].append(major)
                results[model_eval][key][False].append(minor)
                row_major.append(major)
                row_minor.append(minor)

            pred = list(map(int, pred))
            true = list(map(int, true))
            print(pred[sf==0],true[sf==0])
            tn, fp, fn, tp = confusion_matrix(pred[sf==0], true[sf==0]).ravel()
            row_major.append(tp)
            row_major.append(tn)
            row_major.append(fp)
            row_major.append(fn)
            results[model_eval]['TPR'][True].append(tp)
            results[model_eval]['TNR'][True].append(tn)
            results[model_eval]['FPR'][True].append(fp)
            results[model_eval]['FNR'][True].append(fn)

            tn, fp, fn, tp = confusion_matrix(pred[sf==1], true[sf==1]).ravel()
            row_minor.append(tp)
            row_minor.append(tn)
            row_minor.append(fp)
            row_minor.append(fn)
            results[model_eval]['TPR'][False].append(tp)
            results[model_eval]['TNR'][False].append(tn)
            results[model_eval]['FPR'][False].append(fp)
            results[model_eval]['FNR'][False].append(fn)

            df.loc[len(df.index)] = row_major
            df.loc[len(df.index)] = row_minor
            net_eval=None
            gc.collect()
            torch.cuda.empty_cache()

    for m in metrics.keys():
        title= args.cond_path.split('/')[-1].split('_')[0].upper()
        if title.startswith('DC'):
            title='GM'
        plt.title(title)
        sns.violinplot(data=df, x='Model', y=m, hue='Sensitive', split=True, inner="quart")
        plt.savefig(args.cond_path.split('/')[-1]+'_'+m+'.pdf', bbox_inches='tight')
        plt.close()

    for model_eval in model_eval_pool:
        r = results[model_eval]
        eor = np.array(eors[i*args.num_eval:(i+1)*args.num_eval])
        eod = np.array(eods[i*args.num_eval:(i+1)*args.num_eval])
        dpr = np.array(dprs[i*args.num_eval:(i+1)*args.num_eval])
        dpd = np.array(dpds[i*args.num_eval:(i+1)*args.num_eval])
        print('EOR %.2f\\pm%.2f'%(np.mean(eor), np.std(eor)))
        print('EOD %.2f\\pm%.2f'%(np.mean(eod), np.std(eod)))
        print('DPR %.2f\\pm%.2f'%(np.mean(dpr), np.std(dpr)))
        print('DPD %.2f\\pm%.2f'%(np.mean(dpd), np.std(dpd)))
           
        for key in r.keys():
            minor = np.array(r[key][False])
            major = np.array(r[key][True])
            print('%s Sensitive %.2f\\pm%.2f Not %.2f\\pm%.2f'%(key, np.mean(minor), np.std(minor), np.mean(major), np.std(major)))






if __name__ == '__main__':
    main()

