import numpy as np
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio

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
def false_positive_rate(y_true, y_pred, label):
    # Confusion matrix: rows are true classes, columns are predicted classes
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    # FP: Sum of column for 'label', excluding true positives (diagonal)
    FP = cm[:, label].sum() - cm[label, label]
    # True Negatives (TN): Sum of all cells excluding row and column for 'label'
    TN = cm.sum() - cm[label, :].sum() - cm[:, label].sum() + cm[label, label]
    return FP / (FP + TN) if (FP + TN) > 0 else 0

def false_negative_rate(y_true, y_pred, label):
    # Confusion matrix: rows are true classes, columns are predicted classes
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    # FN: Sum of row for 'label', excluding true positives (diagonal)
    FN = cm[label, :].sum() - cm[label, label]
    # True Positives (TP)
    TP = cm[label, label]
    return FN / (FN + TP) if (FN + TP) > 0 else 0


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--cond_path', type=str, default='result', help='path to save results')
    parser.add_argument('--attributes', type=str, default='Blond_Hair')
    parser.add_argument('--sensitive_feature', type=str, default='Male')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True
    args.dsa_strategy ='color_crop_cutout_flip_scale_rotate'
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "", sf=True, args=args)
    model_eval_pool = get_eval_pool('M', None, None)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    model_weights = torch.load(os.path.join(args.cond_path, 'eval.pt'))['weights']

    metrics = {
        'accuracy': accuracy_score,
        #'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
        #'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
        #'f1_score_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        #'precision_micro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro'),
        #'recall_micro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro'),
        #'f1_score_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
    }

    results = dict()
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

            if args.dataset.startswith('MNIST'):
                sf = true == sf
            
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=true,
                y_pred=pred,
                sensitive_features=sf
            )
            eod = equalized_odds_difference(true, pred, sensitive_features=sf)
            print(eod)
            eods.append(eod)
            eor = equalized_odds_ratio(true, pred, sensitive_features=sf)
            print(eor)
            eors.append(eors)

            dpd = demographic_parity_difference(true, pred, sensitive_features=sf)
            print(dpd)
            dpds.append(dpd)
            dpr = demographic_parity_ratio(true, pred, sensitive_features=sf)
            print(dpr)
            dprs.append(dprs)

            # Print the results
            res_grouped = metric_frame.by_group
            for key in res_grouped.keys():
                minor, major = res_grouped[key]
                results[model_eval][key][True].append(major)
                results[model_eval][key][False].append(minor)


            net_eval=None
            gc.collect()
            torch.cuda.empty_cache()

    for i, model_eval in enumerate(model_eval_pool):
        print(model_eval)
        r = results[model_eval]
        eor = eors[i*args.num_eval:(i+1)*args.num_eval]
        eod = eods[i*args.num_eval:(i+1)*args.num_eval]
        dpr = dprs[i*args.num_eval:(i+1)*args.num_eval]
        dpd = dpds[i*args.num_eval:(i+1)*args.num_eval]
        print('EOR %.2f\\pm%.2f'%(key, np.mean(eor), np.std(eor)))
        print('EOD %.2f\\pm%.2f'%(key, np.mean(eod), np.std(eod)))
        print('DPR %.2f\\pm%.2f'%(key, np.mean(dpr), np.std(dpr)))
        print('DPD %.2f\\pm%.2f'%(key, np.mean(dpd), np.std(dpd)))
        for key in r.keys():
            gap = np.abs(np.array(r[key][True]) - np.array(r[key][False]))
            print('%s gap of %.2f\\pm%.2f'%(key, np.mean(gap)*100, np.std(gap)*100))






if __name__ == '__main__':
    main()

