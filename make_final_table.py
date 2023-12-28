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
    parser.add_argument('--methods', type=str, default='kcenter_dm_ours', help='path to save results')

    args = parser.parse_args()

    methods =  args.methods.split('_')
    datasets = ['mnist', 'celeba', 'ham', 'athlets']
    
    rslt_str = ''

    for m in methods:
        rslt_str += m.upper()+' &'
        for d in datasets:
            df = pd.read_csv(os.path.join('results_all', m+'_'+d+'_accs.csv'))
            df = df[df['Model'].map(lambda x: x.startswith('ConvNet'))]
            
            if d == 'mnist':
                acc1, acc2 = df['Acc Red'].to_numpy(),  df['Acc Blue'].to_numpy()
            else:
                acc1, acc2 = df['Acc Male'].to_numpy(),  df['Acc Not Male'].to_numpy()

            diff = np.abs(acc1-acc2)
            
            rslt_str += ' $%.2f\\pm%.2f$ &'%(np.mean(diff)*100, np.std(diff)*100)

        rslt_str = rslt_str[:-2]
        if not m == m[-1]:
            rslt_str += '\\\\\\hline\n'
        

    print(rslt_str[:-2])



if __name__ == '__main__':
    main()