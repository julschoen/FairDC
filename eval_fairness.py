import numpy as np
from fairlearn.metrics import MetricFrame
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
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--cond_path', type=str, default='result', help='path to save results')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True
    args.dsa_strategy ='color_crop_cutout_flip_scale_rotate'
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "", sf=True)
    model_eval_pool = get_eval_pool('M', None, None)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    model_weights = torch.load(os.path.join(args.cond_path, 'eval.pt'))['weights']

    metrics = {
    'accuracy': accuracy_score,
    'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
    'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
    'f1_score_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    'precision_micro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro'),
    'recall_micro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro'),
    'f1_score_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'),
    # Add 'weighted' or other averages as needed
    }

    
    for model_eval in model_eval_pool:
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
            net_eval.load_state_dict(model_weights[model_eval][it_eval])
            pred, true, sf = evaluate_model(net_eval, testloader, args)

            print(true==sf)
            
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=true,
                y_pred=pred,
                sensitive_features=sf
            )

            # Print the results
            print("Metric Frame Results by Group:")
            print(metric_frame.by_group)

            # You can also get the overall metrics (not broken down by group)
            print("\nOverall Metrics:")
            print(metric_frame.overall)

            net_eval=None
            gc.collect()
            torch.cuda.empty_cache()





if __name__ == '__main__':
    main()

