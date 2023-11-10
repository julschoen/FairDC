import os
import time
import copy
import glob
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, TensorDataset, epoch, DiffAugment, ParamDiffAug

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='mtt')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--cond_path', type=str, default='result', help='path to save results')
    args = parser.parse_args()

    data_save = []

    for run in os.path.listdir(args.cond_path):
        image_syn = torch.load(os.path.join(args.cond_path, 'images_best.pt'))
        label_syn = torch.load(os.path.join(args.cond_path, 'labels_best.pt'))
        data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
    torch.save({'data': data_save, 'accs_all_exps': []}, os.path.join(args.cond_path, 'res_%s_%s_%dipc.pt'%(args.method, args.dataset, args.ipc)))



if __name__ == '__main__':
    main()


