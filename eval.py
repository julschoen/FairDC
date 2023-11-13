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
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "")
    model_eval_pool = get_eval_pool('M', None, None)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    model_weights = dict() # record performances of all experiments
    for key in model_eval_pool:
        model_weights[key] = []

    file = glob.glob(os.path.join(args.cond_path, '*.pt'))[-1]

    data = torch.load(file)['data']

    for image_syn, label_syn in data:
        image_syn = torch.clamp(image_syn, -1, 1)
        for model_eval in model_eval_pool:
            accs = []
            weights = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                accs.append(acc_test)
                weights.append(net_eval.state_dict())
                net_eval=None
                gc.collect()
                torch.cuda.empty_cache()
            accs_all_exps[model_eval] += accs
            model_weights[model_eval] += weights


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Evaluate %d random %s on %d SynSets, %.2f\\pm%.2f'%(len(accs)/len(data), key, len(data), np.mean(accs)*100, np.std(accs)*100))

    torch.save({
            'weights': model_weights,
            'accs': accs_all_exps
        }, os.path.join(args.cond_path, 'eval.pt'))



if __name__ == '__main__':
    main()


