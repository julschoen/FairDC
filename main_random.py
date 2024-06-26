import os
import copy
import argparse
import numpy as np
import torch
from utils import get_dataset
import torchvision.utils as vutils

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--save_path', type=str, default='random')
    parser.add_argument('--attributes', type=str, default='Blond_Hair')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, "", args=args)
    data_save = []

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0)
    labels_all = torch.tensor(labels_all, dtype=torch.long)

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data

        vutils.save_image(
            vutils.make_grid(image_syn.detach(), nrow=args.ipc, padding=2, normalize=True)
            , os.path.join(args.save_path,'ims_%s_%dipc_%dexp.png'%(args.dataset, args.ipc, exp)))
        
        data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
        torch.save({'data': data_save}, os.path.join(args.save_path, 'res_%s_%dipc.pt'%(args.dataset, args.ipc)))


if __name__ == '__main__':
    main()


