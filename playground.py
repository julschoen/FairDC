import torchfrom utils
import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from carbontracker.tracker import CarbonTracker


def main():
    parser.add_argument('--dataset', type=str, default='CelebA', help='dataset')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--attributes', type=str, default='Blond_Hair')

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    blond = 0
    not_blond = 0

    for _, y in dst_train:
    	blond += y.sum()
    	not_blond += (len(y)-y.sum())

    print(blond, not_blond)