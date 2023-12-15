import torch


model_weights = torch.load('celeba_full/eval.pt')['weights']
conv_weights = torch.load('celeba_full/eval_conv.pt')['weights']

accs_all_exps = torch.load('celeba_full/eval.pt')['accs']
conv_accs = torch.load('celeba_full/eval_conv.pt')['accs']

model_weights['ConvNet'] = conv_weights['ConvNet']
accs_all_exps['ConvNet'] = conv_accs['ConvNet']

torch.save({
    'weights': model_weights,
    'accs': accs_all_exps
}, 'celeba_full/eval_50ipc.pt')