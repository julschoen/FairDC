import torch

model_weights = torch.load('celeba_full/eval_50ipc.pt')
print(model_weights)

model_weights = torch.load('celeba_full/eval.pt')['weights']
conv_weights = torch.load('celeba_full/eval_conv.pt')['weights']
print(model_weights)
accs_all_exps = torch.load('celeba_full/eval.pt')['accs']
conv_accs = torch.load('celeba_full/eval_conv.pt')['accs']

for key in conv_weights.keys():
	model_weights[key] = conv_weights[key]

for key in conv_accs.keys():
	accs_all_exps[key] = conv_accs[key]

torch.save({
    'weights': model_weights,
    'accs': accs_all_exps
}, 'celeba_full/eval_50ipc.pt')