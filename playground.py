import torch
from dataset import *
import matplotlib.pyplot as plt

train = get_train_loader(128)

for x,y in train:
	print(x.shape)
	print(y[0])

	plt.imshow((x[0].permute(1,2,0).cpu().numpy()+1)/2)
	plt.savefig('../im.png')
	break