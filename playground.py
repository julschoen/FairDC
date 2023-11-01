import torch
from dataset import *

train = get_train_loader(128)

for x,y in train:
	print(x[0].min(), x[0].max())
	print(y[0])

	break