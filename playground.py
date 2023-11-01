import torch
from dataset import *

train = get_train_loader(128)

for x,y in train:
	print(x[1].min(), x[1].max())
	print(y[0][9])

	break