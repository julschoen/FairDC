import torch
from dataset import *

train = get_train_loader(128)

for x,y in train:
	print(x.shape)
	print(y.shape)

	break