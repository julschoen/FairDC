import torch
from dataset import *
import matplotlib.pyplot as plt

train = get_train_loader(200)

def plot_images(images_tensor): 

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flatten()):
        if i >= images_tensor.size(0):  # Check if we have fewer images than grid spaces
            ax.axis('off')  # Hide empty subplots
            continue
        # Permute the tensor dimensions to (H, W, C) for plotting
        image = images_tensor[i].permute(1, 2, 0)
        image = (image+1)/2
        # Convert tensor to numpy for plotting
        ax.imshow(image.numpy())
        ax.axis('off')  # Hide the axes
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

for x,y in train:
	xs = torch.tensor([])
	for c in range(10):
		xs = torch.concat((x[y==c][:10] ,xs),dim=0)
	plot_images(xs)
	break