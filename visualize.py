import torch
from torch.utils.data import DataLoader, random_split
from dataloader import PointDataset
import matplotlib.pyplot as plt

dataset = PointDataset('.')
n_val = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

for batch in train_loader:
    imgs = batch['image']
    #imgs_tensor = torch.from_numpy(imgs)
    plt.imshow(imgs.permute(1, 2, 3, 0))

    location_target = batch['location']
    #location_tensor = torch.from_numpy(location_target)
    plt.imshow(location_target.permute(1, 2, 3, 0))
    break

