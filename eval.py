import torch
from tqdm import tqdm
from utils.loss import dice_coeff

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, location = batch['image'], batch['location']
            imgs = imgs.to(device=device, dtype=torch.float32)
            location_target = location.to(device=device, dtype=mask_type)

            with torch.no_grad():
                location_pred = net(imgs)

            location_pred = torch.sigmoid(location_pred)
            location_pred = (location_pred > 0.5).float()
            tot += dice_coeff(location_pred, location_target).item()
            pbar.update()

    net.train()
    return tot / n_val