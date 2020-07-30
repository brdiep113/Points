import torch
from network.model import PointDetectorNet
from utils.loss import location_loss
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from utils.dataloader import PointDataset
from torch.utils.data import DataLoader, random_split
from eval import eval_net

import logging
import os


def train_net(net, val_percent=0.1, batch_size=128, lr=0.001, epochs=5, save_cp=True):
    dataset = PointDataset()
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
        ''')

    for epoch in range(epochs):

        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                points = batch['location']

                imgs = imgs.to(device=device, dtype=torch.float32)
                location_target = points.to(device=device)

                location_pred = net(imgs)

                loss = location_loss(location_pred, location_target)
                epoch_loss += loss.item() * img.size(0)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    writer.add_images('location/true', location_target, global_step)
                    writer.add_images('location/pred', torch.sigmoid(location_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir('checkpoints')
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                        f'checkpoints/CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = PointDetectorNet()
    net.to(device=device)
    train_net(net)

    torch.save(net.state_dict(), 'checkpoints/final_model_saved.pth')