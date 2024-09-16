import argparse
import random
import sys
import time
import torch
import torchvision
import logging
import os

import numpy as np
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from losses import losses
# from metrics import metrics
# from models import models

# from utils import checkpoint
# from utils import log
# from utils import util
from torch import nn


def load_data(d, device: torch.device) -> torch.Tensor:

    image = d['image'].to(device)
    label = d['label'].to(device).float()
    crs = d['crs'].to(device)
    date = d['date']
    time = d['time']

    return image, label, crs, date, time


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        val = val.data
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

class SpectralAngle(nn.Module):
    def __init__(self):
        super(SpectralAngle, self).__init__()

    @staticmethod
    def forward(a, b):
        numerator = torch.sum(a * b, dim=1)
        denominator = torch.sqrt(torch.sum(a ** 2, dim=1) * torch.sum(b ** 2, dim=1))
        fraction = numerator / denominator
        sa = torch.acos(fraction)
        sa_degrees = torch.rad2deg(sa)
        sa_degrees = torch.mean(sa_degrees)
        return sa_degrees


class PeakSignalToNoiseRatio(nn.Module):
    def __init__(self, max_val=1.0):
        super(PeakSignalToNoiseRatio, self).__init__()
        self.mse = nn.MSELoss()

        self.max_val = max_val

    def forward(self, a, b):
        print(self.mse(a, b))
        return 20 * np.log10(self.max_val) - 10 * torch.log10(self.mse(a, b))


class MeanSquaredError(nn.Module):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, a, b):
        return self.mse(a, b)


class MeanSquaredErrorLoss(nn.Module):
    def __init__(self):
        super(MeanSquaredErrorLoss, self).__init__()
        self.metric = MeanSquaredError()

    def forward(self, x, x_hat):
        return self.metric(x, x_hat)


def train_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, writer):
    model.train()
    device = next(model.parameters()).device

    bpppc = model.bpppc
    cr = model.compression_ratio

    mse_metric = MeanSquaredError()
    psnr_metric = PeakSignalToNoiseRatio()
    sa_metric = SpectralAngle()

    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    sa_meter = AverageMeter()

    loop = tqdm(train_dataloader, leave=True)
    loop.set_description(f"Epoch {epoch} Training  ")
    i=-1
    for data_org in loop:
        i+=1
        image, label, crs, date, time = load_data(data_org, device)

        optimizer.zero_grad()

        data_rec = model(image)

        out_criterion = criterion(image, data_rec)

        out_criterion.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # compute metrics
        loss = out_criterion
        mse = mse_metric(image, data_rec)
        psnr = psnr_metric(image, data_rec)
        sa = sa_metric(image, data_rec)

        # update metric averages
        loss_meter.update(loss)
        mse_meter.update(mse)
        psnr_meter.update(psnr)
        sa_meter.update(sa)

        # update progress bar to show results of current batch
        loop.set_postfix(
            _loss=loss.item(),
            cr=cr,
            bpppc=bpppc,
            mse=mse.item(),
            psnr=psnr.item(),
            sa=sa.item(),
        )
        # Log losses per iteration instead of per epoch
        writer.add_scalar('Loss/train', loss.item(), epoch * len(loop) + i)
        writer.add_scalar('MSE_Loss/train', mse.item(),  epoch * len(loop) + i)
        
    # get average metrics over whole epoch
    loss_avg = loss_meter.avg.item()
    mse_avg = mse_meter.avg.item()
    psnr_avg = psnr_meter.avg.item()
    sa_avg = sa_meter.avg.item()
    writer.add_scalar('Loss/train_per_epoch', loss_avg, epoch)
    writer.add_scalar('MSE_Loss/train_per_epoch', mse_avg, epoch)

    # update progress bar to show results of whole training epoch
    loop.set_postfix(
        _loss=loss_avg,
        cr=cr,
        bpppc=bpppc,
        mse=mse_avg,
        psnr=psnr_avg,
        sa=sa_avg,
    )
    loop.update()
    loop.refresh()

    # log to tensorboard
    # log.log_epoch(writer, "train", epoch, loss_avg, cr, bpppc, mse_avg, psnr_avg, sa_avg, image, data_rec)


def test_epoch(epoch, val_dataloader, model, criterion, writer):
    model.eval()
    device = next(model.parameters()).device

    bpppc = model.bpppc
    cr = model.compression_ratio

    mse_metric = MeanSquaredError()
    psnr_metric = PeakSignalToNoiseRatio()
    sa_metric = SpectralAngle()

    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    sa_meter = AverageMeter()

    with torch.no_grad():
        loop = tqdm(val_dataloader, leave=True)
        loop.set_description(f"Epoch {epoch} Validation")
        for data_org in loop:
            image, label, crs, date, time = load_data(data_org, device)

            data_rec = model(image)

            out_criterion = criterion(image, data_rec)

            # compute metrics
            loss = out_criterion
            mse = mse_metric(image, data_rec)
            psnr = psnr_metric(image, data_rec)
            sa = sa_metric(image, data_rec)

            # update metric averages
            loss_meter.update(loss)
            mse_meter.update(mse)
            psnr_meter.update(psnr)
            sa_meter.update(sa)

            # update progress bar to show results of current batch
            loop.set_postfix(
                _loss=loss.item(),
                cr=cr,
                bpppc=bpppc,
                mse=mse.item(),
                psnr=psnr.item(),
                sa=sa.item(),
            )

        # get average metrics over whole validation set
        loss_avg = loss_meter.avg.item()
        mse_avg = mse_meter.avg.item()
        psnr_avg = psnr_meter.avg.item()
        sa_avg = sa_meter.avg.item()

        # update progress bar to show results of whole validation set
        loop.set_postfix(
            _loss=loss_avg,
            cr=cr,
            bpppc=bpppc,
            mse=mse_avg,
            psnr=psnr_avg,
            sa=sa_avg,
        )
        loop.update()
        loop.refresh()

        # log to tensorboard
        # log.log_epoch(writer, "val", epoch, loss_avg, cr, bpppc, mse_avg, psnr_avg, sa_avg, image, data_rec)

    writer.add_scalar('Loss/test', loss_avg, epoch)
    writer.add_scalar('MSE_Loss/test',mse_avg, epoch)

    return loss_avg


def train_net_multispectral(MODEL_DIR,
    net: torch.nn.Module,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    cfg: dict,
    device: torch.device,
    model_name: str,
    save: bool = True
):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    filename = os.path.join(MODEL_DIR, str(model_name) + '.pth.tar')
    
    net = net.to(device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=cfg['training']['lr'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    criterion = MeanSquaredErrorLoss()

     # Check if a checkpoint exists
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
    else:
        start_epoch = 0
        print(f"No checkpoint found at '{filename}', starting training from scratch")

    best_loss = float("inf")
    log_dir = os.path.join(MODEL_DIR, 'log',
        f"{model_name}"
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, start_epoch + cfg['training']['epochs']):

        train_epoch(
            net,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            cfg['training']['clip_max_norm'],
            writer
        )
        loss = test_epoch(epoch, data_loader_test, net, criterion, writer)
        lr_scheduler.step(loss)
        #writer.add_scalar('Loss/train', loss, epoch)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        logging.info(f"Epoch {epoch}") #- Update: {net.update(force=True)}
 
    writer.close()        

    if save:
        save_model(MODEL_DIR, net, optimizer, lr_scheduler, best_loss, epoch, model_name)

def save_model(MODEL_DIR,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    loss: float,
    epochs: int,
    name: str,
):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    filename = os.path.join(MODEL_DIR, str(name) + '.pth.tar')
    
    torch.save(
        {
            "epoch": epochs,
            "state_dict": model.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        filename,
    )



    # transform = None
    # if args.transform is not None:
    #     if args.transform == "centercrop_16x16":
    #         transform = torchvision.transforms.CenterCrop(16)
    #         random_subsample_factor = None
    #     elif args.transform == "randomcrop_16x16":
    #         transform = torchvision.transforms.RandomCrop(16)
    #         random_subsample_factor = None
    #     elif args.transform == "random_1x1":
    #         transform = None
    #         random_subsample_factor = 128
    #     elif args.transform == "random_2x2":
    #         transform = None
    #         random_subsample_factor = 64
    #     elif args.transform == "random_4x4":
    #         transform = None
    #         random_subsample_factor = 32
    #     elif args.transform == "random_8x8":
    #         transform = None
    #         random_subsample_factor = 16
    #     elif args.transform == "random_16x16":
    #         transform = None
    #         random_subsample_factor = 8
    #     elif args.transform == "random_32x32":
    #         transform = None
    #         random_subsample_factor = 4
    #     elif args.transform == "random_64x64":
    #         transform = None
    #         random_subsample_factor = 2
    # else:
    #     transform = None
    #     random_subsample_factor = None

    # parser.add_argument(
    #     "--transform",
    #     type=str,
    #     default=None,
    #     choices=[
    #         None,
    #         "centercrop_16x16", "randomcrop_16x16",
    #         "random_1x1", "random_2x2", "random_4x4", "random_8x8", "random_16x16", "random_32x32", "random_64x64"
    #     ],
    #     help="Dataset transformation (default: %(default)s)"
    # )
    # parser.add_argument(
    #     "--num-channels",
    #     type=int,
    #     default=202,
    #     help="Number of data channels, (default: %(default)s)"
    # )

    # parser.add_argument(
    #     "--loss",
    #     default="mse",
    #     choices=losses.keys(),
    #     type=str,
    #     help="Loss (default: %(default)s)",
    # )

    # parser.add_argument(
    #     "-e",
    #     "--epochs",
    #     default=2000,
    #     type=int,
    #     help="Number of epochs (default: %(default)s)",
    # )
    # parser.add_argument(
    #     "-lr",
    #     "--learning-rate",
    #     default=1e-3,
    #     type=float,
    #     help="Learning rate (default: %(default)s)",
    # )


