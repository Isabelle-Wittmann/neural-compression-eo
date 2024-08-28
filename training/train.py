import logging
import torch
import glob
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from compressai.losses import RateDistortionLoss
from .utils import *
from utils import load_data
from evaluation.neural import *
import shutil
logging.basicConfig(level=logging.INFO)

def train_one_epoch(
    model: torch.nn.Module,
    criterion: RateDistortionLoss,
    data_loader_train: DataLoader,
    optimizer: torch.optim.Optimizer,
    aux_optimizer: torch.optim.Optimizer,
    epoch: int,
    clip_max_norm: float,
    dataset_name: str,
    writer: SummaryWriter,
    lr_scheduler
) -> None:
    model.train()
    device = next(model.parameters()).device
    loss_ls, mse_loss_ls, bpp_loss_ls, aux_loss_ls = [], [], [], []
    for i, d in enumerate(data_loader_train):


        image, label, crs, date, time = load_data(d, dataset_name, device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(image, None, crs)
        out_criterion = criterion(out_net, image)
        
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            logging.info(
                f"Train epoch {epoch}: [{i * len(image)}/{len(data_loader_train.dataset)} "
                f"({100. * i / len(data_loader_train):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.3f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.3f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                f'Aux loss: {aux_loss.item():.2f}'
            )

        loss_ls += [out_criterion["loss"].item()]
        mse_loss_ls += [out_criterion["mse_loss"].item()]
        bpp_loss_ls += [out_criterion["bpp_loss"].item()]
        aux_loss_ls += [aux_loss.item()]

        # Log losses per iteration instead of per epoch
        writer.add_scalar('Loss/train', out_criterion["loss"].item(), epoch * len(data_loader_train) + i)
        writer.add_scalar('MSE_Loss/train', out_criterion["mse_loss"].item(), epoch * len(data_loader_train) + i)
        writer.add_scalar('Bpp_Loss/train', out_criterion["bpp_loss"].item(), epoch * len(data_loader_train) + i)
        writer.add_scalar('Aux_Loss/train', aux_loss.item(), epoch * len(data_loader_train) + i)

    avg_loss = sum(loss_ls) / len(loss_ls) if loss_ls else 0
    avg_mse_loss = sum(mse_loss_ls) / len(mse_loss_ls) if mse_loss_ls else 0
    avg_bpp_loss = sum(bpp_loss_ls) / len(bpp_loss_ls) if bpp_loss_ls else 0
    avg_aux_loss = sum(aux_loss_ls) / len(aux_loss_ls) if aux_loss_ls else 0

    writer.add_scalar('Loss/train_per_epoch', avg_loss, epoch)
    writer.add_scalar('MSE_Loss/train_per_epoch', avg_mse_loss, epoch)
    writer.add_scalar('Bpp_Loss/train_per_epoch', avg_bpp_loss, epoch)
    writer.add_scalar('Aux_Loss/train_per_epoch', avg_aux_loss, epoch)
    writer.add_scalar('Learning_Rate', lr_scheduler.get_last_lr(), epoch)
    writer.add_scalar('Learning_Rate_Aux', lr_scheduler.get_last_lr(), epoch)


def test_epoch(
    epoch: int,
    data_loader_test: DataLoader,
    model: torch.nn.Module,
    criterion: RateDistortionLoss,
    dataset_name: str,
    writer: SummaryWriter
) -> float:
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in data_loader_test:
            image, label, crs, date, time = load_data(d, dataset_name, device)

            out_net = model(image, None, crs)
            out_criterion = criterion(out_net, image)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    logging.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE loss: {mse_loss.avg:.3f} | "
        f"Bpp loss: {bpp_loss.avg:.2f} | "
        f"Aux loss: {aux_loss.avg:.2f}"
    )
    writer.add_scalar('Loss/test', loss.avg, epoch)
    writer.add_scalar('MSE_Loss/test', mse_loss.avg, epoch)
    writer.add_scalar('Bpp_Loss/test', bpp_loss.avg, epoch)
    writer.add_scalar('Aux_Loss/test', aux_loss.avg, epoch)

    return loss.avg

def train_net(MODEL_DIR,
    net: torch.nn.Module,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    cfg: dict,
    device: torch.device,
    model_name: str,
    save: bool = True,
    checkpoint_interval: int = 5,
    keep_last_n_checkpoints: int = 1
):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    filename = os.path.join(MODEL_DIR, str(model_name) + '.pth.tar')
    
    net = net.to(device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(
        net,
        cfg['training']['optim'],
        cfg['training']['optim_aux'],
        cfg['training']['lr'],
        cfg['training']['lr_aux'],
    )
   
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg['training']['epochs'])
    criterion = RateDistortionLoss(cfg['training']['lmbda'])

    # Check if a checkpoint exists
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
    else:
        start_epoch = 0
        best_loss = float("inf")
        print(f"No checkpoint found at '{filename}', starting training from scratch")

    best_loss = float("inf")
    log_dir = os.path.join(MODEL_DIR, 'log',
        f"{model_name}"
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, start_epoch + cfg['training']['epochs']):

        train_one_epoch(
            net,
            criterion,
            data_loader_train,
            optimizer,
            aux_optimizer,
            epoch,
            cfg['training']['clip_max_norm'],
            cfg['dataset']['name'],
            writer,
            lr_scheduler
        )
        loss = test_epoch(epoch, data_loader_test, net, criterion, cfg['dataset']['name'],writer)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        if is_best:
            best_loss = loss
        
        if epoch % (checkpoint_interval*2) == 0:
            save_sample_reconstruction(device, data_loader_test.dataset[145], net, os.path.join(MODEL_DIR, 'reconstructions', model_name + '_epoch' + str(epoch)))

        if epoch % checkpoint_interval == 0 or is_best:
            checkpoint_filename = os.path.join(MODEL_DIR, f"{model_name}_epoch_{epoch}.pth.tar")
            save_model(checkpoint_filename, net, optimizer, aux_optimizer, lr_scheduler, best_loss, epoch, model_name)
            if is_best:
                best_model_filename = os.path.join(MODEL_DIR, f"{model_name}_best.pth.tar")
                save_model(best_model_filename, net, optimizer, aux_optimizer, lr_scheduler, best_loss, epoch, model_name)

            cleanup_checkpoints(MODEL_DIR, model_name, keep_last_n_checkpoints)

        logging.info(f"Epoch {epoch} - Update: {net.update(force=True)}")

    writer.close()        

    if save:
        save_model(MODEL_DIR, net, optimizer, aux_optimizer, lr_scheduler, best_loss, epoch, model_name)

def save_model(MODEL_DIR,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    aux_optimizer: torch.optim.Optimizer,
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
            "aux_optimizer": aux_optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        filename,
    )

def cleanup_checkpoints(MODEL_DIR, model_name, keep_last_n):
    """Deletes older checkpoints, keeping only the most recent `keep_last_n` checkpoints."""
    checkpoint_files = sorted(
        glob.glob(os.path.join(MODEL_DIR, f"{model_name}_epoch_*.pth.tar")),
        key=os.path.getmtime
    )

    if len(checkpoint_files) > keep_last_n:
        for file_to_delete in checkpoint_files[:-keep_last_n]:
            try:
                if os.path.isfile(file_to_delete): 
                    os.remove(file_to_delete)
                    print(f"Deleted old checkpoint: {file_to_delete}")
                elif os.path.isdir(file_to_delete):
                    shutil.rmtree(file_to_delete)
                    print(f"Deleted old directory: {file_to_delete}")
                else:
                    print(f"Skipping deletion, not a file or directory: {file_to_delete}")
            except OSError as e:
                print(f"Error deleting file or directory {file_to_delete}: {e}")


def save_sample_reconstruction(device, image, model, path):

        image, label, crs, date, time = load_data(image, True, device)
        rv = model(image.unsqueeze(0), None, crs.unsqueeze(0))
        decompressed = rv['x_hat']

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        img = image.squeeze()
        img_dec = decompressed.squeeze()

        img = transforms.ToPILImage()(np.clip(img.cpu().detach()/ 0.3, 0, 1))
        axes[0].imshow(img)
        axes[0].title.set_text('Original')

        img_dec = transforms.ToPILImage()(np.clip(img_dec.cpu().detach()/ 0.3, 0, 1))
        axes[1].imshow(img_dec)
        axes[1].title.set_text('Reconstruction')

        plt.savefig(path + '_sample_reconstruction.png')
        plt.close()