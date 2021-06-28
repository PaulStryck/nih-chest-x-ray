import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from torchinfo import summary
from torchvision import transforms

from modules import collate, dataset, loss, net, trainer
from modules.dataset import (ChestXRayImageDataset, ChestXRayImages,
                             ChestXRayNPYDataset)

configs = {
    'ft_resnet_50_adam_steplr': {
        'lr': 0.0001,
        'step_size': 2,
        'gamma': 0.5,
        'epochs': 10
    },
    'ft_resnet_50_adam_redplat': {
        'lr': 0.0001,
        'factor': 0.25,
        'patience': 2
}

transform = transforms.Compose([
    transforms.RandomRotation((-7, 7)),
    transforms.RandomHorizontalFlip(p=0.25)
])

def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test_model():
    data_test  = ChestXRayNPYDataset(file      = args.data_test,
                                     transform = transform)
    test_loader  = torch.utils.data.DataLoader(data_test,
                                               batch_size=args.test_bs)

def ft_resnet_50_adam_redplat(
    config,
    device: str,
    log_interval: int,
    save_interval: int,
    out_path: str,
    train_loader,
    val_loader,
    seed: int
):
    seed_everything(seed)

    model = net.get_model(len(ChestXRayNPYDataset.labels))
    model_path = os.path.join(out_path, 'models')
    eval_path = os.path.join(out_path, 'eval')

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Print Network and training info
    summary(model, input_size=(1, 3, 244, 244))
    print('Using device: {}'.format(device))

    trainer.criterion_t = nn.BCEWithLogitsLoss()
    trainer.criterion_v = nn.BCEWithLogitsLoss()

    trainer.optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config['lr']
    )

    trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        factor=config['factor'],
        patience=config['patience']
    )

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = config['epochs'],
                log_interval  = log_interval,
                save_interval = save_interval,
                labels        = ChestXRayNPYDataset.labels,
                model_dir     = model_path,
                stage         = '0')

def ft_resnet_50_adam_steplr(
    config,
    device: str,
    log_interval: int,
    save_interval: int,
    out_path: str,
    train_loader,
    val_loader,
    seed: int
):
    seed_everything(seed)

    model = net.get_model(len(ChestXRayNPYDataset.labels))
    model_path = os.path.join(out_path, 'models')
    eval_path = os.path.join(out_path, 'eval')

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Print Network and training info
    summary(model, input_size=(1, 3, 244, 244))
    print('Using device: {}'.format(device))

    trainer.criterion_t = nn.BCEWithLogitsLoss()
    trainer.criterion_v = nn.BCEWithLogitsLoss()

    trainer.optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config['lr']
    )

    trainer.scheduler = optim.lr_scheduler.StepLR(
        trainer.optimizer,
        step_size = config['step_size'],
        gamma = config['gamma']
    )

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = config['epochs'],
                log_interval  = log_interval,
                save_interval = save_interval,
                labels        = ChestXRayNPYDataset.labels,
                model_dir     = model_path,
                stage         = '0')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train', type = str)
    parser.add_argument('--data-test', type = str)
    parser.add_argument('--save-path', type = str, help = 'Path to store models')
    parser.add_argument('--test-bs', type = int, default = 64, help = 'test batch size')
    parser.add_argument('--val-bs', type = int, default = 64, help = 'val batch size')
    parser.add_argument('--train-bs', type = int, default = 64, help = 'train batch size')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Force usage of device')
    parser.add_argument('--log-interval', type = int, default = 5, help = 'log every n batches')
    parser.add_argument('--save-interval', type = int, default = 5, help = 'save every n batches')
    # parser.add_argument('--data-frac', type = float, default = 1, help = 'use only fraction of the data')
    parser.add_argument('--folds', type=int, default=5, help='how many folds to produce')
    parser.add_argument('--val-id', type=int, default=0, help='Which fold id to use for test/val split')
    parser.add_argument('--seed', type=int, default=0, help='Seed the random generator to get reproducability')
    args = parser.parse_args()

    # Use exactly the supplied device. No error handling whatsoever
    device = torch.device(args.device)

    # Loads the entire train/val dataset with official test data left out
    data       = ChestXRayNPYDataset(file      = args.data_train,
                                     transform = transform)

    # Perform a k-fold split with random.shuffle()
    split      = dataset.k_fold_split_patient_aware(dataset = data,
                                                    folds   = args.folds,
                                                    val_id  = args.val_id)
    data_val, data_train = split


    # simple data loaders are enough, as everything is in memory anyway
    # and using a single gpu suffices. As GPU speed is not the bottleneck
    val_loader   = torch.utils.data.DataLoader(data_val,
                                               batch_size=args.val_bs,
                                               collate_fn=collate.cf)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size = args.train_bs,
                                               collate_fn = collate.cf,
                                               sampler    = RandomSampler(range(len(data_train))))


    ft_resnet_50_adam_steplr(config = configs['ft_resnet_50_adam_steplr'],
                             device = device,
                             log_interval = args.log_interval,
                             save_interval = args.save_interval,
                             out_path = os.path.join(args.save_path,
                                                     'resnet_50_adam_steplr'),
                             train_loader = train_loader,
                             val_loader = val_loader,
                             seed = args.seed)

    ft_resnet_50_adam_redplat(config = configs['ft_resnet_50_adam_redplat'],
                              device = device,
                              log_interval = args.log_interval,
                              save_interval = args.save_interval,
                              out_path = os.path.join(args.save_path,
                                                      'resnet_50_adam_redplat'),
                              train_loader = train_loader,
                              val_loader = val_loader,
                              seed = args.seed)

if __name__ == "__main__":
    main()
