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
        'epochs': 5
    },
    'ft_resnet_50_adam_redplat': {
        'lr': 0.00005,
        'epochs': 5,
        'factor': 0.25,
        'patience': 2
    },
    'ft_resnet_50_adam_exponential': {
        'lr': 0.0001,
        'epochs': 10,
        'gamma': 0.1
    },
    'ft_resnet_50_adam_steplr_staged': [
        { 'lr': 1e-4,
           'epochs': 2,
           'step_size': 2},
        { 'lr': 1e-4,
           'epochs': 5,
           'step_size': 2},
        { 'lr': 5e-4,
           'epochs': 4,
           'step_size': 2},
        { 'lr': 1e-3,
           'epochs': 3,
           'step_size': 2}
        ]
}

# transform = transforms.Compose([
#      transforms.RandomRotation((-7, 7)),
#      transforms.RandomHorizontalFlip(p=0.25)
# ])

# random vertical flip with probabilty of 50%
def transform(x: np.ndarray):
    if random.uniform(0, 1) < 0.5:
        return np.flip(x, axis=2)
    return x


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test_model(model, model_weights_path, test_loader, device):
    model.load_state_dict(torch.load(model_weights_path))

    trainer.val_epoch(device = device,
                      loader = test_loader,
                      model = model,
                      labels = ChestXRayNPYDataset.labels,
                      epochs_till_now = 0,
                      final_epoch = 0,
                      log_interval = 20)
def ft_resnet_50_adam_steplr_staged(
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

    print('------ STAGE 0 --------')
    for name, param in model.named_parameters():
        if ('layer2' in name) or ('layer3' in name) or ('layer4' in name) or ('fc' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainer.optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config[0]['lr']
    )
    trainer.scheduler = optim.lr_scheduler.ExponentialLR(
        trainer.optimizer,
        gamma = config[0]['step_size']
    )

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = config[0]['epochs'],
                log_interval  = log_interval,
                save_interval = save_interval,
                labels        = ChestXRayNPYDataset.labels,
                model_dir     = model_path,
                stage         = '0')

    print('------ STAGE 1 --------')
    for name, param in model.named_parameters():
        if ('layer3' in name) or ('layer4' in name) or ('fc' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainer.optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config[1]['lr']
    )
    trainer.scheduler = optim.lr_scheduler.ExponentialLR(
        trainer.optimizer,
        gamma = config[1]['step_size']
    )

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = config[1]['epochs'],
                log_interval  = log_interval,
                save_interval = save_interval,
                labels        = ChestXRayNPYDataset.labels,
                model_dir     = model_path,
                stage         = '1')

    print('------ STAGE 2 --------')
    for name, param in model.named_parameters():
        if ('layer4' in name) or ('fc' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainer.optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config[2]['lr']
    )
    trainer.scheduler = optim.lr_scheduler.ExponentialLR(
        trainer.optimizer,
        gamma = config[2]['step_size']
    )

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = config[2]['epochs'],
                log_interval  = log_interval,
                save_interval = save_interval,
                labels        = ChestXRayNPYDataset.labels,
                model_dir     = model_path,
                stage         = '2')

    print('------ STAGE 3 --------')
    for name, param in model.named_parameters():
        if  ('fc' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainer.optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config[3]['lr']
    )
    trainer.scheduler = optim.lr_scheduler.ExponentialLR(
        trainer.optimizer,
        gamma = config[3]['step_size']
    )

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = config[3]['epochs'],
                log_interval  = log_interval,
                save_interval = save_interval,
                labels        = ChestXRayNPYDataset.labels,
                model_dir     = model_path,
                stage         = '3')


def ft_resnet_50_adam_exponential(
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
    trainer.scheduler = optim.lr_scheduler.ExponentialLR(
        trainer.optimizer,
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
    parser.add_argument('--runs', nargs='+', type = int)
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
    parser.add_argument('--official', type = bool, default = False, help = 'Use official train/test split. overrides folds and val-id')
    args = parser.parse_args()
    print(args.runs)

    # Use exactly the supplied device. No error handling whatsoever
    device = torch.device(args.device)

    # Loads the entire train/val dataset with official test data left out
    data       = ChestXRayNPYDataset(file      = args.data_train,
                                     transform = transform)

    if args.official:
        print("Using official train/test split")
        data_train = data
        data_val   = ChestXRayNPYDataset(file      = args.data_test,
                                         transform = None)
        data_test = data_val
    else:
        # Perform a k-fold split with random.shuffle()
        split      = dataset.k_fold_split_patient_aware(dataset = data,
                                                        folds   = args.folds,
                                                        val_id  = args.val_id)
        data_val, data_train = split
        data_test   = ChestXRayNPYDataset(file      = args.data_test,
                                          transform = None)


    # simple data loaders are enough, as everything is in memory anyway
    # and using a single gpu suffices. As GPU speed is not the bottleneck
    val_loader   = torch.utils.data.DataLoader(data_val,
                                               batch_size=args.val_bs,
                                               collate_fn=collate.cf)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size = args.train_bs,
                                               collate_fn = collate.cf,
                                               sampler    = RandomSampler(range(len(data_train))))


    if 0 in args.runs:
        print("resnet 40 adam, steplr")
        ft_resnet_50_adam_steplr(config = configs['ft_resnet_50_adam_steplr'],
                                 device = device,
                                 log_interval = args.log_interval,
                                 save_interval = args.save_interval,
                                 out_path = os.path.join(args.save_path,
                                                         'resnet_50_adam_steplr'),
                                 train_loader = train_loader,
                                 val_loader = val_loader,
                                 seed = args.seed)

    if 1 in args.runs:
        ft_resnet_50_adam_redplat(config = configs['ft_resnet_50_adam_redplat'],
                                  device = device,
                                  log_interval = args.log_interval,
                                  save_interval = args.save_interval,
                                  out_path = os.path.join(args.save_path,
                                                          'resnet_50_adam_redplat'),
                                  train_loader = train_loader,
                                  val_loader = val_loader,
                                  seed = args.seed)

    if 2 in args.runs:
        print("resnet 50 adam exponential")
        ft_resnet_50_adam_exponential(config = configs['ft_resnet_50_adam_exponential'],
                                      device = device,
                                      log_interval = args.log_interval,
                                      save_interval = args.save_interval,
                                      out_path = os.path.join(args.save_path,
                                                              'resnet_50_adam_exponential'),
                                      train_loader = train_loader,
                                      val_loader = val_loader,
                                      seed = args.seed)

    if 3 in args.runs:
        print("resnet 50 staged steplr")
        ft_resnet_50_adam_steplr_staged(config = configs['ft_resnet_50_adam_steplr_staged'],
                                      device = device,
                                      log_interval = args.log_interval,
                                      save_interval = args.save_interval,
                                      out_path = os.path.join(args.save_path,
                                                              'resnet_50_adam_exponential'),
                                      train_loader = train_loader,
                                      val_loader = val_loader,
                                      seed = args.seed)

if __name__ == "__main__":
    main()
