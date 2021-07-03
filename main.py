import argparse
import os
import random
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchinfo import summary
from torchvision import transforms

from modules.utils import seed_everything
from modules import collate, dataset, loss, net, trainer
from modules.sampler import weights
from modules.dataset import (ChestXRayImageDataset, ChestXRayImages,
                             ChestXRayNPYDataset)

class BaseNet(Enum):
    RESNET_34   = 1
    RESNET_50   = 2

    EFFNET_B0   = 3
    EFFNET_B7   = 4

    GOOGLENET   = 5

    DENSENET161 = 6

    LARGERES50  = 7

class BaseOptimizer(Enum):
    ADAM = 1
    SGD  = 2

class BaseScheduler(Enum):
    STEPLR          = 1
    REDUCEONPLATEAU = 2
    EXPONENTIAL     = 3

class BaseLoss(Enum):
    BCE     = 1
    BPMLL   = 2
    HAMMING = 3


# use Optimizer.add_param_group() in callback for finetuning
    # 'ft_effnet_b7_adam_exponential': {
    #     'lr': 0.0001,
    #     'epochs': 10,
    #     'gamma': 0.1
    # },
    # 'ft_effnet_b7_adam_steplr': {
    #     'lr': 0.001,
    #     'step_size': 2,
    #     'gamma': 0.5,
    #     'epochs': 5
    # },
    # 'ft_effnet_b7_sgd_steplr': {
    #     'lr': 0.001,
    #     'step_size': 2,
    #     'gamma': 0.5,
    #     'epochs': 5
    # },
    # 'ft_effnet_b0_sgd_steplr': {
    #     'lr': 0.001,
    #     'step_size': 2,
    #     'gamma': 0.5,
    #     'epochs': 5
    # },
    # 'ft_effnet_b0_adam_steplr': {
    #     'lr': 0.001,
    #     'step_size': 2,
    #     'gamma': 0.5,
    #     'epochs': 5
    # }
    # 'ft_resnet_34_adam_steplr': {
    #     'lr': 0.0001,
    #     'step_size': 1,
    #     'gamma': 0.25,
    #     'epochs': 5
    # },
    # 'ft_resnet_50_adam_steplr': {
    #     'lr': 0.0001,
    #     'step_size': 2,
    #     'gamma': 0.5,
    #     'epochs': 5
    # },
    # 'ft_resnet_50_adam_steplr_scratch': {
    #     'lr': 0.01,
    #     'step_size': 2,
    #     'gamma': 0.5,
    #     'epochs': 10
    # },
    # 'ft_resnet_50_adam_redplat': {
    #     'lr': 0.00005,
    #     'epochs': 5,
    #     'factor': 0.25,
    #     'patience': 2
    # },
    # 'ft_resnet_50_adam_exponential': {
    #     'lr': 0.0001,
    #     'epochs': 10,
    #     'gamma': 0.1
    # },

large_conf =  [
    {
        'name': 'ft_largeres_50_adam_steplr_0',
        'net': BaseNet.LARGERES50,
        'epochs': 10,
        'bs': 64,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 5e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 2,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE,
    },
    {
        'name': 'ft_largeres_50_adam_steplr_1',
        'net': BaseNet.LARGERES50,
        'epochs': 10,
        'bs': 64,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 1e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 2,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE,
    },
    {
        'name': 'ft_largeres_50_adam_exponential',
        'net': BaseNet.LARGERES50,
        'epochs': 10,
        'bs': 64,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 1e-3,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.EXPONENTIAL,
            'gamma': 0.1
        },
        'loss': BaseLoss.BCE,
    }
]

conf = [
    {
        'name': 'ft_resnet_50_adam_steplr_0',
        'net' : BaseNet.RESNET_50,
        'epochs': 10,
        'bs': 64,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 1e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 2,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE
    },
    {
        'name': 'ft_resnet_50_adam_steplr_1',
        'net' : BaseNet.RESNET_50,
        'epochs': 10,
        'bs': 64,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 5e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 2,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE
    },
    {
        'name': 'ft_resnet_50_adam_exponential',
        'net' : BaseNet.RESNET_50,
        'epochs': 10,
        'bs': 64,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 1e-3,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.EXPONENTIAL,
            'gamma': 0.1
        },
        'loss': BaseLoss.BCE
    },
    {
        'name': 'ft_dense161_adam_steplr_0',
        'net': BaseNet.DENSENET161,
        'epochs': 10,
        'bs': 32,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 5e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 2,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE,
    },
    {
        'name': 'ft_googlenet_adam_steplr_2',
        'net': BaseNet.GOOGLENET,
        'epochs': 10,
        'bs': 128,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 5e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 2,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE,
        'scratch': False
    },
    {
        'name': 'ft_googlenet_adam_steplr_1',
        'net': BaseNet.GOOGLENET,
        'epochs': 10,
        'bs': 128,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 1e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 5,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE,
        'scratch': False
    },
    {
        'name': 'ft_googlenet_adam_steplr',
        'net': BaseNet.GOOGLENET,
        'epochs': 7,
        'bs': 128,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 5e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 5,
            'gamma': 0.5
        },
        'loss': BaseLoss.BCE,
        'scratch': False
    },
    {
        'name': 'ft_resnet_34_adam_steplr',
        'net': BaseNet.RESNET_34,
        'epochs': 5,
        'bs': 128,
        'callback': None,
        'optim': {
            'type': BaseOptimizer.ADAM,
            'lr': 1e-3,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': BaseScheduler.STEPLR,
            'step_size': 2,
            'gamma': 0.1
        },
        'loss': BaseLoss.BCE
    }
]

def run_conf(
    config,
    device: str,
    log_images: int,
    base_path: str,
    data_train,
    data_val,
    seed: int
):
    seed_everything(seed)

    # simple data loaders are enough, as everything is in memory anyway
    # and using a single gpu suffices. As GPU speed is not the bottleneck
    val_loader   = DataLoader(data_val,
                              batch_size=config['bs'],
                              collate_fn=collate.cf)

    train_loader = DataLoader(data_train,
                              batch_size = config['bs'],
                              collate_fn = collate.cf,
                              sampler    = WeightedRandomSampler(
                                  weights(data_train),
                                  len(data_train)
                              ))
    num_classes = 15
    pretrained = False if config.get('scratch', False) else True

    out_path = os.path.join(base_path, config['name'])

    model_path = os.path.join(out_path, 'models')
    eval_path = os.path.join(out_path, 'eval')

    if config['net'] == BaseNet.RESNET_34:
        model = net.get_resnet_34(num_classes,
                                  pretrained = pretrained)
    elif config['net'] == BaseNet.RESNET_50:
        model = net.get_resnet_50(num_classes,
                                  pretrained = pretrained)
    elif config['net'] == BaseNet.EFFNET_B0:
        model = net.get_effnet_b0(num_classes)
    elif config['net'] == BaseNet.EFFNET_B7:
        model = net.get_effnet_b7(num_classes)
    elif config['net'] == BaseNet.GOOGLENET:
        model = net.get_googlenet(num_classes,
                                  pretrained = pretrained)
    elif config['net'] == BaseNet.DENSENET161:
        model = net.get_densenet161(num_classes,
                                    pretrained = pretrained)
    elif config['net'] == BaseNet.LARGERES50:
        model = net.LargeResNet(num_classes,
                                pretrained = pretrained)
    else:
        raise ValueError("Network {} not defined".format(config['net']))

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)


    # Setup Loss Function
    if config['loss'] == BaseLoss.BCE:
        trainer.criterion_t = nn.BCEWithLogitsLoss()
        trainer.criterion_v = nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Loss {} not defined'.format(config['loss']))


    # Setup optimizer
    if config['optim']['type'] == BaseOptimizer.ADAM:
        trainer.optimizer  = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = config['optim']['lr'],
            betas = config['optim']['betas']
        )
    elif config['optim']['type'] == BaseOptimizer.SGD:
        trainer.optimizer  = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = config['lr'],
            momentum = config['momentum']
        )
    else:
        raise ValueError('Optimzer {} not defined'.format(config['optim']['type']))

    # Setup scheduler
    if config['scheduler']['type'] == BaseScheduler.STEPLR:
        trainer.scheduler = optim.lr_scheduler.StepLR(
            trainer.optimizer,
            step_size = config['scheduler']['step_size'],
            gamma = config['scheduler']['gamma']
        )
    elif config['scheduler']['type'] == BaseScheduler.EXPONENTIAL:
        trainer.scheduler = optim.lr_scheduler.ExponentialLR(
            trainer.optimizer,
            gamma = config['scheduler']['gamma']
        )
    else:
        raise ValueError('scheduler {} not defined'.format(config['scheduler']['type']))

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = config['epochs'],
                log_interval  = int(log_images/config['bs']),
                save_interval = 1,
                labels        = ChestXRayNPYDataset.labels,
                model_dir     = model_path,
                stage         = '0',
                callback      = config['callback'])



configs = {
    'ft_resnet_34_adam_steplr': {
        'lr': 0.0001,
        'step_size': 1,
        'gamma': 0.25,
        'epochs': 5
    },
    'ft_resnet_50_adam_steplr': {
        'lr': 0.0001,
        'step_size': 2,
        'gamma': 0.5,
        'epochs': 5
    },
    'ft_resnet_50_adam_steplr_scratch': {
        'lr': 0.01,
        'step_size': 2,
        'gamma': 0.5,
        'epochs': 10
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
        ],
    'ft_effnet_b7_adam_exponential': {
        'lr': 0.0001,
        'epochs': 10,
        'gamma': 0.1
    },
    'ft_effnet_b7_adam_steplr': {
        'lr': 0.001,
        'step_size': 2,
        'gamma': 0.5,
        'epochs': 5
    },
    'ft_effnet_b7_sgd_steplr': {
        'lr': 0.001,
        'step_size': 2,
        'gamma': 0.5,
        'epochs': 5
    },
    'ft_effnet_b0_sgd_steplr': {
        'lr': 0.001,
        'step_size': 2,
        'gamma': 0.5,
        'epochs': 5
    },
    'ft_effnet_b0_adam_steplr': {
        'lr': 0.001,
        'step_size': 2,
        'gamma': 0.5,
        'epochs': 5
    }
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


def test_model(model, model_weights_path, test_loader, device):
    model.load_state_dict(torch.load(model_weights_path))

    trainer.val_epoch(device = device,
                      loader = test_loader,
                      model = model,
                      labels = ChestXRayNPYDataset.labels,
                      epochs_till_now = 0,
                      final_epoch = 0,
                      log_interval = 20)

def ft_effnet_b0_sgd_steplr(
    config,
    device: str,
    log_interval,
    save_interval: int,
    out_path: str,
    train_loader,
    val_loader,
    seed: int
):
    seed_everything(seed)

    model = net.get_effnet_b0(len(ChestXRayNPYDataset.labels))
    model.to(device)
    model_path = os.path.join(out_path, 'models')
    eval_path = os.path.join(out_path, 'eval')

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Print Network and training info
    summary(model, input_size=(1, 3, 244, 244))
    print('Using device: {}'.format(device))

    trainer.criterion_t = nn.BCEWithLogitsLoss()
    trainer.criterion_v = nn.BCEWithLogitsLoss()

    trainer.optimizer  = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config['lr'],
        momentum = 0.9
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

def ft_effnet_b0_adam_steplr(
    config,
    device: str,
    log_interval,
    save_interval: int,
    out_path: str,
    train_loader,
    val_loader,
    seed: int
):
    seed_everything(seed)

    model = net.get_effnet_b0(len(ChestXRayNPYDataset.labels))
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

def ft_effnet_b7_sgd_steplr(
    config,
    device: str,
    log_interval,
    save_interval: int,
    out_path: str,
    train_loader,
    val_loader,
    seed: int
):
    seed_everything(seed)

    model = net.get_effnet(len(ChestXRayNPYDataset.labels))
    model.to(device)
    model_path = os.path.join(out_path, 'models')
    eval_path = os.path.join(out_path, 'eval')

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Print Network and training info
    summary(model, input_size=(1, 3, 244, 244))
    print('Using device: {}'.format(device))

    trainer.criterion_t = nn.BCEWithLogitsLoss()
    trainer.criterion_v = nn.BCEWithLogitsLoss()

    trainer.optimizer  = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config['lr'],
        momentum = 0.9
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

def ft_effnet_b7_adam_steplr(
    config,
    device: str,
    log_interval,
    save_interval: int,
    out_path: str,
    train_loader,
    val_loader,
    seed: int
):
    seed_everything(seed)

    model = net.get_effnet(len(ChestXRayNPYDataset.labels))
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


def ft_effnet_b7_adam_exponential(
    config,
    device: str,
    log_interval,
    save_interval: int,
    out_path: str,
    train_loader,
    val_loader,
    seed: int
):
    seed_everything(seed)

    model = net.get_effnet(len(ChestXRayNPYDataset.labels))
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

def ft_resnet_50_adam_steplr_scratch(
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

    model = net.get_resnet_50(len(ChestXRayNPYDataset.labels), False)
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

def ft_resnet_34_adam_steplr(
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

    model = net.get_resnet_34(len(ChestXRayNPYDataset.labels))
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
    parser.add_argument('--labels-train', type = str)
    parser.add_argument('--labels-test', type = str)
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
                                     targets   = args.labels_train,
                                     transform = transform)

    if args.official:
        print("Using official train/test split")
        data_train = data
        data_val   = ChestXRayNPYDataset(file      = args.data_test,
                                         targets   = args.labels_test,
                                         transform = None)
        data_test = data_val
    else:
        # Perform a k-fold split with random.shuffle()
        split      = dataset.k_fold_split_patient_aware(dataset = data,
                                                        folds   = args.folds,
                                                        val_id  = args.val_id)
        data_val, data_train = split
        data_test   = ChestXRayNPYDataset(file      = args.data_test,
                                          targets   = args.labels_test,
                                          transform = None)


    # simple data loaders are enough, as everything is in memory anyway
    # and using a single gpu suffices. As GPU speed is not the bottleneck
    val_loader   = DataLoader(data_val,
                              batch_size=args.val_bs,
                              collate_fn=collate.cf,
                              persistent_workers = True)

    train_loader = DataLoader(data_train,
                              batch_size = args.train_bs,
                              collate_fn = collate.cf,
                              sampler    = WeightedRandomSampler(
                                  weights(data_train),
                                  len(data_train)
                              ),
                              persistent_workers = True)


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

    if 4 in args.runs:
        print("Effnet b7 adam exponential")
        ft_effnet_b7_adam_exponential(config = configs['ft_effnet_b7_adam_exponential'],
                                      device = device,
                                      log_interval = args.log_interval,
                                      save_interval = args.save_interval,
                                      out_path = os.path.join(args.save_path,
                                                              'effnet_b7_adam_exponential'),
                                      train_loader = train_loader,
                                      val_loader = val_loader,
                                      seed = args.seed)

    if 5 in args.runs:
        print("Effnet b0 sgd steplr")
        ft_effnet_b0_sgd_steplr(config = configs['ft_effnet_b0_sgd_steplr'],
                                device = device,
                                log_interval = args.log_interval,
                                save_interval = args.save_interval,
                                out_path = os.path.join(args.save_path,
                                                        'effnet_b0_sgd_steplr'),
                                train_loader = train_loader,
                                val_loader = val_loader,
                                seed = args.seed)

    if 6 in args.runs:
        print("Effnet b0 adam steplr")
        ft_effnet_b0_adam_steplr(config = configs['ft_effnet_b0_adam_steplr'],
                                 device = device,
                                 log_interval = args.log_interval,
                                 save_interval = args.save_interval,
                                 out_path = os.path.join(args.save_path,
                                                         'effnet_b0_adam_steplr'),
                                 train_loader = train_loader,
                                 val_loader = val_loader,
                                 seed = args.seed)

    if 7 in args.runs:
        print("Effnet b7 sgd steplr")
        ft_effnet_b7_sgd_steplr(config = configs['ft_effnet_b7_sgd_steplr'],
                                device = device,
                                log_interval = args.log_interval,
                                save_interval = args.save_interval,
                                out_path = os.path.join(args.save_path,
                                                        'effnet_b7_sgd_steplr'),
                                train_loader = train_loader,
                                val_loader = val_loader,
                                seed = args.seed)
    if 8 in args.runs:
        print("Effnet b7 adam steplr")
        ft_effnet_b7_adam_steplr(config = configs['ft_effnet_b7_adam_steplr'],
                                device = device,
                                log_interval = args.log_interval,
                                save_interval = args.save_interval,
                                out_path = os.path.join(args.save_path,
                                                        'effnet_b7_adam_steplr'),
                                train_loader = train_loader,
                                val_loader = val_loader,
                                seed = args.seed)

    if 9 in args.runs:
        print("resnet 50 adam, steplr, scratch")
        ft_resnet_50_adam_steplr_scratch(config = configs['ft_resnet_50_adam_steplr_scratch'],
                                         device = device,
                                         log_interval = args.log_interval,
                                         save_interval = args.save_interval,
                                         out_path = os.path.join(args.save_path,
                                                                 'resnet_50_adam_steplr_scratch'),
                                         train_loader = train_loader,
                                         val_loader = val_loader,
                                         seed = args.seed)

    if 10 in args.runs:
        print("resnet 34 adam, steplr")
        ft_resnet_34_adam_steplr(config = configs['ft_resnet_34_adam_steplr'],
                                         device = device,
                                         log_interval = args.log_interval,
                                         save_interval = args.save_interval,
                                         out_path = os.path.join(args.save_path,
                                                                 'resnet_34_adam_steplr'),
                                         train_loader = train_loader,
                                         val_loader = val_loader,
                                         seed = args.seed)


def main_new(args):
    # Use exactly the supplied device. No error handling whatsoever
    device = torch.device(args.device)

    # Loads the entire train/val dataset with official test data left out
    data       = ChestXRayNPYDataset(file      = args.data_train,
                                     targets   = args.labels_train,
                                     transform = transform)

    if args.official:
        print("Using official train/test split")
        data_train = data
        data_val   = ChestXRayNPYDataset(file      = args.data_test,
                                         targets   = args.labels_test,
                                         transform = None)
        data_test = data_val
    else:
        # Perform a k-fold split with random.shuffle()
        split      = dataset.k_fold_split_patient_aware(dataset = data,
                                                        folds   = args.folds,
                                                        val_id  = args.val_id)
        data_val, data_train = split
        data_test   = ChestXRayNPYDataset(file      = args.data_test,
                                          targets   = args.labels_train,
                                          transform = None)


    for c in conf:
        run_conf(config     = c,
                 device     = device,
                 log_images = args.log_images,
                 base_path  = args.save_path,
                 data_train = data_train,
                 data_val   = data_val,
                 seed       = args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train', type = str)
    parser.add_argument('--data-test', type = str)
    parser.add_argument('--labels-train', type = str)
    parser.add_argument('--labels-test', type = str)
    parser.add_argument('--bs', type = int)
    parser.add_argument('--runs', nargs='+', type = int)
    parser.add_argument('--save-path', type = str, help = 'Path to store models')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Force usage of device')
    parser.add_argument('--log-images', type = int, default = 5, help = 'log every n batches')
    parser.add_argument('--folds', type=int, default=5, help='how many folds to produce')
    parser.add_argument('--val-id', type=int, default=0, help='Which fold id to use for test/val split')
    parser.add_argument('--seed', type=int, default=0, help='Seed the random generator to get reproducability')
    parser.add_argument('--official', type = bool, default = False, help = 'Use official train/test split. overrides folds and val-id')
    args = parser.parse_args()
    main_new(args)
