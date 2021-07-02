import os
import glob
import argparse
from enum import Enum

import torch
import pandas as pd

from modules import net, collate
from modules.dataset import ChestXRayNPYDataset

class Network(Enum):
    RESNET_50 = 1
    EFFNET_b0 = 2
    EFFNET_b7 = 3


def load_network(net_type: Network):
    if net_type == Network.RESNET_50:
        return net.get_resnet_50(15)

    if net_type == Network.EFFNET_b0:
        return net.get_effnet_b0(15)

    if net_type == Network.EFFNET_b7:
        return net.get_effnet_b7(15)

def test_model(model, test_data_loader, device):
    targets     = []
    predictions = []
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(test_data_loader):
            img    = img.to(device)

            out = torch.sigmoid(model(img))

            targets.append(target.cpu())
            predictions.append(out.cpu())

    targets     = torch.cat(targets, dim=0)
    predictions = torch.cat(predictions, dim=0)

    return targets, predictions



def test_all(base_dir, model, test_data_loader, device):
    model_dir = os.path.join(base_dir, 'models/*.pth')
    out_dir = os.path.join(base_dir, 'out')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for m in glob.iglob(model_dir):
        model_name = os.path.basename(m)

        print('Testing {}'.format(model_name))

        model.load_state_dict(torch.load(m, map_location=device))
        tar, pred = test_model(model, test_data_loader, device)
        concat = torch.cat([tar, pred], dim=1)  # each row contains 30 entries

        tar_labels  = ['{}_tar'.format(l) for l in ChestXRayNPYDataset.labels]
        pred_labels = ['{}_pred'.format(l) for l in ChestXRayNPYDataset.labels]

        results = pd.DataFrame(concat,
                               columns = tar_labels + pred_labels,
                               dtype   = float)

        results.to_csv(os.path.join(out_dir, '{}.csv'.format(model_name)))


def main(args):
    data_test   = ChestXRayNPYDataset(file      = args.data_test,
                                      transform = None)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size = args.test_bs,
                                              collate_fn = collate.cf)
    for d in glob.iglob(os.path.join(args.base_dir, '*')):
        if not os.path.isdir(d):
            continue

        base_name = os.path.basename(d)

        if 'resnet_50' in base_name:
            net_type = Network.RESNET_50
        elif 'effnet_b0' in base_name:
            net_type = Network.EFFNET_b0
        elif 'effnet_b7' in base_name:
            net_type = Network.EFFNET_b7
        else:
            raise ValueError("Network type unknown for {}".format(base_name))

        model = load_network(net_type)

        test_all(d, model, test_loader, args.device)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-test', type = str)
    parser.add_argument('--base-dir', type = str)
    parser.add_argument('--test-bs', type = int, default = 64, help = 'test batch size')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Force usage of device')
    parser.add_argument('--log-interval', type = int, default = 5, help = 'log every n batches')
    args = parser.parse_args()

    main(args)
