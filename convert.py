import argparse
import glob
import os

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_grey = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456],
                         std=[0.224])
])


def parse_csv(root: str, rel_label_file: str, rel_test_list: str, frac: float):
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
              'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
              'Pneumothorax', 'none']

    _data = pd.read_csv(
        os.path.join(root, rel_label_file),
        usecols=['Image Index', 'Finding Labels', 'Patient ID']
    )

    if(frac < 1):
        _data = _data.sample(frac=frac)

    _data.rename(columns = {
        'Image Index': 'idx',
        'Finding Labels': 'findings',
        'Patient ID': 'patient'
    }, inplace = True)

    # replace 'No Finding' with none
    _data['findings'] = _data['findings'].map(lambda x: x.replace('No Finding',
                                                                  'none'))

    # | split labels to list
    _data['findings'] = _data['findings'].map(lambda x: x.split('|')).tolist()

    for label in labels:
        _data[label] = _data['findings'].map(lambda finding: 1.0 if label in finding else 0.0)

    _test_files = pd.read_csv(
        os.path.join(root, rel_test_list),
        header=None,
        squeeze=True
    )
    # split test/train data
    test_filter = pd.Index(_data['idx']).isin(_test_files)
    _data_test  = _data.loc[test_filter].reset_index(drop=True)
    _data_train = _data.loc[[not x for x in test_filter]].reset_index(drop=True)

    return (_data_test, _data_train)


def preload(df: pd.DataFrame, img_dir: str, size: int, rgb: bool):
    _labels = []
    _data   = []

    # L for greyscale
    conversion = 'RGB' if rgb else 'L'

    for idx, row in df.iterrows():
        print('Progress: {}'.format(idx))
        img_path = os.path.join(img_dir, row[0])
        img_path = glob.glob(img_path)
        img = Image.open(img_path[0]).convert(conversion).resize((size, size))
        if rgb:
            transform = transform_rgb
        else:
            transform = transform_grey

        img_tr = transform(img)

        _data.append(np.array(img_tr))
        print(conversion)
        _labels.append(row[2:18].values)

    return np.array(_labels), np.array(_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--file-labels', type=str, default='Data_Entry_2017.csv')
    parser.add_argument('--file-test',   type=str, default='test_list.txt')
    parser.add_argument('--frac',        type=float, default=1)
    parser.add_argument('--store-dir',   type=str, default='./')
    parser.add_argument('--size',        type=int, default=224)
    parser.add_argument('--rgb',         type=int, default=1)
    args = parser.parse_args()
    args.rgb = bool(args.rgb)

    test_data_file    = os.path.join(args.store_dir, 'test_{}.npy'.format(args.size))
    test_target_file  = os.path.join(args.store_dir, 'test_tar_{}.npy'.format(args.size))
    train_data_file   = os.path.join(args.store_dir, 'train_{}.npy'.format(args.size))
    train_target_file = os.path.join(args.store_dir, 'train_tar_{}.npy'.format(args.size))

    test, train = parse_csv(root           = args.data_dir,
                            rel_label_file = args.file_labels,
                            rel_test_list  = args.file_test,
                            frac           = args.frac)
    print(args.rgb)
    np_test_labels, np_test_data = preload(test,
                                           os.path.join(args.data_dir,
                                                        'images_*/images'),
                                           args.size,
                                           args.rgb)
    with open(test_data_file, 'wb') as f:
        np.save(f, np_test_data)
    del np_test_data

    with open(test_target_file, 'wb') as f:
        np.save(f, np_test_labels)
    del np_test_labels

    np_train_labels, np_train_data = preload(train,
                                             os.path.join(args.data_dir,
                                                          'images_*/images'),
                                           args.size,
                                           args.rgb)
    with open(train_data_file, 'wb') as f:
        np.save(f, np_train_data)
    del np_train_data

    with open(train_target_file, 'wb') as f:
        np.save(f, np_train_labels)
    del np_train_labels



if __name__ == "__main__":
    main()
