from modules.dataset import ChestXRayImages
import argparse
from modules import dataset
from pprint import pp
from main import seed_everything
import termplotlib as tpl


def main():
    imgs = ChestXRayImages(root = './data/orig',
                           folds = 5)

    test = imgs.data_test
    train = imgs.data_train_all

    findings = {
        'Atelectasis': {},
        'Cardiomegaly': {},
        'Consolidation': {},
        'Edema': {},
        'Effusion': {},
        'Emphysema': {},
        'Fibrosis': {},
        'Hernia': {},
        'Infiltration': {},
        'Mass': {},
        'Nodule': {},
        'Pleural_Thickening': {},
        'Pneumonia': {},
        'Pneumothorax': {},
        'none': {}
    }

    for f in findings:
        findings[f] = {'test': 0, 'train': 0}


    for _, d in test.iterrows():
        for f in d['findings']:
            findings[f]['test'] += 1

    for _, d in train.iterrows():
        for f in d['findings']:
            findings[f]['train'] += 1

    f_test = []
    f_train = []
    f_labels = []
    for f in findings:
        f_test.append(findings[f]['test'])
        f_train.append(findings[f]['train'])
        f_labels.append(f)



    fig_1 = tpl.figure()
    fig_1.barh(f_test, f_labels, force_ascii=True)

    fig_2 = tpl.figure()
    fig_2.barh(f_train, f_labels, force_ascii=True)

    print("--- Test Data ---")
    fig_1.show()
    print()
    print()
    print("--- Train Data ---")
    fig_2.show()

def main_folds():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train', type = str)
    parser.add_argument('--seed', type=int, default=0, help='Seed the random generator to get reproducability')
    args = parser.parse_args()
    seed_everything(args.seed)

    findings = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Effusion',
        'Emphysema',
        'Fibrosis',
        'Hernia',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pleural_Thickening',
        'Pneumonia',
        'Pneumothorax',
        'none'
    ]

    data       = dataset.ChestXRayNPYDataset(file      = args.data_train,
                                             transform = None)
    for val_id in range(5):
        split      = dataset.k_fold_split_patient_aware(dataset = data,
                                                        folds   = 5,
                                                        val_id  = val_id)
        data_val, data_train  = split
        combined_val_targets  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        combined_train_targets = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for _, t in data_val:
            combined_val_targets = [sum(x) for x in zip(combined_val_targets, t)]

        for _, t in data_train:
            combined_train_targets = [sum(x) for x in zip(combined_train_targets, t)]

        fig_1 = tpl.figure()
        fig_1.barh(combined_val_targets, findings, force_ascii=True)

        fig_2 = tpl.figure()
        fig_2.barh(combined_train_targets, findings, force_ascii=True)

        print('--- Val Data {} ---'.format(val_id))
        fig_1.show()
        print()
        print()
        print('--- Train Data {} ---'.format(val_id))
        fig_2.show()

if __name__ == "__main__":
    main_folds()
