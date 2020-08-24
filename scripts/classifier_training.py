import numpy as np
import torch
import glob
import yaml
import sys
import os
import functools

from tqdm import tqdm
from catalyst import dl
from catalyst.dl.callbacks import AccuracyCallback
from catalyst.core.callbacks.early_stop import EarlyStoppingCallback

from torch.utils.data import ConcatDataset, DataLoader

import vxs

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    group = config['group']
    data_info = config['data']
    pretrained_caes_path = config['pretrained_caes_path']
        
    for PAD_SPECGRAM in config['sgram_lengths']:
        
        datasets = []
        for ds in data_info['datasets']:
            if ds['type'] == 'samples':
                dataset = vxs.SimpleSampleSet(glob.glob(ds['path']))
            elif ds['type'] == 'csv':
                dataset = vxs.SimpleSampleSet.from_csv(ds['path'])

            print(f"Dataset '{ds['name']}' included, {len(dataset)} samples")
            datasets.append(dataset)

        common_set = ConcatDataset(datasets)
        print(f'Total samples: {len(common_set)}')

        feat_trans = functools.partial(vxs.bark_specgram_cae, pad_time=PAD_SPECGRAM)
        trans_set = vxs.TransformDataset(common_set, ft=feat_trans)

        save_file_name = f'../data_temp/classifier_{group}_{PAD_SPECGRAM}.pt'
        os.makedirs('../data_temp', exist_ok=True)
        tensors_set = vxs.save_or_load_spectrograms(trans_set, save_file_name)

        splitter = vxs.DataSplit(tensors_set, **data_info['splitter'])
        loaders = {
            'train': splitter.get_train_loader(),
            'valid': splitter.get_validation_loader()
        }

        num_epochs_head = config['num_epochs_head']
        num_epochs_fine = config['num_epochs_fine']
        experiments = config['experiments']
        num_experiments = len(experiments)

        for i, model_type in enumerate(experiments):
            print(f"Running experiment '{model_type}' ({i}/{num_experiments})")

            ckp_path = pretrained_caes_path.format(model_type)
            model = vxs.get_CAE_classifier(model_type, PAD_SPECGRAM, ckp_path)

            print('Training head-only')
            exp_name = f'{group}_{model_type}_{PAD_SPECGRAM}_head'

            runner = dl.SupervisedRunner(device='cuda:0')
            criterion = torch.nn.CrossEntropyLoss()
            optimizer_head = torch.optim.Adam(model.head_parameters(), lr=0.001)
            callbacks = [
                AccuracyCallback(num_classes=4),
                #EarlyStoppingCallback(patience=10, metric='accuracy01', minimize=False)
            ]

            runner.train(
                model=model,
                optimizer=optimizer_head,
                criterion=criterion,
                loaders=loaders,
                num_epochs=num_epochs_head,
                main_metric='accuracy01',
                minimize_metric=False,
                verbose=True,
                timeit=False,
                logdir=f"../logs/classifiers/{exp_name}",
                load_best_on_end=True,
                initial_seed=42,
                callbacks=callbacks+[vxs.alchemy_logger(group, exp_name)]
            )

            exp_name = f'{group}_{model_type}_{PAD_SPECGRAM}_fine'
            print('Fine-tuning')
            optimizer_fine = torch.optim.Adam(model.parameters(), lr=0.0002)
            runner.train(
                model=model,
                optimizer=optimizer_fine,
                criterion=criterion,
                loaders=loaders,
                num_epochs=num_epochs_fine,
                main_metric='accuracy01',
                minimize_metric=False,
                verbose=True,
                timeit=False,
                logdir=f"../logs/classifiers/{exp_name}",
                initial_seed=43,
                callbacks=callbacks+[vxs.alchemy_logger(group, exp_name)]
            )

if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)