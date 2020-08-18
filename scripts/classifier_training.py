import numpy as np
import torch
import glob
import yaml
import sys
import os
import functools
from tqdm import tqdm

from torch.utils.data import ConcatDataset, DataLoader

import vxs

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    group = config['group']
    data_info = config['data']
    pretrained_caes_path = config['pretrained_caes_path']
        
    PAD_SPECGRAM = 24
        
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
    
    feat_trans = functools.partial(vxs.bark_specgram_cae, pad_time=24, device='cuda:0')
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
        print(f"Running experiment '{experiment}' ({i}/{num_experiments})")

        ckp_path = pretrained_caes_path.format(model_type)
        model = vxs.get_CAE_classifier(model_type, PAD_SPECGRAM, ckp_path)
        
        print('Training head-only')
        exp_name = f'classifier_{group}_{model_type}_head'

        runner = vxs.ClassifierRunner()
        optimizer_head = torch.optim.Adam(nn_classifier.head_parameters(), lr=0.001)
        runner.train(
            model=model,
            optimizer=optimizer_head,
            loaders=loaders,
            num_epochs=num_epochs_head,
            verbose=True,
            timeit=False,
            logdir=f"../logs/{exp_name}",
            callbacks={'alchemy_logger': vxs.alchemy_logger(group, exp_name)}
        )
        
        print('Fine-tuning')
        optimizer_fine = torch.optim.Adam(nn_classifier.parameters(), lr=0.001)
        runner.train(
            model=model,
            optimizer=optimizer_fine,
            loaders=loaders,
            num_epochs=num_epochs_fine,
            verbose=True,
            timeit=False,
            logdir=f"../logs/{exp_name}",
            callbacks={'alchemy_logger': vxs.alchemy_logger(group, exp_name)}
        )

if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)