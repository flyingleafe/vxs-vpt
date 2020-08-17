import numpy as np
import torch
import glob
import yaml
import sys
import os
from tqdm import tqdm

from torch.utils.data import ConcatDataset, DataLoader

import vxs

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    group = config['group']
    data_info = config['data']
    model_type = config.get('model_type', 'cae')
    sgram_type = config.get('sgram_type', 'mel')
    
    SGRAM_KWARGS = {'sgram_type': sgram_type}
    
    if model_type == 'cae':
        BASE_FRAME = data_info['base_frame']
        PAD_TRACK_LEN = BASE_FRAME * data_info['pad_track_n']
        PAD_SPECGRAM = PAD_TRACK_LEN // 512
    else:
        BASE_FRAME = 4096
        PAD_TRACK_LEN = BASE_FRAME*2
        PAD_SPECGRAM = PAD_TRACK_LEN // 128
        SGRAM_KWARGS = {
            'win_size': 1024,
            'hop_size': 128,
            'n_mels': 64,
            **SGRAM_KWARGS
        }
        #kl_loss_weight = config.get('kl_loss_weight', 0.1)
        
    datasets = []
    for ds in data_info['datasets']:
        if ds['type'] == 'samples':
            dataset = vxs.SampleSet(glob.glob(ds['path']), pad_track=PAD_TRACK_LEN,
                                    pad_specgram=PAD_SPECGRAM, **SGRAM_KWARGS)
        elif ds['type'] == 'segments':
            DatasetClass = getattr(vxs, ds['subtype'])
            subset = DatasetClass(**ds['subargs'])
            frame_window = BASE_FRAME * ds['frame_window_n']
            dataset = vxs.SampleSet(tracks=vxs.SegmentSet(subset, frame_window=frame_window, return_class=False),
                                    pad_track=PAD_TRACK_LEN, pad_specgram=PAD_SPECGRAM, **SGRAM_KWARGS)

        print(f"Dataset '{ds['name']}' included, {len(dataset)} samples")
        datasets.append(dataset)

    common_set = ConcatDataset(datasets)
    print(f'Total samples: {len(common_set)}')

    save_file_name = f'../data_temp/{group}_{PAD_TRACK_LEN}.pt'
    os.makedirs('../data_temp', exist_ok=True)
    tensors_set = vxs.save_or_load_spectrograms(common_set, save_file_name)

    splitter = vxs.DataSplit(tensors_set, **data_info['splitter'])
    loaders = {
        'train': splitter.get_train_loader(),
        'valid': splitter.get_validation_loader()
    }

    num_epochs = config['num_epochs']
    experiments = config['experiments']
    num_experiments = len(experiments)

    for i, experiment in enumerate(experiments):
        print(f"Running experiment '{experiment}' ({i}/{num_experiments})")

        if type(experiments) == dict:
            params = experiments[experiment]
        else:
            params = vxs.CAE_CONFIGS[experiment] if model_type == 'cae' else vxs.VAR_CAE_CONFIGS[experiment]
            
        model = vxs.ConvAE(**params) if model_type == 'cae' else vxs.ConvVAE(**params)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        exp_name = f'{group}_{PAD_TRACK_LEN}_{experiment}'

        runner = vxs.ConvAERunner() if model_type == 'cae' else vxs.ConvVAERunner()
        runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=num_epochs,
            verbose=True,
            timeit=False,
            logdir=f"../logs/{exp_name}",
            callbacks={'alchemy_logger': vxs.alchemy_logger(group, exp_name)}
        )

if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)
