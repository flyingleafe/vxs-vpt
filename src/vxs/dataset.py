import pandas as pd
import numpy as np
import glob
import logging
import torch
import torchaudio
import torch.nn.functional as F

from pathlib import PurePath
from functools import lru_cache
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from .track import Track

def read_annotation(path):
    return pd.read_csv(path, names=['time', 'class'])

class TrackSet:
    """
    Abstract class which defines the set of tracks
    """
    def __init__(self, root_dir, **kwargs):
        self.root_dir = PurePath(root_dir)
        self.annotated_track_names = self.get_filenames(**kwargs)
        self.track_map = {PurePath(tp).stem: idx for (idx, (tp, ap)) in enumerate(self.annotated_track_names)}
        
    def __len__(self):
        return len(self.annotated_track_names)
    
    def __getitem__(self, index):
        if type(index) == str:
            return self.annotated_track_names[self.track_map[index]]
        else:
            return self.annotated_track_names[index]
    
    def get(self, index):
        track_name, annotation_name = self[index]
        return Track(track_name), read_annotation(annotation_name)
    
    def get_filenames(self, **kwargs):
        raise NotImplementedError()
        
    def annotated_tracks(self):
        for (track_name, annotation_name) in self.annotated_track_names:
            track = Track(track_name)
            annotation = read_annotation(annotation_name)
            yield (track, annotation)
            
class AVPTrackSet(TrackSet):
    def get_filenames(self, **kwargs):
        avp_paths = [PurePath(path) for path in glob.glob(str(self.root_dir / '*/*/*.wav'))]
        return [(str(fp), str(fp.with_suffix('.csv'))) for fp in avp_paths]
    
class Beatbox1TrackSet(TrackSet):
    def get_filenames(self, annotation_type, **kwargs):
        bbs_files = [PurePath(path).stem for path in glob.glob(str(self.root_dir / '*.wav'))]
        
        # bad (non-really-readable) files removal
        # TODO: fix those files instead
        bbs_files.remove('putfile_dbztenkaichi')
        bbs_files.remove('callout_Pneumatic')
        bbs_files.remove('putfile_vonny')
        bbs_files.remove('putfile_pepouni')
        
        if annotation_type == 'HT':
            annotations_path = self.root_dir / 'Annotations_HT'
        elif annotation_type == 'DR':
            annotations_path = self.root_dir / 'Annotations_DR'
        else:
            raise ValueError('Unknown annotations variant: ' + annotation_type)
        
        return [(self.root_dir / (stem + '.wav'), annotations_path / (stem + '.csv')) for stem in bbs_files]
    
    
class SampleSet(Dataset):
    def __init__(self, filenames, normalize=False):
        self.filenames = filenames
        self.normalize = normalize
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        # TODO: setup proper parameters for filterbank and FFT as defined in Mehrabi's
        wave, samplerate = torchaudio.load(self.filenames[index], normalization=self.normalize)
        # danger!
        assert samplerate == 44100
        
        win_length = int(samplerate * 0.093)  # 93ms window
        hop_length = int(win_length * 0.125)  # 87.5% overlap
        n_fft = max(400, win_length)
        
        # there are some very short clips so that default 400-sample FFT cannot pass (actually wtf?)
        # pad those clips with zeros then (or maybe better mirroring?)
        if wave.size()[-1] < n_fft:
            new_wave = torch.zeros(wave.size()[0], n_fft)
            new_wave[:, :wave.size()[-1]] = wave
            wave = new_wave
        
        mel_specgram = MelSpectrogram(samplerate, n_fft=n_fft,
                                      win_length=win_length, hop_length=hop_length)(wave)
        mel_specgram_db = AmplitudeToDB()(mel_specgram)
        
        if mel_specgram_db.size()[-1] >= 128:
            mel_specgram_db = mel_specgram_db[:, :, :128]
        else:
            mel_specgram_db = F.pad(mel_specgram_db, (0, 128 - mel_specgram_db.size()[-1], 0, 0, 0, 0),
                                    mode='constant', value=0)
        
        return mel_specgram_db


class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[ : validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler,
                                       shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler,
                                     shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler,
                                      shuffle=False, num_workers=num_workers)
        return self.test_loader