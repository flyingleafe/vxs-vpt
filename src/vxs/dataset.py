import pandas as pd
import numpy as np
import librosa as lr
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
from .features import mel_specgram_cae

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
    def get_filenames(self, subset='*', **kwargs):
        avp_paths = [PurePath(path) for path in glob.glob(str(self.root_dir / (subset + '/*/*.wav')))]
        return [(str(fp), str(fp.with_suffix('.csv'))) for fp in avp_paths]
    
class Beatbox1TrackSet(TrackSet):
    def get_filenames(self, annotation_type, **kwargs):
        bbs_files = [PurePath(path).stem for path in glob.glob(str(self.root_dir / '*.wav'))]
        
        if annotation_type == 'HT':
            annotations_path = self.root_dir / 'Annotations_HT'
        elif annotation_type == 'DR':
            annotations_path = self.root_dir / 'Annotations_DR'
        else:
            raise ValueError('Unknown annotations variant: ' + annotation_type)
        
        return [(self.root_dir / (stem + '.wav'), annotations_path / (stem + '.csv')) for stem in bbs_files]
    
    
class SampleSet(Dataset):
    def __init__(self, filenames=None, tracks=None, normalize=True, wave_only=False,
                 spectre_only=True, cache_specgram=True, pad_specgram=True):
        self.filenames = filenames
        self.tracks = tracks
        self.normalize = normalize
        self.wave_only = wave_only
        self.spectre_only = spectre_only
        self.pad_specgram = pad_specgram
        if cache_specgram:
            self.specgram_cache = {}
        else:
            self.specgram_cache = None
            
        self.cuda = torch.cuda.is_available()
        
    def __len__(self):
        if self.tracks is None:
            return len(self.filenames)
        else:
            return len(self.tracks)
    
    def __getitem__(self, index):
        if self.spectre_only and self.specgram_cache is not None and index in self.specgram_cache:
            return self.specgram_cache[index]
        
        if self.tracks is None:
            track = Track(self.filenames[index])
        else:
            track = self.tracks[index]
        
        if self.wave_only:
            return track.wave
                
        device = 'cuda' if self.cuda else 'cpu'
        mel_specgram_db = mel_specgram_cae(track, pad_time=128, device=device, normalize=self.normalize)
        
        if self.specgram_cache is not None:
            self.specgram_cache[index] = mel_specgram_db
        
        if self.spectre_only:
            return mel_specgram_db
        else:
            return mel_specgram_db, track.wave


class SegmentSet(Dataset):
    def __init__(self, trackset, classes=['kd', 'sd', 'hho', 'hhc'], frame_window=4096, return_class=True):
        self.classes = classes
        self.return_class = return_class
        self.frame_window = frame_window
        self.segments = []
        for (track, annotation) in trackset.annotated_tracks():
            for idx, row in annotation.iterrows():
                time, event_class = row[0], row[1]
                if event_class in self.classes:
                    segm = track.segment_frames(int(time*track.rate), self.frame_window)
                    if segm.n_samples == self.frame_window:  # do not add cropped stuff on the end
                        self.segments.append((segm, event_class))
                    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, index):
        if self.return_class:
            return self.segments[index]
        else:
            return self.segments[index][0]
            

class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False, size_limit=None):
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

        if size_limit is not None:
            self.train_indices = self.train_indices
        
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