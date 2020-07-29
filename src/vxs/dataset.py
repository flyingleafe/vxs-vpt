import pandas as pd
import numpy as np
import librosa as lr
import glob
import logging
import torch
import torchaudio
import torch.nn.functional as F
import note_seq

from pathlib import PurePath
from functools import lru_cache
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from vxs import constants
from vxs.track import Track
from vxs.features import mel_specgram_cae
from vxs.encoding import note_seq_to_annotation

class Annotation(pd.DataFrame):
    """
    Special DataFrame which can also contain info about bpm
    """
    def __init__(self, *args, bpm=None, **kwargs):
        super(Annotation, self).__init__(*args, **kwargs)
        self.bpm = bpm

def read_annotation(path, total_duration=None):
    path = PurePath(path)
    if path.suffix == '.csv':
        df = Annotation(pd.read_csv(path, names=['time', 'class']))
    elif path.suffix == '.mid':
        seq = note_seq.midi_file_to_note_sequence(path)
        df = Annotation(note_seq_to_annotation(seq), bpm=seq.tempos[0].qpm)
    else:
        raise ValueError(f'Unsupported annotation file extension: {path.suffix}')

    df['class'] = df['class'].str.strip()
    if total_duration is not None:
        df = Annotation(df[df['time'] < total_duration], bpm=df.bpm)
    return df

class ListDataset(Dataset):
    """
    Simple class to wrap a list of tensors
    """
    def __init__(self, data):
        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

class TrackSet(Dataset):
    """
    Abstract class which defines the set of tracks
    """
    def __init__(self, root_dir, **kwargs):
        self.root_dir = PurePath(root_dir)
        self.annotated_track_names = self.get_filenames(**kwargs)
        self.track_map = {
            PurePath(tp).stem: idx
            for (idx, (tp, ap)) in enumerate(self.annotated_track_names)
        }

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
            # there are silly onsets which are after the track's end or
            # a couple of milliseconds before that
            # we'll say that onsets which are after 10ms before track end
            # are silly and we won't count them
            annotation = read_annotation(annotation_name, track.duration - 0.01)
            yield (track, annotation)

class GenTrackSet(TrackSet):
    def get_filenames(self, anno_ext='.mid', **kwargs):
        wav_paths = [PurePath(p) for p in glob.glob(str(self.root_dir / '*.wav'))]
        anno_paths = [p.with_suffix(anno_ext) for p in wav_paths]
        return list(zip(wav_paths, anno_paths))

class AVPTrackSet(TrackSet):

    CLASS_MAP = {
        'kd': 'Kick',
        'sd': 'Snare',
        'hhc': 'HHopened',
        'hho': 'HHclosed'
    }

    def get_filenames(self, subset='*', recordings_type='all',
                      participant=None, **kwargs):

        if participant is None or type(participant) != int:
            participant_mask = '*'
        else:
            participant_mask = f'Participant_{participant}'

        glob_path = str(self.root_dir / f'{subset}/{participant_mask}/*.wav')
        avp_paths = [PurePath(path) for path in glob.glob(glob_path)]

        if len(avp_paths) == 0:
            raise ValueError('No paths are found; check your parameters')

        if participant is not None and type(participant) != int:
            avp_paths = [p for p in avp_paths if int(p.parts[-2].split('_')[1]) in participant]

        if recordings_type == 'all':
            pass
        elif recordings_type == 'hits':
            avp_paths = [p for p in avp_paths if 'Improv' not in str(p)]
        elif recordings_type == 'improvs':
            avp_paths = [p for p in avp_paths if 'Improv' in str(p)]
        elif recordings_type in AVPTrackSet.CLASS_MAP:
            avp_paths = [p for p in avp_paths
                         if AVPTrackSet.CLASS_MAP[recordings_type] in str(p)]
        else:
            raise ValueError(f'Unknown recording type {recordings_type} (expected all, hits, improvs or class type)')

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
                 spectre_only=True, cache_specgram=True, pad_specgram=128, pad_track=None):
        if isinstance(filenames, (str, PurePath)):
            filenames = glob.glob(str(filenames))

        self.filenames = filenames
        self.tracks = tracks
        self.normalize = normalize
        self.wave_only = wave_only
        self.spectre_only = spectre_only
        self.pad_specgram = pad_specgram
        self.pad_track = pad_track
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
            return track

        if self.pad_track is not None:
            track = track.cut_or_pad(self.pad_track)

        device = 'cpu' # 'cuda' if self.cuda else 'cpu'
        mel_specgram_db = mel_specgram_cae(track, pad_time=self.pad_specgram,
                                           device=device, normalize=self.normalize)

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
            for idx in range(len(annotation)):
                row = annotation.iloc[idx]
                time, event_class = row['time'], row['class']
                if event_class in self.classes:
                    if frame_window is not None:
                        segm = track.segment_frames(int(time*track.rate), self.frame_window)
                        if segm.n_samples == self.frame_window:  # do not add cropped stuff on the end
                            self.segments.append((segm, event_class))
                    else:
                        if idx < len(annotation) - 1:
                            end_time = annotation.loc[idx+1, 'time']
                            end_frame = int(end_time*track.rate) - 1
                        else:
                            end_frame = track.n_samples - 1
                        cur_frame = int(time * track.rate)
                        win_len = end_frame - cur_frame
                        segm = track.segment_frames(cur_frame, win_len)
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
