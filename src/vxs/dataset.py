import pandas as pd
import glob

from pathlib import PurePath

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