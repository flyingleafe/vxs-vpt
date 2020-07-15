import numpy as np
import aubio
import pandas as pd

from librosa import core as lrcore
from pathlib import PurePath

class Track:
    """
    Abstracts out the audio track. Only mono tracks are supported currently.
    """
    def __init__(self, source, samplerate=44100):
        if isinstance(source, PurePath):
            source = str(source)
        
        if type(source) == str:
            filepath = source
            source, samplerate = lrcore.load(filepath, sr=samplerate, mono=True)
        else:
            filepath = None
        
        self.filepath = filepath
        self.n_samples = len(source)
        self.rate = samplerate
        self.duration = self.n_samples / self.rate
        self.hop_size = 512
        self.wave = source
            
    def segment(self, start, duration):
        return self.segment_frames(int(start*self.rate), int(duration*self.rate))
    
    def segment_frames(self, start, length):
        segm = self.wave[start:start+length]
        return Track(segm, samplerate=self.rate)

def detect_onsets(track, method):
    onset_detector = aubio.onset(method=method)
    hs = track.hop_size
    N = track.n_samples // hs
    onsets_sec = []
    onsets = []
    for i in range(N):
        chunk = track.wave[i*hs:(i+1)*hs]
        if onset_detector(chunk):
            onsets_sec.append(onset_detector.get_last_s())
            onsets.append(onset_detector.get_last())

    classes = ['x']*len(onsets_sec)
    onsets_detected = pd.DataFrame.from_dict({
        'time': np.array(onsets_sec),
        'frame': np.array(onsets),
        'class': classes
    })
    return onsets_detected