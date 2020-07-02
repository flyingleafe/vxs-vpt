import numpy as np
import aubio
import pandas as pd

from pathlib import PurePath

class Track:
    """
    Abstracts out the audio track. Only mono tracks are supported currently.
    """
    def __init__(self, source):
        if isinstance(source, PurePath):
            source = str(source)
        
        if type(source) == str:
            source = aubio.source(source)
        if source.channels > 1:
            raise Exception('File {} has {} channels instead of 1'.format(source.uri, source.channels))
        
        self.filepath = source.uri
        self.n_samples = source.duration
        self.rate = source.samplerate
        self.duration = self.n_samples / self.rate
        self.hop_size = source.hop_size
        
        self.wave = aubio.fvec(self.n_samples)
        total_read = 0
        for sample in source:
            m = sample.shape[0]
            self.wave[total_read:total_read+m] = sample
            total_read += m
            
    def segment(self, start, duration):
        return self.wave[int(start*self.rate):int((start+duration)*self.rate)]

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