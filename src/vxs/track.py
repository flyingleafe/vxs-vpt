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
        self.wave = source.astype('float32')
            
    def segment(self, start, duration):
        return self.segment_frames(int(start*self.rate), int(duration*self.rate))
    
    def segment_frames(self, start, length):
        segm = self.wave[start:start+length]
        return Track(segm, samplerate=self.rate)
    
    def cut_or_pad(self, to_length):
        if to_length == self.n_samples:
            return self
        elif to_length < self.n_samples:
            res = Track(self.wave[:to_length], self.rate)
            res.filepath = self.filepath
            return res
        else:
            res = Track(np.pad(self.wave, (0, to_length - self.n_samples)), self.rate)
            res.filepath = self.filepath
            return res

def detect_onsets(track, method, buf_size=1024, hop_size=512):
    onset_detector = aubio.onset(method=method, buf_size=buf_size, hop_size=hop_size,
                                 samplerate=track.rate)
    N = track.n_samples // hop_size
    onsets_sec = []
    onsets = []
    for i in range(N):
        chunk = track.wave[i*hop_size:(i+1)*hop_size]
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

def detect_beats(track, method, buf_size=1024, hop_size=512):
    beat_detector = aubio.tempo(method=method, buf_size=buf_size, hop_size=hop_size,
                                samplerate=track.rate)
    N = track.n_samples // hop_size
    beats = []
    for i in range(N):
        chunk = track.wave[i*hop_size:(i+1)*hop_size]
        if beat_detector(chunk):
            beats.append(beat_detector.get_last_s())
            
    bdiff = 60./ np.diff(beats)
    tempo = np.mean(bdiff)
    
    classes = ['b']*len(beats)
    beats_detected = pd.DataFrame.from_dict({
        'time': np.array(beats),
        'class': classes
    })
    return beats_detected, tempo

def specdesc(track, method, buf_size=1024, hop_size=512):
    pv = aubio.pvoc(buf_size, hop_size)
    sd = aubio.specdesc(method, buf_size)
    N = track.n_samples // hop_size
    desc = np.array([])
    for i in range(N):
        chunk = track.wave[i*hop_size:(i+1)*hop_size]
        desc = np.append(desc, sd(pv(chunk)))
        
    return desc