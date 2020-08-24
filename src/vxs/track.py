import abc
import numpy as np
import aubio
import pandas as pd
import soundfile as sf
import librosa as lr

from librosa import core as lrcore
from pathlib import PurePath

from vxs import constants

class Track:
    """
    Abstracts out the audio track. Only mono tracks are supported currently.
    """
    def __init__(self, source, samplerate=constants.DEFAULT_SAMPLE_RATE):
        if isinstance(source, Track):
            self.filepath = source.filepath
            self.rate = source.rate
            self.wave = source.wave
            return

        if isinstance(source, PurePath):
            source = str(source)

        if type(source) == str:
            filepath = source
            source, samplerate = lrcore.load(filepath, sr=samplerate, mono=True)
        else:
            filepath = None

        self.filepath = filepath
        self.rate = samplerate
        self.wave = source.astype('float32')

    @abc.abstractproperty
    def n_samples(self):
        return len(self.wave)

    @abc.abstractproperty
    def duration(self):
        return self.n_samples / self.rate

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

    def save(self, filepath):
        sf.write(str(filepath), self.wave, self.rate)

def specdesc(track, method, buf_size=constants.DEFAULT_ONSET_BUF_SIZE,
             hop_size=constants.DEFAULT_ONSET_HOP_SIZE, **kwargs):
    pv = aubio.pvoc(buf_size, hop_size)
    sd = aubio.specdesc(method, buf_size)
    N = track.n_samples // hop_size
    desc = np.array([])
    for i in range(N):
        chunk = track.wave[i*hop_size:(i+1)*hop_size]
        desc = np.append(desc, sd(pv(chunk)))

    return desc

def detect_onsets(track, method, buf_size=constants.DEFAULT_ONSET_BUF_SIZE,
                  hop_size=constants.DEFAULT_ONSET_HOP_SIZE, backtrack=True, **kwargs):
    onset_detector = aubio.onset(method=method, buf_size=buf_size, hop_size=hop_size,
                                 samplerate=track.rate)
    N = track.n_samples // hop_size
    onsets = []
    for i in range(N):
        chunk = track.wave[i*hop_size:(i+1)*hop_size]
        if onset_detector(chunk):
            onsets.append(onset_detector.get_last())

    onsets = np.array(onsets)
    
    if backtrack:
        old_onsets = onsets
        onsets_frames = lr.samples_to_frames(onsets, hop_length=hop_size)
        rms = lr.feature.rms(y=track.wave, frame_length=hop_size, hop_length=hop_size)[0]
        onsets_frames = lr.onset.onset_backtrack(onsets_frames, rms)
        onsets = lr.frames_to_samples(onsets_frames, hop_length=hop_size)
    
    classes = ['x']*len(onsets)
    onsets_sec = onsets / float(track.rate)
    onsets_detected = pd.DataFrame.from_dict({
        'time': onsets_sec,
        'frame': onsets,
        'class': classes
    })
    return onsets_detected

def detect_onsets_lr(track, method, buf_size=constants.DEFAULT_ONSET_BUF_SIZE,
                     hop_size=constants.DEFAULT_ONSET_HOP_SIZE, backtrack=True, **kwargs):
    env = specdesc(track, method, buf_size, hop_size)
    onsets = lr.onset.onset_detect(onset_envelope=env, sr=track.rate,
                                   hop_length=hop_size, backtrack=backtrack, units='samples')
    onsets_sec = onsets / float(track.rate)
    classes = ['x']*len(onsets)
    onsets_detected = pd.DataFrame.from_dict({
        'time': np.array(onsets_sec),
        'frame': np.array(onsets),
        'class': classes
    })
    return onsets_detected

def apply_time_gap(onsets, gap):
    if not gap:
        return onsets
    
    i = 1
    while i < len(onsets):
        cur_time = onsets.iloc[i]['time']
        prev_time = onsets.iloc[i-1]['time']
        if cur_time - prev_time < gap:
            onsets.drop(onsets.index[i], inplace=True)
        else:
            i += 1
    return onsets.reset_index()

def detect_beats(track, method, buf_size=constants.DEFAULT_ONSET_BUF_SIZE,
                 hop_size=constants.DEFAULT_ONSET_HOP_SIZE, **kwargs):
    beat_detector = aubio.tempo(method=method, buf_size=buf_size, hop_size=hop_size,
                                samplerate=track.rate)
    N = track.n_samples // hop_size
    beats = []
    for i in range(N):
        chunk = track.wave[i*hop_size:(i+1)*hop_size]
        if beat_detector(chunk):
            beats.append(beat_detector.get_last_s())

    bdiff = 60./ np.diff(beats)
    tempo = np.median(bdiff)

    classes = ['b']*len(beats)
    beats_detected = pd.DataFrame.from_dict({
        'time': np.array(beats),
        'class': classes
    })
    return beats_detected, tempo

def detect_tempo_lr(track, method, prior=90.0, buf_size=constants.DEFAULT_ONSET_BUF_SIZE,
                    hop_size=constants.DEFAULT_ONSET_HOP_SIZE, **kwargs):
    sd = specdesc(track, method=method, buf_size=buf_size, hop_size=hop_size)
    tempo = lr.beat.tempo(onset_envelope=sd, sr=track.rate, start_bpm=prior)[0]
    return tempo
