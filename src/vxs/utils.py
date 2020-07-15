import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from .track import *

def plot_track(track, onsets=None, event_type=None, title=None, return_events=False):
    fig = plt.figure(figsize=(20, 5))
    plt.plot(np.linspace(0, track.duration, track.n_samples), track.wave)
    
    events = []
    if onsets is not None:
        for (idx, row) in onsets.iterrows():
            if event_type is None or row['class'] == event_type:
                events.append(row['time'])
                plt.axvline(x=row['time'], color='r')
                
    plt.ylim((-1.5, 1.5))
    plt.xlabel('Time, seconds')
    plt.ylabel('Signal amplitude')
    if title is not None:
        plt.title(title)
    plt.show()
    
    if return_events:
        return events
    
def plot_segment(track, segm_start, segm_len=0.093):
    segm = track.segment(segm_start, segm_len)
    fig = plt.figure(figsize=(5,2))
    plt.plot(np.arange(len(segm)), segm)
    plt.show()
    return segm

def play_audio(track, sr=44100):
    if isinstance(track, Track):
        return display.Audio(track.wave, rate=track.rate)
    else:
        return display.Audio(track, rate=sr)