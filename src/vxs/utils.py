import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from IPython import display

from .track import *

def plot_track(track, onsets=None, event_type=None, color_events=False,
               title=None, return_events=False):
    fig = plt.figure(figsize=(20, 5))
    plt.plot(np.linspace(0, track.duration, track.n_samples), track.wave)

    events = []
    if onsets is not None:
        if not isinstance(onsets, pd.DataFrame):
            onsets = pd.DataFrame.from_dict({
                'time': onsets,
                'class': ['x']*len(onsets)
            })

        if color_events:
            color_map = {}
            classes = sorted(onsets['class'].unique())
            colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']
            patches = []
            for cl, color in zip(classes, colors):
                color_map[cl] = color
                patches.append(mpatches.Patch(color=color, label=cl))
            plt.legend(handles=patches, loc='upper right')

        for (idx, row) in onsets.iterrows():
            if event_type is None or row['class'] == event_type:
                events.append(row['time'])
                color = 'r' if not color_events else color_map[row['class']]
                plt.axvline(x=row['time'], color=color)

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
    plt.plot(np.arange(segm.n_samples), segm.wave)
    plt.show()
    return segm

def play_audio(track, sr=44100):
    if isinstance(track, Track):
        dsp = display.Audio(track.wave, rate=track.rate)
    else:
        dsp = display.Audio(track, rate=sr)
    display.display(dsp)

def display_track(track):
    print(track.filepath)
    plot_track(track)
    play_audio(track)

def unzip_dataset(dataset):
    X = []
    y = []
    for features, label in dataset:
        X.append(features)
        y.append(label)
    return X, y
