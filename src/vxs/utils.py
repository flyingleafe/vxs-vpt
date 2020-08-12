import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from IPython import display

from .track import *

def plot_track(track, onsets=None, event_type=None, color_events=False,
               title=None, return_events=False, bpm=None, ax=None, figsize=(20, 5), fsize=16,
               ylabel='Signal amplitude', xlabel='Time, seconds'):

    if ax is None:
        fig = plt.figure(figsize=figsize)
    else:
        plt.sca(ax)

    plt.plot(np.linspace(0, track.duration, track.n_samples), track.wave)

    events = []
    if onsets is not None:
        if not isinstance(onsets, pd.DataFrame):
            onsets = pd.DataFrame.from_dict({
                'time': onsets,
                'class': ['x']*len(onsets)
            })

        if color_events:
            color_map = {
                'kd': 'r',
                'sd': 'g',
                'hhc': 'c',
                'hho': 'm',
            }
            classes = sorted(onsets['class'].unique())
            add_classes = [cl for cl in classes if cl not in color_map]
            add_colors = ['y', 'k', 'b', 'beige', 'grey', 'purple', 'lime']
            for cl, col in zip(add_classes, add_colors):
                color_map[cl] = col

            patches = []
            for cl in classes:
                patches.append(mpatches.Patch(color=color_map[cl], label=cl))
            plt.legend(handles=patches, loc='upper right')

        for (idx, row) in onsets.iterrows():
            if event_type is None or row['class'] == event_type:
                events.append(row['time'])
                color = 'r' if not color_events else color_map[row['class']]
                plt.axvline(x=row['time'], color=color)

        if bpm is not None:
            fst_onset = onsets['time'].values[0]
            last_onset = onsets['time'].values[-1]
            sec_per_beat = 60.0 / bpm / 4
            xticks = np.arange(fst_onset, last_onset + 0.2, sec_per_beat)
            xlabels = []
            for i in range(len(xticks)):
                if i % 4 == 0:
                    xlabels.append(str((i // 4) % 4 + 1))
                else:
                    xlabels.append('')
            plt.xticks(ticks=xticks, labels=xlabels, fontsize=fsize-2)
            plt.grid(which='both', axis='x')

    max_amp = np.max(track.wave)
    min_amp = np.min(track.wave)
    plt.ylim((min_amp - 0.05, max_amp + 0.05))
    plt.yticks(fontsize=fsize-2)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fsize)

    if title is not None:
        plt.title(title)
    if ax is None:
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

def save_plot(name):
    os.makedirs('../plots', exist_ok=True)
    plt.savefig(f'../plots/{name}.pdf', bbox_inches='tight')
