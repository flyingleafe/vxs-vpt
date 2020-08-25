import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle

from tqdm import tqdm
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
    plt.plot(np.arange(segm.n_samples) / segm.rate, segm.wave)
    return segm

def play_audio(track, sr=44100):
    if isinstance(track, Track):
        dsp = display.Audio(track.wave, rate=track.rate)
    else:
        dsp = display.Audio(track, rate=sr)
    display.display(dsp)

def display_track(track, annotation=None):
    print(track.filepath)
    plot_track(track, annotation)
    play_audio(track)

def unzip_dataset(dataset, tqdm_name=None):
    X = []
    y = []
    iterator = dataset if tqdm_name is None else tqdm(dataset, tqdm_name)
    for features, label in iterator:
        X.append(features)
        y.append(label)
    return X, y

def save_plot(name):
    os.makedirs('../plots', exist_ok=True)
    plt.savefig(f'../plots/{name}.pdf', bbox_inches='tight')
    
def save_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj
        
def dict_min_depth(dct):
    if type(dct) != dict:
        return 0
    else:
        depths = [dict_min_depth(sub) for sub in dct.values()]
        return np.min(depths) + 1

def apply_on_depth(dct, fun, depth, cur_d=0):
    if cur_d == depth:
        return fun(dct)
    else:
        assert type(dct) == dict
        return {key: apply_on_depth(sub, fun, depth, cur_d+1) for key, sub in dct.items()}

def zip_dicts_multi_named(dicts):
    if len(dicts) == 0:
        return dicts
    
    min_depth = False
    for v in dicts.values():
        if type(v) != dict:
            min_depth = True
            break
            
    if min_depth:
        return dicts
    else:
        dcts = list(dicts.values())
        common_keys = set(dcts[0].keys())
        for dct in dcts[1:]:
            common_keys = common_keys.intersection(set(dct.keys()))
        
        return {
            key: zip_dicts_multi_named({name: dct[key] for name, dct in dicts.items()})
            for key in common_keys
        }
    
def unzip_dict_ordered(dct):
    keys, vals = zip(*list(dct.items()))
    keys = np.array(keys)
    vals = np.array(vals)
    sorting = np.argsort(keys)
    return keys[sorting], vals[sorting]

def dict_argmax(dct):
    if type(dct) != dict:
        return [], dct
    else:
        keys, subs = zip(*list(dct.items()))
        paths, vals = zip(*list(map(dict_argmax, subs)))
        ix = np.argmax(vals)
        return [keys[ix]] + paths[ix], vals[ix]
    
class MultilevelDict:
    def __init__(self, dct, levels):
        self.depth = dict_min_depth(dct)
        assert self.depth == len(levels)
        self.levels = levels
        self.dct = dct        
    
    def sub(self, **kwargs):
        key_idxs = []
        keys = np.array(list(kwargs.keys()))
        for key in keys:
            if key not in self.levels:
                raise ValueError(f'key {key} not in '+str(self.levels))
            key_idxs.append(self.levels.index(key))
        
        sorting = np.argsort(key_idxs)
        keys = keys[sorting]
        ixs = np.array(key_idxs)[sorting]
        d = 0
        dct = self.dct
        for key, ix in zip(keys, ixs):
            val = kwargs[key]
            dct = apply_on_depth(dct, lambda sub: sub[val], ix-d)
            d += 1
        new_levels = [lvl for lvl in self.levels if lvl not in keys]
        return MultilevelDict(dct, new_levels)
    
    def sink(self, key):
        assert key in self.levels
        ix = self.levels.index(key)
        if ix == len(self.levels) - 1:
            return self
        
        dct = apply_on_depth(self.dct, zip_dicts_multi_named, ix)
        new_levels = [lvl for lvl in self.levels if lvl != key] + [key]
        return MultilevelDict(dct, new_levels)
        
#     def swap(self, key1, key2):
#         assert key1 in self.levels
#         assert key2 in self.levels
#         ix1 = self.levels.index(key1)
#         ix2 = self.levels.index(key2)
#         l = min(ix1, ix2)
#         r = max(ix1, ix2)
#         if l == r:
#             return self
