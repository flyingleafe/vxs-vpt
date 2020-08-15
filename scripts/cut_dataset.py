import numpy as np
import pandas as pd
import os
import sys
import argparse

from tqdm import tqdm
from glob import glob
from pathlib import PurePath

import vxs

def save_segments(ds, pdir):
    os.makedirs(pdir, exist_ok=True)
    cls_counters = {}
    for i in tqdm(range(len(ds)), 'Segments'):
        segm, cl = ds[i]
        try:
            cl_i = cls_counters[cl]
        except KeyError:
            cls_counters[cl] = 0
            cl_i = 0
        segm.save((pdir + f'/{cl}_{cl_i}.wav'))
        cls_counters[cl] += 1

def cut_beatboxset1(root, savedir, anno_type):
    ds = vxs.Beatbox1TrackSet(root, annotation_type=anno_type)
    for track, annotation in tqdm(ds.annotated_tracks(), 'Tracks'):
        person = PurePath(track.filepath).stem.split('_')[1]
        pdir = str(savedir / person)
        segments = vxs.cut_track_into_segments(
            track, annotation, classes=vxs.constants.ANNOTATION_CLASSES['beatboxset1'])
        save_segments(segments, pdir)

def cut_avp(root, savedir, subset):
    subset_dir = savedir / subset.lower()
    subset_root = root / subset
    os.makedirs(subset_dir, exist_ok=True)

    participants = sorted([
        int(PurePath(p).stem.split('_')[1])
        for p in glob(str(subset_root / 'Participant_*'))
    ])

    savedir_f = str(subset_dir / 'participant_{}')

    for p in tqdm(participants, 'Participants'):
        ds = vxs.SegmentSet(
            vxs.AVPTrackSet(root, subset=subset, participant=p, recordings_type='hits'),
            frame_window=None)
        pdir = savedir_f.format(p)
        save_segments(ds, pdir)

ENST_DRUM_TYPES = {
    'hi-hat': ['chh', 'ohh'],
    'kick': ['bd'],
    'snare': ['sd', 'sd-'],
}
        
def cut_enst(root, savedir):
    for audio_type, fit_classes in ENST_DRUM_TYPES.items():
        pdir = str(savedir / audio_type)
        print(f'Extracting {audio_type}')
        ds = vxs.ENSTDrumsTrackSet(root, audio_type=audio_type)
        for track, anno in tqdm(ds.annotated_tracks(), 'Tracks'):
            anno = anno[anno['class'].isin(fit_classes)].reset_index()
            segments = vxs.cut_track_into_segments(track, anno)
            save_segments(segments, pdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut beatbox dataset into chunks')
    parser.add_argument('type', choices=['avp', 'beatboxset1', 'enst'], type=str,
                        help='Type of input dataset')
    parser.add_argument('root', metavar='AVP_ROOT', type=PurePath,
                        help='Dataset root directory')
    parser.add_argument('save_dir', metavar='SAVE_DIR', type=PurePath,
                        help='Directory to save output files')
    parser.add_argument('--subset', choices=['Fixed', 'Personal'], type=str,
                        required='avp' in sys.argv,
                        help='Subset of AVP to choose from')
    parser.add_argument('--anno_type', choices=['DR', 'HT'], type=str,
                        required='beatboxset1' in sys.argv,
                        help='Type of annotations for beatboxset1 to use')

    args = parser.parse_args()
    if args.type == 'avp':
        cut_avp(args.root, args.save_dir, args.subset)
    elif args.type == 'beatboxset1':
        cut_beatboxset1(args.root, args.save_dir, args.anno_type)
    elif args.type == 'enst':
        cut_enst(args.root, args.save_dir)
