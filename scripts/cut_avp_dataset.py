import numpy as np
import pandas as pd
import os
import argparse

from tqdm import tqdm
from glob import glob
from pathlib import PurePath

import vxs

def main(root, savedir, subset):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut AVP dataset into chunks')
    parser.add_argument('avp_root', metavar='AVP_ROOT', type=PurePath,
                        help='AVP Dataset root directory')
    parser.add_argument('save_dir', metavar='SAVE_DIR', type=PurePath,
                        help='Directory to save output files')
    parser.add_argument('subset', choices=['Fixed', 'Personal'], type=str,
                        help='Subset of AVP to choose from')

    args = parser.parse_args()
    main(args.avp_root, args.save_dir, args.subset)
