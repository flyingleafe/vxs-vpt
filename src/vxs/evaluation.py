import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import PurePath

import vxs

def segment_classify_all(trackset, model, lang_model=None, verbose=False,
                         predefined_bpm=False, predefined_onsets=False, **kwargs):
    total_cf = None
    for i, (track, anno) in tqdm(enumerate(trackset.annotated_tracks()), 'Analysing tracks'):
        if verbose:
            print(f'Analysing track {track.filepath}')

        onsets = anno if predefined_onsets else None
        bpm = anno.bpm if predefined_bpm else None
        analysis = vxs.segment_classify(track, model, lang_model, bpm=bpm, onsets=onsets, **kwargs)
        cf = vxs.classes_F1_score(analysis['onsets'], anno)

        if total_cf is None:
            total_cf = cf
        else:
            total_cf = total_cf.add(cf, fill_value=0)

    if 'pm' in total_cf.columns:
        total_cf = total_cf.drop('pm', 0).drop('pm', 1)
    if '' in total_cf.columns:
        total_cf = total_cf.drop('', 0).drop('', 1)

    scores = vxs.cf_to_prec_rec_F1(total_cf)

    return total_cf, scores

def dataset_onset_scores(trackset: vxs.TrackSet, methods=['hfc', 'complex', 'specflux'], **kwargs):
    columns = ['track']
    for method in methods:
        columns += [f'{method}_F1', f'{method}_prec', f'{method}_rec']

    scores = pd.DataFrame(columns=columns)

    for (i, (track, annotation)) in tqdm(enumerate(trackset.annotated_tracks()), 'Detecting onsets'):
        row = [PurePath(track.filepath).stem]
        for method in methods:
            onsets_pred = vxs.detect_onsets(track, method=method, **kwargs)

            f1, prec, rec = \
                vxs.onsets_F1_score(onsets_pred['time'].values, annotation['time'].values, prec_rec=True, **kwargs)

            row += [f1, prec, rec]

        scores.loc[i] = row

    return scores

def dataset_bpms(trackset: vxs.TrackSet, method='complex', **kwargs):
    df = pd.DataFrame(columns=['track', 'real_bpm', 'predicted_bpm'])

    for track, anno in tqdm(trackset.annotated_tracks(), 'Detecting tempo'):
        _, pred_bpm = vxs.detect_beats(track, method=method, **kwargs)
        df.loc[len(df)] = [PurePath(track.filepath).stem, anno.bpm, pred_bpm]
    return df
