import pandas as pd

from pathlib import PurePath

from .dataset import TrackSet
from .track import *

def onsets_F1_score(pred, target, ms_threshold=50, prec_rec=False):
    thr = ms_threshold / 1000
    i = 0
    j = 0
    tp = 0
    fp = 0
    fn = 0
    while i < len(pred) or j < len(target):
        if i >= len(pred):
            fn += 1
            j += 1
        elif j >= len(target):
            fp += 1
            i += 1
        elif pred[i] >= target[j] - thr and pred[i] <= target[j] + thr:
            tp += 1
            i += 1
            j += 1
        elif pred[i] < target[j] - thr:
            fp += 1
            i += 1
        else:
            fn += 1
            j += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    if prec_rec:
        return f1, precision, recall
    else:
        return f1
    
def dataset_onset_scores(trackset: TrackSet):
    scores = pd.DataFrame(columns=[
        'track',
        'HFC_F1', 'HFC_prec', 'HFC_rec',
        'Complex_F1', 'Complex_prec', 'Complex_rec'
    ])
    
    for (i, (track, annotation)) in enumerate(trackset.annotated_tracks()):
        onsets_pred_hfc = detect_onsets(track, method='hfc')
        onsets_pred_cp = detect_onsets(track, method='complex')
        
        f1_hfc, prec_hfc, rec_hfc = \
            onsets_F1_score(onsets_pred_hfc['time'].values, annotation['time'].values, prec_rec=True)
    
        f1_cp, prec_cp, rec_cp = \
            onsets_F1_score(onsets_pred_cp['time'].values, annotation['time'].values, prec_rec=True)
        
        scores.loc[i] = [
            PurePath(track.filepath).stem,
            f1_hfc, prec_hfc, rec_hfc,
            f1_cp, prec_cp, rec_cp
        ]
    
    return scores