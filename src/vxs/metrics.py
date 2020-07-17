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
    
def classes_F1_score(pred, target, classes=None, ms_threshold=50, confusion_matrix=True):
    thr = ms_threshold / 1000
    
    if classes is None:
        classes = np.append(np.union1d(pred['class'].unique(), target['class'].unique()), 'sil')
    else:
        classes = np.append(classes, 'sil')
        
    confusion_matrix = pd.DataFrame(columns=classes, index=classes).fillna(0)
    
    i = 0
    j = 0
    
    while i < len(pred) or j < len(target):
        if i >= len(pred):
            true_class = target.loc[j, 'class']
            confusion_matrix.loc[true_class, 'sil'] += 1
            j += 1
        elif j >= len(target):
            pred_class = pred.loc[i, 'class']
            confusion_matrix.loc['sil', pred_class] += 1
            i += 1
        elif pred.loc[i, 'time'] >= target.loc[j, 'time'] - thr and pred.loc[i, 'time'] <= target.loc[j, 'time'] + thr:
            pred_class = pred.loc[i, 'class']
            true_class = target.loc[j, 'class']
            confusion_matrix.loc[true_class, pred_class] += 1
            i += 1
            j += 1
        elif pred.loc[i, 'time'] < target.loc[j, 'time'] - thr:
            pred_class = pred.loc[i, 'class']
            confusion_matrix.loc['sil', pred_class] += 1
            i += 1
        else:
            true_class = target.loc[j, 'class']
            confusion_matrix.loc[true_class, 'sil'] += 1
            j += 1
    
    return confusion_matrix

def cf_to_prec_rec_F1(cf):
    classes = cf.columns
    classes.remove('sil')
    res = pd.DataFrame(columns=['prec', 'rec', 'F1'], index=classes)
    for cl in classes:
        tp = cf.loc[cl, cl]
        fp = cf.loc[cl, :].sum() - tp
        fn = cf.loc[:, cl].sum() - tp
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec == 0 or rec == 0:
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        res.loc[cl] = [prec, rec, f1]
    return res
    
def dataset_onset_scores(trackset: TrackSet, methods=['hfc', 'complex', 'specflux'], **kwargs):
    columns = ['track']
    for method in methods:
        columns += [f'{method}_F1', f'{method}_prec', f'{method}_rec']
    
    scores = pd.DataFrame(columns=columns)
    
    for (i, (track, annotation)) in enumerate(trackset.annotated_tracks()):
        row = [PurePath(track.filepath).stem]
        for method in methods:
            onsets_pred = detect_onsets(track, method=method)
        
            f1, prec, rec = \
                onsets_F1_score(onsets_pred['time'].values, annotation['time'].values, prec_rec=True, **kwargs)
    
            row += [f1, prec, rec]
        
        scores.loc[i] = row
    
    return scores