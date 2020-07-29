import pandas as pd
import mir_eval

from pathlib import PurePath

from .dataset import TrackSet
from .track import *

def onsets_F1_score(pred, target, ms_threshold=50, prec_rec=False):
    f1, prec, rec = mir_eval.onset.f_measure(target, pred, ms_threshold / 1000.0)
    if prec_rec:
        return f1, prec, rec
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
    classes = list(cf.columns)
    classes.remove('sil')
    res = pd.DataFrame(columns=['prec', 'rec', 'F1'], index=classes)
    for cl in classes:
        tp = cf.loc[cl, cl]
        fp = cf.loc[:, cl].sum() - tp
        fn = cf.loc[cl, :].sum() - tp
        if tp == 0:
            prec = 0
            rec = 0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
        if prec == 0 or rec == 0:
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        res.loc[cl] = [prec, rec, f1]
    return res
