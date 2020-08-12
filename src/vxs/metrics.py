import pandas as pd
import mir_eval

from pathlib import PurePath
from scipy import optimize

from vxs.dataset import TrackSet
from vxs.track import *
from vxs.constants import EVENT_CLASSES, EVENT_CLASS_IXS

def onsets_F1_score(pred, target, ms_threshold=50, prec_rec=False, **kwargs):
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

    target_times = target['time'].values
    pred_times = pred['time'].values
    target_classes = np.array([EVENT_CLASS_IXS[cl] for cl in target['class'].values])
    pred_classes = np.array([EVENT_CLASS_IXS[cl] for cl in pred['class'].values])

    # time_hit_matrix = np.abs(np.subtract.outer(target_times, pred_times)) <= thr
    # class_hit_matrix = np.equal.outer(target_classes, pred_classes)
    # hit_matrix = time_hit_matrix * class_hit_matrix
    # hits = np.where(hit_matrix)

    # G = {}
    # for ref_i, est_i in zip(*hits):
    #     if est_i not in G:
    #         G[est_i] = []
    #     G[est_i].append(ref_i)

    # matching = sorted(mir_eval.util._bipartite_match(G).items())


    # for ref_i, est_i in matching:
    #     cl_ix = target_classes[ref_i]
    #     assert cl_ix == pred_classes[est_i]
    #     cl = EVENT_CLASSES[cl_ix]


    i = 0
    j = 0

    while i < len(pred) or j < len(target):
        if i >= len(pred):
            true_class = target_classes[j]
            confusion_matrix.loc[true_class, 'sil'] += 1
            j += 1
        elif j >= len(target):
            pred_class = pred_classes[i]
            confusion_matrix.loc['sil', pred_class] += 1
            i += 1
        elif pred_times[i] >= target_times[j] - thr and pred_times[i] <= target_times[j] + thr:
            pred_class = pred_classes[i]
            true_class = target_classes[j]
            confusion_matrix.loc[true_class, pred_class] += 1
            i += 1
            j += 1
        elif pred_times[i] < target_times[j] - thr:
            pred_class = pred_classes[i]
            confusion_matrix.loc['sil', pred_class] += 1
            i += 1
        else:
            true_class = target_classes[j]
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
