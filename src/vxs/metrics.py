import pandas as pd
import mir_eval

from pathlib import PurePath
from scipy import optimize

from .dataset import TrackSet
from .track import *

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
    target_classes = target['class'].values
    pred_classes = pred['class'].values

    # solve the optimal assignment between target and prediction (true positive matches are preferred)
    # for some reasons, doesn't really work
    # hits = mir_eval.util._fast_hit_windows(target_times, pred_times, thr)
    # edges = list(zip(*hits))
    # weights = np.zeros(len(edges))
    # A_eq_t = np.zeros((len(target_times), len(edges)))
    # A_eq_p = np.zeros((len(pred_times), len(edges)))

    # for i, (t_ix, p_ix) in enumerate(edges):
        # weights[i] = 2.0 if target_classes[t_ix] == pred_classes[p_ix] else 1.0
        # A_eq_t[t_ix, i] = 1.0
        # A_eq_p[p_ix, i] = 1.0

    # A_eq = np.vstack((A_eq_t, A_eq_p))
    # b_eq = np.ones(A_eq.shape[0])
    # res = optimize.linprog(-weights, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
    # chosen_edges = np.arange(len(edges))[res.x.astype(bool)]
    # chosen_t = np.zeros(len(target_times)).astype(bool)
    # chosen_p = np.zeros(len(pred_times)).astype(bool)

    # for e_ix in chosen_edges:
        # t_ix, p_ix = edges[e_ix]
        # true_class = target_classes[t_ix]
        # pred_class = pred_classes[p_ix]
        # confusion_matrix.loc[true_class, pred_class] += 1
        # chosen_t[t_ix] = True
        # chosen_p[p_ix] = True

    # for true_class in target_classes[~chosen_t]:
        # confusion_matrix.loc[true_class, 'sil'] += 1
    # for pred_class in pred_classes[~chosen_p]:
        # confusion_matrix.loc['sil', pred_class] += 1

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
