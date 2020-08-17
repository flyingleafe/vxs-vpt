import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from pathlib import PurePath
from IPython import display

import vxs
import vxs.utils as vxsu

def segment_classify_all(trackset, model, lang_model=None, verbose=False,
                         save_dir=None, reuse_saved=True,
                         predefined_bpm=False, predefined_onsets=False, **kwargs):
    total_cf = None
    for i, (track, anno) in tqdm(enumerate(trackset.annotated_tracks()), 'Analysing tracks'):
        if verbose:
            print(f'Analysing track {track.filepath}')

        onsets = anno if predefined_onsets else None
        bpm = anno.bpm if predefined_bpm else None

        analysis = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = (PurePath(save_dir) / PurePath(track.filepath).stem).with_suffix('.csv')

            if reuse_saved:
                try:
                    saved_onsets = pd.read_csv(save_path)
                    analysis = {'onsets': saved_onsets}
                except FileNotFoundError:
                    pass

        if analysis is None:
            analysis = vxs.segment_classify(track, model, lang_model, bpm=bpm, onsets=onsets, **kwargs)

            if save_dir is not None:
                analysis['onsets'].to_csv(save_path)

        cf = vxs.classes_conf_matrix(analysis['onsets'], anno)

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


def evaluate_with_different_data_sizes(training_set, eval_set, lang_model, model_fn, savedir_pattern,
                                       percentages=[10, 25, 40, 55, 70, 85, 100], random_seed=126,
                                       reuse_saved=True, verbose=True, **kwargs):
    evaluation_cfs = {}
    for percentage in percentages:
        p = percentage / 100
        savedir = savedir_pattern.format(percentage)

        evaluation_cfs[percentage] = {}

        if verbose:
            print(f'{percentage}% of dataset')
            print('Classifier training...')
        ds = vxs.stratified_subset(training_set, p, random_seed=random_seed)
        X, y = vxsu.unzip_dataset(ds)
        model = model_fn()
        model.fit(X, y)

        if verbose:
            print('Evaluating (no LM)')
        nolm_cf, nolm_scores = vxs.segment_classify_all(eval_set, model,
                                                        save_dir=savedir+'/nolm', reuse_saved=reuse_saved,
                                                        quantization_conflict_resolution=None, **kwargs)
        evaluation_cfs[percentage]['nolm'] = nolm_cf

        if verbose:
            display.display(nolm_scores)
            display.display(nolm_scores.mean())
            print('Evaluating (LM)')

        lm_cf, lm_scores = vxs.segment_classify_all(eval_set, model, lang_model,
                                                    save_dir=savedir+'/lm', reuse_saved=reuse_saved,
                                                    **kwargs)
        evaluation_cfs[percentage]['lm'] = lm_cf

        if verbose:
            display.display(lm_scores)
            display.display(lm_scores.mean())

    return evaluation_cfs

def metrics_series(eval_cfs, model_type):
    series = {
        'keys': sorted(list(eval_cfs.keys())),
        'prec': {
            'kd': [],
            'sd': [],
            'hhc': [],
            'hho': [],
            'mean': [],
        },
        'rec': {
            'kd': [],
            'sd': [],
            'hhc': [],
            'hho': [],
            'mean': [],
        },
        'F1': {
            'kd': [],
            'sd': [],
            'hhc': [],
            'hho': [],
            'mean': [],
        }
    }
    classes = ['kd', 'sd', 'hhc', 'hho']
    for k in series['keys']:
        cf = eval_cfs[k][model_type]
        scores = vxs.cf_to_prec_rec_F1(cf)
        scores_mean = scores.mean()
        for metric in ['prec', 'rec', 'F1']:
            for cl in classes:
                series[metric][cl].append(scores[metric][cl])
            series[metric]['mean'].append(scores_mean[metric])
    return series
