import numpy as np
import pandas as pd
import note_seq.sequences_lib as notes

from .track import *
from .language_model import drum_track_to_mono_classes

def segment_track(track, onset_times, segm_frames=4096, **kwargs):
    """
    Given the onset times, fetch the segments of the
    fixed length from the track.

    Returns:
        times     - np.array containing the time stamps of segments beginning
        segments  - np.array containing the segments (stacked)
    """
    segments = []
    for time in onset_times:
        segm = track.segment_frames(int(track.rate * time), segm_frames)
        segments.append(segm)

    return np.stack(segments)


def quantize_onsets(onset_times, bpm, steps_per_quarter=4,
                    first_onset_zero=True, quantization_conflict_resolution='delete',
                    quantize_cutoff=0.5, **kwargs):
    """
    Given a BPM, quantizes the onsets to the nearest smallest note step
    """
    first_offset = 0.0
    if first_onset_zero:
        first_offset = onset_times[0]
        onset_times = onset_times - first_offset

    steps_per_second = notes.steps_per_quarter_to_steps_per_second(steps_per_quarter, bpm)
    onset_steps = (onset_times * steps_per_second + (1 - quantize_cutoff)).astype('int')

    deleted_onsets = []
    if quantization_conflict_resolution is not None:
        if quantization_conflict_resolution == 'shift':
            # TODO: come up with something better for conflict resolution maybe?
            # because it doesn't really work well and can shift onsets a lot towards the grid
            # actual solution would be removing the weaker onset (should somehow
            # return onset strength then eh?)
            for i in range(len(onset_steps)-1):
                stp = onset_steps[i]
                if stp == onset_steps[i+1]:
                    if i > 0 and stp-1 == onset_steps[i-1]:
                        # move forward, no choice
                        onset_steps[i+1] += 1
                    else:
                        d1 = np.abs(onset_times[i] * steps_per_second - stp)
                        d2 = np.abs(onset_times[i+1] * steps_per_second - stp)
                        if d1 >= d2:
                            onset_steps[i] -= 1
                        else:
                            onset_steps[i+1] += 1

        elif quantization_conflict_resolution == 'delete':
            i = 0
            while i < len(onset_steps) - 1:
                if onset_steps[i] == onset_steps[i+1]:
                    deleted_onsets.append(i+1)
                    onset_steps = np.delete(onset_steps, i+1)
                else:
                    i += 1

    onset_times = onset_times + first_offset
    if deleted_onsets:
        onset_times = np.delete(onset_times, deleted_onsets)
    onset_times_quantized = onset_steps.astype('float32') / steps_per_second + first_offset

    assert len(onset_steps) == len(onset_times)
    assert len(onset_steps) == len(onset_times_quantized)
    return onset_steps, onset_times, onset_times_quantized

_COUNTED_CLASS_PROBAS = np.array([
    0.19744821667574,
    0.35148473660812984,
    0.27987600054202216,
    0.171191046174108,
])

_COUNTED_CLASS_PROBAS_DISCOUNTED = np.array([
    0.17397626019336182,
    0.38819583766243526,
    0.2754395642454968,
    0.16238833789870616,
])

def onsets_probas_to_pianoroll(onset_steps, probas,
                               onehot_len=512, silence_prob=0.0, **kwargs):
    """
    Given onset steps and class probabilities, return a 2d
    matrix with probabilities of each class (and silences) for each
    note frame
    """
    num_frames = onset_steps[-1] + 1
    num_classes = probas.shape[1]
    probas = probas / _COUNTED_CLASS_PROBAS_DISCOUNTED   # do that in order to obtain scaled likelihoods
    # TODO: how to combine with silences?

    probas_sil = np.hstack((np.ones((len(probas), 1))*silence_prob, probas*(1-silence_prob)))

    if onehot_len is None:
        onehot_len = 2 ** num_classes
    onehot_idxs = np.concatenate(([0], 2**np.arange(num_classes)))

    roll = np.zeros((num_frames, onehot_len))
    roll[:, 0] = 1.0       # all frames without detected onsets are silence with probability 1
    for i, i2 in enumerate(onehot_idxs):
        roll[onset_steps, i2] = probas_sil[:, i]

    return roll

def drum_track_to_onset_steps(drum_track, class_order):
    mono_classes = drum_track_to_mono_classes(drum_track)
    steps = np.arange(len(mono_classes))[mono_classes != 0]  # remove silence
    class_idxs = np.log2(mono_classes[steps]).astype('int')
    return steps, np.array(class_order)[class_idxs]

def segment_classify(track, classifier, lang_model=None, bpm=None, onsets=None,
                     onset_method='complex', remove_unquantized_onsets=False,
                     class_order=['kd', 'sd', 'hhc', 'hho'], softmax_size=512, **kwargs):
    """
    Perform full segment + classify procedure on a track
    """
    if bpm is None:
        _, bpm = detect_beats(track, method=onset_method)

    if onsets is None:
        onsets = detect_onsets(track, method=onset_method)
    onset_times = onsets['time'].values

    onset_steps, new_onset_times, onsets_quantized = quantize_onsets(onset_times, bpm, **kwargs)
    if lang_model is not None or remove_unquantized_onsets:
        onset_times = new_onset_times

    segments = segment_track(track, onset_times, **kwargs)
    probas = classifier.predict_proba(segments)

    if lang_model is not None:
        order_reindex = np.argsort(np.argsort(class_order))
        probas_reindexed = probas[:, order_reindex]
        roll = onsets_probas_to_pianoroll(onset_steps, probas_reindexed, **kwargs)

        generated_events = lang_model.modify_observation_probas(roll, **kwargs)
        lang_onset_steps, preds = drum_track_to_onset_steps(generated_events, class_order)

        # TODO: how to meaningfully allow the model to add new onsets or discard old ones?
        # if not np.array_equal(onset_steps, lang_onset_steps):
        #     print('init onsets', onset_steps)
        #     print('then onsets', lang_onset_steps)
        #     raise ValueError('onsets dont match')
        onset_included = np.zeros(len(onset_steps)).astype(bool)
        j = 0
        for i in range(len(onset_steps)):
            if j < len(lang_onset_steps) and onset_steps[i] == lang_onset_steps[j]:
                onset_included[i] = True
                j += 1

        onset_times = onset_times[onset_included]
        onset_steps = lang_onset_steps
        onsets_quantized = onsets_quantized[onset_included]

    else:
        classes = np.array(sorted(classifier.classes_))
        preds = classes[np.argmax(probas, axis=1)]

    onsets = pd.DataFrame.from_dict({
        'time': onset_times,
        'step': onset_steps,
        'time_quantized': onsets_quantized,
        'class': preds
    })

    result = {
        'onsets': onsets,
        'bpm': bpm
    }

    if lang_model is not None:
        result['note_seq'] = generated_events.to_sequence(qpm=bpm)

    return result
