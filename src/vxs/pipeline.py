import numpy as np
import pandas as pd
import note_seq.sequences_lib as notes

from .track import *

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


# def segment_classify_track(track, classifier, segm_frames=4096,
#                            onset_method='complex'):
#     """
#     Segment and classify 
#     """

def quantize_onsets(onset_times, bpm, steps_per_quarter=4,
                    first_onset_zero=True, quantize_cutoff=0.5, **kwargs):
    """
    Given a BPM, quantizes the onsets to the nearest smallest note step
    """
    first_offset = 0.0
    if first_onset_zero:
        first_offset = onset_times[0]
        onset_times = onset_times - first_offset
        
    steps_per_second = notes.steps_per_quarter_to_steps_per_second(steps_per_quarter, bpm)
    onset_steps = (onset_times * steps_per_second + (1 - quantize_cutoff)).astype('int')
    onset_times_quantized = onset_steps.astype('float32') / steps_per_second + first_offset
    
    return onset_steps, onset_times_quantized

def onsets_probas_to_pianoroll(onset_steps, probas, **kwargs):
    """
    Given onset steps and class probabilities, return a 2d
    matrix with probabilities of each class (and silences) for each
    note frame
    """
    num_frames = onset_steps[-1] + 1
    num_classes = probas.shape[1]
    probas_sil = np.hstack((probas, np.zeros((len(probas), 1))))
    
    roll = np.zeros((num_frames, num_classes + 1))
    roll[:, num_classes] = 1.0       # all frames without detected onsets are silence with probability 1
    roll[onset_steps] = probas_sil
    
    return roll

def segment_classify(track, classifier, onset_method='complex', **kwargs):
    """
    Perform full segment + classify procedure on a track
    """
    onsets = detect_onsets(track, method=onset_method)
    _, bpm = detect_beats(track, method=onset_method)
    onset_times = onsets['time'].values
    
    onset_steps, onsets_quantized = quantize_onsets(onset_times, bpm, **kwargs)
    segments = segment_track(track, onset_times, **kwargs)
    probas = classifier.predict_proba(segments)
    roll = onsets_probas_to_pianoroll(onset_steps, probas, **kwargs)
    
    # TODO: insert language model
    
    classes = np.array(sorted(classifier.classes_))
    preds = classes[np.argmax(probas, axis=1)]
    
    onsets['step'] = onset_steps
    onsets['time_quantized'] = onsets_quantized
    onsets['class'] = preds
    
    return onsets, roll