import numpy as np
import pandas as pd

import note_seq

from note_seq.drums_encoder_decoder import DEFAULT_DRUM_TYPE_PITCHES
from vxs import constants

_DRUM_MAP = dict(zip(['kd', 'sd', 'hhc', 'hho'], DEFAULT_DRUM_TYPE_PITCHES))
_INV_DRUM_MAP = dict((pitch, cl) for cl, pitches in _DRUM_MAP.items() for pitch in pitches)

def classes_to_drums(classes):
    sets = [
        frozenset()
        if cl == '' or cl == 'sil'
        else frozenset([constants.MIDI_PITCHES[cl]])
        for cl in classes
    ]
    return note_seq.DrumTrack(sets, start_step=0, steps_per_bar=4, steps_per_quarter=4)

def note_seq_to_annotation(seq, ignore_non_drums=False):
    anno = pd.DataFrame(columns=['time', 'class'])
    for note in seq.notes:
        try:
            cl = _INV_DRUM_MAP[note.pitch]
            time = note.start_time
            anno.loc[len(anno)] = [time, cl]
        except KeyError:
            if not ignore_non_drums:
                raise ValueError(f'Encountered non-drum note: {note}')

    return anno
