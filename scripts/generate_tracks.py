import numpy as np
import os
import glob
import tensorflow as tf
import argparse

from tqdm import tqdm
from pathlib import PurePath

import vxs
import note_seq

PRIMERS = [
    ['kd', '', '', '', 'sd', '', 'kd', '', '', '', 'kd', '', 'sd'],
    ['kd', '', 'hhc', '', 'sd', '', 'hhc'],
    ['kd', '', 'hhc', '', 'sd', '', 'hho'],
    ['kd', '', '', 'kd', '', '', 'hhc', '', 'kd'],
    ['kd', '', 'hhc', 'hhc', 'kd'],
    ['kd', '', 'hhc', 'hhc'],
    ['kd', '', '', 'kd', '', '', 'hho'],
    ['kd', 'sd', '', 'sd', 'kd'],
    ['kd', 'hhc', 'hhc', 'hhc'],
    ['kd', '', '', 'hhc', '', 'kd', 'sd']
    # need more?
]

SF2_INCLUDED_SAMPLES_FIXED = {
    1: ['sd_12', 'kd_23', 'kd_19', 'sd_25', 'hhc_29', 'hhc_1', 'hho_8', 'hho_15'],
    2: ['sd_13', 'kd_4', 'kd_23', 'sd_21', 'hhc_15', 'hhc_4', 'hho_23', 'hho_7'],
    3: ['sd_4', 'kd_8', 'kd_22', 'sd_23', 'hhc_23', 'hhc_16', 'hho_22', 'hho_0'],
    4: ['sd_3', 'kd_0', 'kd_17', 'sd_14', 'hhc_11', 'hhc_15', 'hho_8', 'hho_24'],
    5: ['sd_1', 'kd_4', 'kd_10', 'sd_0', 'hhc_0', 'hhc_19', 'hho_0', 'hho_1']
}

LONG_TO_SHORT = {
    36: 35,
    38: 27,
    42: 44,
    46: 67,
}

def generate_track(model, primer, length,
                   bpm=100, temperature=1.0,
                   adapt_short_sounds=True, longer_sounds=True):
    primer_drums = vxs.classes_to_drums(primer)
    track_raw = model.generate_drum_track(length, primer_drums, temperature=temperature, branch_factor=1)

    if adapt_short_sounds:
        # use shorter sounds
        track_adapted = []
        for i in range(len(track_raw)):
            if i < len(track_raw) - 1 and track_raw[i+1] is not frozenset():
                track_adapted.append(
                    frozenset(LONG_TO_SHORT[e] for e in track_raw[i]))
            else:
                track_adapted.append(track_raw[i])
        track_raw = note_seq.DrumTrack(track_adapted, start_step=0,
                                       steps_per_bar=4, steps_per_quarter=4)

    seq = track_raw.to_sequence(qpm=bpm)
    if longer_sounds:
        for i in range(len(seq.notes)-1):
            seq.notes[i].end_time = seq.notes[i+1].start_time
        seq.notes[-1].end_time += 0.25
        seq = note_seq.sequences_lib.shift_sequence_times(seq, 0.10)
        seq.tempos[0].time = 0.0
    return seq

def add_noise(track, scale):
    track.wave = track.wave + np.random.normal(scale=scale, size=len(track.wave))
    return track

def main(soundfonts_dir, save_dir, tracks_per_sf, random_seed,
         temperature=1.0, noise=0.0, dry_run=False, no_primers=False):
    os.makedirs(save_dir, exist_ok=True)
    soundfonts = glob.glob(str(soundfonts_dir / '*.sf2'))

    if len(soundfonts) == 0:
        raise ValueError(f'No soundfonts are found in {soundfonts_dir}')

    model = vxs.load_model_from_bundle('../data/drum_kit_rnn.mag')
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    for sf in tqdm(soundfonts, 'processing soundfonts'):
        sf_name = PurePath(sf).stem
        for i in tqdm(range(tracks_per_sf), 'generating tracks'):
            if no_primers:
                primer = ['kd']
            else:
                primer_idx = np.random.choice(len(PRIMERS))
                primer = PRIMERS[primer_idx]
            length = 32*np.random.randint(2, 4)
            bpm = np.random.randint(80, 101)
            track = generate_track(model, primer, length, bpm, temperature)

            track_name = f'{sf_name}_{i}'
            sound = vxs.Track(note_seq.fluidsynth(track, sample_rate=44100, sf2_path=sf))
            if noise > 0:
                sound = add_noise(sound, noise)

            if dry_run:
                print(f'sequence generated: length {length}, bpm {track.tempos[0].qpm} {track.tempos[0].time}')
            else:
                sound.save(save_dir / (track_name + '.wav'))
                note_seq.sequence_proto_to_midi_file(track, save_dir / (track_name + '.mid'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate beatbox-like tracks')
    parser.add_argument('sf_dir', metavar='SOUNDFONTS_DIR', type=PurePath,
                        help='Directory containing soundfonts')
    parser.add_argument('save_dir', metavar='SAVE_DIR', type=PurePath,
                        help='Directory to save output files')
    parser.add_argument('-n', '--tracks_per_sf', type=int, dest='tracks_per_sf',
                        help='Number of tracks to generate per soundfont')
    parser.add_argument('--seed', type=int, dest='seed', default=42,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, dest='temperature', default=1.0,
                        help='Generative model temperature (larger temperature - more random tracks)')
    parser.add_argument('--noise', type=float, dest='noise', default=0.0,
                        help='Amplitude of noise to add to the track')
    parser.add_argument('--dry_run', action='store_true',
                        help='Do not write files; only report the data stats')
    parser.add_argument('--no_primers', action='store_true',
                        help='Use predefined primer sequences')

    args = parser.parse_args()

    main(args.sf_dir, args.save_dir, args.tracks_per_sf,
         random_seed=args.seed, temperature=args.temperature,
         noise=args.noise, dry_run=args.dry_run, no_primers=args.no_primers)
