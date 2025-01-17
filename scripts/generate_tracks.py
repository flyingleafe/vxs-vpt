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
    track_raw = model.generate_drum_track(length, primer_drums, temperature=temperature)

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

# def window_segm(segm):
#     n = len(segm)
#     w = np.ones(n)
#     front_w = lr.filters.get_window('hann', 256, False)[:128]
#     back_w = lr.filters.get_window('hann', 1024, False)[512:]
#     w[:128] = front_w
#     w[-512:] = back_w
#     return segm * w

# def synthesize_track(seq, samples):
#     class_samples = {
#         cl: samples.tracks[samples.classes == cl]
#         for cl in ['kd', 'sd', 'hhc', 'hho']
#     }

#     track_start_time = 0.3
#     fr_offset = int(track_start_time * 44100)
#     track_end_time = seq.notes[-1].end_time + track_start_time + 0.5
#     wave = np.zeros(int(track_end_time * 44100))
#     for note in seq.notes:zz
#         cl = vxs.constants.MIDI_PITCHES_INV[note.pitch]
#         start_fr = int(note.start_time * 44100) + fr_offset
#         end_fr = int(note.end_time * 44100) + fr_offset
#         sample = np.random.choice(class_samples[cl]).segment_frames(0, end_fr - start_fr).wave
#         sample = window_segm(sample)
#         real_end_fr = min(end_fr, start_fr+len(sample))
#         wave[start_fr:real_end_fr] = window_segm(sample)

#     return vxs.Track(wave)

def add_noise(track, scale):
    track.wave = track.wave + np.random.normal(scale=scale, size=len(track.wave))
    return track

def main(soundfonts_dir, save_dir, tracks_per_sf, random_seed,
         temperature=1.0, noise=0.0, dry_run=False, no_primers=False):
    os.makedirs(save_dir, exist_ok=True)
    soundfonts = glob.glob(str(soundfonts_dir / '*.sf2'))
    # soundfonts = [(soundfonts_dir / f'participant_{i}') for i in range(1, 6)]

    if len(soundfonts) == 0:
        raise ValueError(f'No soundfonts are found in {soundfonts_dir}')

    model = vxs.load_model_from_bundle('../data/drum_kit_rnn.mag')
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    for sf in tqdm(soundfonts, 'processing soundfonts'):
        sf_name = PurePath(sf).stem
        # sampleset = vxs.SimpleSampleSet(glob.glob(str(sf / '*.wav')))

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
            # sound = synthesize_track(track, sampleset)
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
