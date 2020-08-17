import numpy as np
import librosa as lr
import torch
import torch.nn.functional as F

from torchaudio.transforms import AmplitudeToDB
from vxs import constants

def hz_to_bark(hz):
    return 6 * np.arcsinh(hz/600)

def bark_to_hz(bark):
    return 600*np.sinh(bark/6)

def bark_frequencies(n_barks=128, fmin=60.0, fmax=13500.0, **kwargs):
    """
    Adapted from Librosa's `mel_frequencies`
    """
    min_bark = hz_to_bark(fmin)
    max_bark = hz_to_bark(fmax)
    barks = np.linspace(min_bark, max_bark, n_barks)
    return bark_to_hz(barks)

@lr.cache(level=10)
def bark(sr, n_fft, n_barks=128, fmin=60.0, fmax=13500.0, norm=1):
    """
    Adapted from Librosa's `filters.mel`
    """
    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ValueError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_barks = int(n_barks)
    weights = np.zeros((n_barks, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = lr.core.time_frequency.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    bark_f = bark_frequencies(n_barks + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(bark_f)
    ramps = np.subtract.outer(bark_f, fftfreqs)

    for i in range(n_barks):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (bark_f[2:n_barks+2] - bark_f[:n_barks])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((bark_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights

def ternhardt_ear_curve(hz):
    khz = hz / 1000.0
    curve_db = -3.64*khz**(-0.8) + 6.5*np.exp(-0.6*(khz-3.3)**2) - 0.001*khz**4
    return lr.core.db_to_power(curve_db)

def barkspectrogram(y=None, sr=44100, S=None, n_fft=2048, hop_length=512,
                    power=2.0, window="hann", center=True, pad_mode="reflect", **kwargs):
    """
    Adapted from Librosa's `melspectrogram`
    """
    S, n_fft = lr.core.spectrum._spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                                             power=power, window=window, center=center, pad_mode=pad_mode)

    # Build a Bark filter
    bark_basis = bark(sr, n_fft, **kwargs)
    return np.dot(bark_basis, S)
    
def to_ternhardt_db_scale(sgram, **kwargs):
    freqs = bark_frequencies(n_barks=sgram.shape[0], **kwargs)
    curve = ternhardt_ear_curve(freqs)
    sgram = sgram * curve.reshape(-1, 1)
    return lr.core.power_to_db(sgram)
    
def bark_specgram(track, n_mels=128,
                  win_size=constants.DEFAULT_STFT_WINDOW,
                  hop_size=constants.DEFAULT_STFT_HOP_SIZE,
                  to_ternhardt_db=True, **kwargs):
    sgram = barkspectrogram(track.wave, track.rate,
                            n_barks=n_mels, n_fft=win_size, hop_length=hop_size, **kwargs)
    if to_ternhardt_db:
        sgram = to_ternhardt_db_scale(sgram)

    return sgram
    
def mel_specgram(track, n_mels=128,
                 win_size=constants.DEFAULT_STFT_WINDOW,
                 hop_size=constants.DEFAULT_STFT_HOP_SIZE,
                 to_db=True, **kwargs):
    sgram = lr.feature.melspectrogram(track.wave, track.rate,
                                      n_mels=n_mels, n_fft=win_size, hop_length=hop_size, **kwargs)
    if to_db:
        sgram = lr.core.power_to_db(sgram)

    return sgram

def mel_specgram_cae(track, pad_time=None, device='cpu', normalize=True, **kwargs):
    S = torch.tensor(mel_specgram(track, to_db=False, **kwargs), device=device).unsqueeze(0)

    if pad_time is not None:
        if S.size()[-1] >= pad_time:
            S = S[:, :, :pad_time]
        else:
            S = F.pad(S, (0, pad_time - S.size()[-1], 0, 0, 0, 0),
                      mode='constant', value=0)

    S_db = AmplitudeToDB()(S)
    if normalize:
        S_db /= 80   # arbitrary value, but we'll see

    return S_db

def bark_specgram_cae(track, pad_time=None, device='cpu', normalize=True, **kwargs):
    S = bark_specgram(track, to_ternhardt_db=False, **kwargs)
    if pad_time is not None:
        if S.shape[1] >= pad_time:
            S = S[:, :pad_time]
        else:
            S = np.pad(S, ((0, 0), (0, pad_time - S.shape[1])))
            
    S_db = to_ternhardt_db_scale(S, **kwargs)
    if normalize:
        S_db /= 80.0
    return torch.tensor(S_db, device=device).float().unsqueeze(0)

def mfcc(track, n_mfcc=20, deltas=0, exclude_F0=False,
         win_size=constants.DEFAULT_STFT_WINDOW,
         hop_size=constants.DEFAULT_STFT_HOP_SIZE, **kwargs):
    if exclude_F0:
        n_mfcc += 1

    mfccs = lr.feature.mfcc(track.wave, track.rate,
                            n_mfcc=n_mfcc, n_fft=win_size, hop_length=hop_size, **kwargs)
    if exclude_F0:
        mfccs = mfccs[1:]

    if deltas > 0:
        feats = [mfccs]
        for order in range(1, deltas+1):
            feats.append(lr.feature.delta(mfccs, order=order, mode='nearest'))

        mfccs = np.vstack(feats)
    return mfccs


def spectral_decrease(S):
    S_sum = S.sum(axis=0)
    freq_ixs = np.arange(1, S.shape[0]).reshape(-1, 1)
    S_diff = (S[1:] - S[0]) / freq_ixs
    S_diff_sum = S_diff.sum(axis=0)
    return np.divide(S_diff_sum, S_sum, out=np.zeros_like(S_diff_sum), where=S_sum!=0)
    
def spectral_moment(S, freq, centroid, p=2):
    bdw = lr.feature.spectral_bandwidth(S=S, freq=freq, centroid=centroid)**p
    sm = S.sum(axis=0)
    return np.divide(bdw, sm, out=np.zeros_like(bdw), where=sm!=0)
    
def spectral_skewness(S, freq, spread, centroid):
    moment_3 = spectral_moment(S, freq, centroid, 3)
    return np.divide(moment_3, np.sqrt(spread)**3, out=np.zeros_like(moment_3), where=spread!=0)

def spectral_kurtosis(S, freq, spread, centroid):
    moment_4 = spectral_moment(S, freq, centroid, 4)
    return np.divide(moment_4, spread**2, out=np.zeros_like(moment_4), where=spread!=0)

def ramires_features(track):
    """
    Reimplementation of Ramires' feature extraction routine
    """
    FRAME_LEN = 4096
    HOP_LEN = 512
    
    # temporal features
    rms = lr.feature.rms(track.wave, frame_length=FRAME_LEN, hop_length=HOP_LEN, center=False)
    zero_crossings = lr.util.frame(lr.zero_crossings(track.wave), FRAME_LEN, HOP_LEN).sum(axis=0).reshape(1, -1)
    
    # spectral features
    S = np.abs(lr.stft(track.wave, n_fft=FRAME_LEN, hop_length=HOP_LEN, center=False))
    freq = lr.fft_frequencies(sr=track.rate, n_fft=FRAME_LEN)
    
    centroid = lr.feature.spectral_centroid(S=S, freq=freq)
    spread = spectral_moment(S, freq, centroid)
    slope = lr.feature.poly_features(S=S, freq=freq)[:1, :]
    decrease = spectral_decrease(S).reshape(1, -1)
    rolloff = lr.feature.spectral_rolloff(S=S, freq=freq, roll_percent=0.95)
    skewness = spectral_skewness(S, freq, spread, centroid)
    specflux = lr.onset.onset_strength(S=S).reshape(1, -1)
    kurtosis = spectral_kurtosis(S, freq, spread, centroid)
    
    # flatness for different bands
    band_boundaries = np.searchsorted(freq, [250, 500, 1000, 2000, 4000])
    flat_bands = []
    for i in range(len(band_boundaries)-1):
        l = band_boundaries[i]
        r = band_boundaries[i+1]
        flat_bands.append(lr.feature.spectral_flatness(S=S[l:r]))    
    flatness = np.vstack(flat_bands)
    
    S_mel = lr.power_to_db(lr.feature.melspectrogram(S=S**2))
    S_bark = lr.power_to_db(barkspectrogram(S=S**2))
    mfccs = lr.feature.mfcc(S=S_mel, n_mfcc=20)
    bfccs = lr.feature.mfcc(S=S_bark, n_mfcc=20)
    
    all_features = [
        rms,
        zero_crossings,
        centroid,
        spread,
        slope,
        decrease,
        rolloff,
        skewness,
        specflux,
        kurtosis,
        flatness,
        mfccs,
        bfccs
    ]
    
    return np.vstack(all_features)
