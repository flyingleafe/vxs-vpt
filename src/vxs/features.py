import numpy as np
import librosa as lr
import torch
import torch.nn.functional as F

from torchaudio.transforms import AmplitudeToDB

def mel_specgram(track, n_mels=128, win_size=4096, hop_size=512, to_db=True, **kwargs):
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
         

def mfcc(track, n_mfcc=20, win_size=4096, hop_size=512, **kwargs):
    return lr.feature.mfcc(track.wave, track.rate,
                           n_mfcc=n_mfcc, n_fft=win_size, hop_length=hop_size, **kwargs)


def ramires_features(track):
    # TODO: fill more features
    fft = lr.core.fft.get_fftlib()
    
    DEFAULT_SR = 44100
    N_FFT = 4096
    
    S = np.expand_dims(np.abs(fft.rfft(track.wave, n=N_FFT)), -1)
    freq = lr.core.time_frequency.fft_frequencies(sr=DEFAULT_SR, n_fft=N_FFT)
    
    centroid = lr.feature.spectral_centroid(S=S, freq=freq)
    spread = lr.feature.spectral_bandwidth(S=S, freq=freq, centroid=centroid)
    slope = lr.feature.poly_features(S=S, freq=freq)[:1, :]
    # decrease?
    rolloff = lr.feature.spectral_rolloff(S=S, freq=freq, roll_percent=0.95)
    # skewness?
    # flux?
    # curtosis?
    flatness = lr.feature.spectral_flatness(S=S)  # for different bands?
    mfccs = lr.feature.mfcc(S=S, n_mfcc=20)
    # bfccs?
    
    return np.concatenate([centroid, spread, slope, rolloff, flatness, mfccs])