import numpy as np
import torch
import abc

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import LeaveOneOut
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from vxs.features import *
from vxs.dataset import *
from vxs import constants

class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=10, covariance_type='full', **kwargs):
        self.gmm_args = {
            'n_components': n_components,
            'covariance_type': covariance_type,
            **kwargs
        }
        self.mixtures = None

    def get_params(self, deep=True):
        return self.gmm_args

    def set_params(self, **params):
        self.gmm_args = params

    def fit(self, X, y):
        self.classes_ = np.array(sorted(np.unique(y)))
        self.mixtures = []

        for cl in self.classes_:
            X_cl = X[y == cl]
            mixture = GaussianMixture(**self.gmm_args)
            mixture.fit(X_cl)
            self.mixtures.append(mixture)

    def predict_log_proba(self, X):
        cl_log_probas = []
        for mx in self.mixtures:
            cl_log_probas.append(mx.score_samples(X).reshape(-1, 1))
        return np.hstack(cl_log_probas)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        cl_ixs = np.argmax(log_probas, axis=1)
        return self.classes_[cl_ixs]

class ClassicFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, feature_type, frame_len=constants.DEFAULT_STFT_WINDOW, **kwargs):
        if feature_type not in ('mfcc', 'ramires'):
            raise ValueError('Unknown feature type:' + feature_type)
        self.feature_type = feature_type
        self.frame_len = frame_len

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = []
        for segm in X:
            segm = segm.cut_or_pad(self.frame_len)
            if self.feature_type == 'mfcc':
                features = mfcc(segm, center=False).ravel()
            elif self.feature_type == 'ramires':
                features = ramires_features(segm).ravel()
            X_.append(features)
        return np.array(X_)

class CAEFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, encoder, sgram_type='mel', frame_len=16384, **kwargs):
        self.encoder = encoder
        self.frame_len = frame_len
        self.sgram_type = sgram_type
        self.encoder.eval()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = []
        for segm in X:
            pad_sgram = None
            if self.frame_len is not None:
                segm = segm.cut_or_pad(self.frame_len)
                pad_sgram = self.frame_len // 512

            if self.sgram_type == 'mel':
                S = mel_specgram_cae(segm, pad_time=pad_sgram)
            else:
                S = bark_specgram_cae(segm, pad_time=pad_sgram)
            z = self.encoder(S.unsqueeze(0))
            X_.append(z.detach().squeeze().numpy().ravel())
        return np.array(X_)

class CVAEFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, model, **kwargs):
        self.model = model
        self.model.eval()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = []
        for segm in X:
            segm = segm.cut_or_pad(4096*2)
            S = mel_specgram_cae(segm, win_size=1024, hop_size=128, n_mels=64, pad_time=64)
            z = self.model.representation(S.unsqueeze(0))
            X_.append(z.detach().squeeze().numpy())
        return np.array(X_)

    
class RamiresClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, n_neighbors=3, verbose=0, cv=LeaveOneOut()):
        self.audio_features = ClassicFeatureTransform('ramires')
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.sfs = SFS(self.knn, k_features='best', cv=cv, verbose=verbose)
        
    def get_params(self, deep=True):
        return {
            'knn': self.knn.get_params(deep),
            'sfs': self.sfs.get_params(deep)
        }
    
    @abc.abstractproperty
    def classes_(self):
        return self.knn.classes_
    
    def fit(self, X, y):
        X_t = self.audio_features.transform(X)
        self.sfs.fit(X_t, y)
        X_best = X_t[:, self.sfs.k_feature_idx_]
        self.knn.fit(X_best, y)
    
    def transform(self, X):
        X_t = self.audio_features.transform(X)
        return X_t[:, self.sfs.k_feature_idx_]
        
    def predict(self, X):
        return self.knn.predict(self.transform(X))
    
    def predict_proba(self, X):
        return self.knn.predict_proba(self.transform(X))
    
def prepend_normalizer(model):
    return make_pipeline(Normalizer(), model)
    
def make_pipeline_util(classifier, transformer, normalize=True):
    estimators = [('features', transformer)]
    if normalize:
        estimators.append(('normalizer', Normalizer()))
    estimators.append(('classifier', classifier))
    return Pipeline(estimators)

def make_knn_classic(feature_type, normalize=True, **kwargs):
    return make_pipeline_util(KNeighborsClassifier(**kwargs),
                              ClassicFeatureTransform(feature_type),
                              normalize=normalize)

def make_knn_cae(encoder, normalize=True, sgram_type='mel', frame_len=16348, **kwargs):
    return make_pipeline_util(KNeighborsClassifier(**kwargs),
                              CAEFeatureTransform(encoder, sgram_type=sgram_type, frame_len=frame_len),
                              normalize=normalize)

def make_knn_vae(encoder, normalize=True, **kwargs):
    return make_pipeline_util(KNeighborsClassifier(**kwargs),
                              CVAEFeatureTransform(encoder),
                              normalize=normalize)