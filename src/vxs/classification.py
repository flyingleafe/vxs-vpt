import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from vxs.features import *
from vxs.dataset import *
from vxs import constants

class ClassicFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, feature_type, frame_len=constants.DEFAULT_STFT_WINDOW):
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
        return X_

class CAEFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, encoder, frame_len=4096):
        self.encoder = encoder
        self.frame_len = frame_len
        self.encoder.eval()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = []
        for segm in X:
            if self.frame_len is not None:
                segm = segm.cut_or_pad(self.frame_len)
            S = mel_specgram_cae(segm)
            z = self.encoder(S.unsqueeze(0))
            X_.append(z.detach().squeeze().numpy().ravel())
        return X_

def make_knn_classic(feature_type, normalize=True, **kwargs):
    feature_transformer = ClassicFeatureTransform(feature_type)
    estimators = [('features', feature_transformer)]
    if normalize:
        estimators.append(('normalizer', Normalizer()))
    estimators.append(('knn', KNeighborsClassifier(**kwargs)))
    return Pipeline(estimators)

def make_knn_cae(encoder, normalize=True, **kwargs):
    feature_transformer = CAEFeatureTransform(encoder)
    estimators = [('cae_features', feature_transformer)]
    if normalize:
        estimators.append(('normalizer', Normalizer()))
    estimators.append(('knn', KNeighborsClassifier(**kwargs)))
    return Pipeline(estimators)
