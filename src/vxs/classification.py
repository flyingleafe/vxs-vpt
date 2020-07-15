import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .features import *
from .dataset import *

class ClassicFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_type):
        if feature_type not in ('mfcc', 'ramires'):
            raise ValueError('Unknown feature type:' + feature_type)
        self.feature_type = feature_type
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = []
        for segm in X:
            if self.feature_type == 'mfcc':
                features = mfcc(segm, center=False).ravel()
            elif self.feature_type == 'ramires':
                features = ramires_features(segm).ravel()
            X_.append(features)
        return X_
    
def make_knn_classic(feature_type, normalize=True, **kwargs):
    feature_transformer = ClassicFeatureTransformer(feature_type)
    estimators = [('features', feature_transformer)]
    if normalize:
        estimators.append(('normalizer', Normalizer()))
    estimators.append(('knn', KNeighborsClassifier(**kwargs)))
    return Pipeline(estimators)