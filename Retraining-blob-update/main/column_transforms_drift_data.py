import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging
from sklearn.pipeline import Pipeline



class ContinuousTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        self.skewness_ = X[self.features].skew().to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        # logging.info(f"Handling Continuous features....Total {len(self.features)}")
        for feature in self.features:
            skew = self.skewness_[feature]
            if abs(skew) >= 0.5:
                # logging.info(f"Handling skew in {feature}")
                X[feature] = np.log1p(X[feature])
            # else:
                # logging.info(f"No significant skew in {feature}")
        return X

class DiscreteTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, max_bin=10, factor=5):
        self.features = features
        self.max_bin = max_bin
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # logging.info(f"Handling Discrete features....Total {len(self.features)}")
        for feature in self.features:
            unique_values = X[feature].nunique()
            if unique_values <= self.max_bin:
                logging.info(f"No binning needed for {feature}")
            else:
                bins = min(self.max_bin, max(unique_values // self.factor, 2))
                # logging.info(f"Binning {feature} into {bins} bins")
                X[feature] = pd.cut(X[feature], bins=bins, labels=False)
        return X

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.enc_dict_ = {}

    def fit(self, X, y=None):
        for feature in self.features:
            series_ = X[feature]
            if feature == "Gender" and series_.nunique() == 2:
                self.enc_dict_[feature] = {"Male": 1, "Female": 0}
            else:
                freq_map = series_.value_counts().to_dict()
                if len(set(freq_map.values())) == len(freq_map):
                    self.enc_dict_[feature] = freq_map
                else:
                    self.enc_dict_[feature] = {k: v for v, k in enumerate(series_.unique())}
        return self

    def transform(self, X):
        X = X.copy()
        # logging.info(f"Handling Categorical features....Total {len(self.features)}")
        for feature in self.features:
            # logging.info(f"Encoding categorical feature {feature}")
            X[feature] = X[feature].map(self.enc_dict_[feature])
        return X


def get_drift_handling_pipeline(feature_categories):
    continuous_features = feature_categories.get('continuous', [])
    discrete_features = feature_categories.get('discrete', [])
    categorical_features = feature_categories.get('categorical', [])

    pipeline = Pipeline(steps=[
        ('continuous', ContinuousTransformer(continuous_features)),
        ('discrete', DiscreteTransformer(discrete_features)),
        ('categorical', CategoricalTransformer(categorical_features)),
    ])
    logging.info("Preprocessing has been completed!!!***")
    return pipeline