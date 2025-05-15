from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DropNA(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.columns_to_keep = None

    def fit(self, X, y=None):
        self.columns_to_keep = X.columns[X.isnull().mean(axis=0) < self.threshold]
        return self

    def transform(self, X):

        columns_common = [col for col in self.columns_to_keep if col in X.columns]
        if len(columns_common) == 0:
            raise ValueError("No columns to keep after dropping NA.")
        
        X_common = X.loc[:, columns_common]
        X_common = X_common.reindex(columns=self.columns_to_keep) # fill with NA

        return X_common
    

def make_preprocessor():
    
    preprocessor = Pipeline([
        ('drop_na', DropNA(threshold=0.3)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
    ])

    return preprocessor

