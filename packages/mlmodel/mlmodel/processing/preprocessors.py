# imports 
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveFeatures(BaseEstimator, TransformerMixin):
    """Remove features from the input dataset"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X = X.drop([feature], axis=1)
        return X


class TreatContinuous(BaseEstimator, TransformerMixin):
    """Impute values for continuous variables"""

    def __init__(self, variables=None, max_variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        if not isinstance(variables, list):
            self.max_variables = [max_variables]
        else:
            self.max_variables = max_variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        # for the max 
        for feature in self.max_variables:
            max_stints_default = X[feature].max() # default parameter could be made more sophisticated, but that's it for now
            X[feature] = X[feature].fillna(max_stints_default)
        
        # for the rest
        for feature in self.variables:
            X[feature] = X[feature].fillna(0) 
        
        return X


class TreatCategorical(BaseEstimator, TransformerMixin):
    """Impute values for categorical values and get dummy variables"""

    def __init__(self, variables=None, continous_as_categorical_variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.continous_as_categorical_variables = continous_as_categorical_variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for features in variables:
            X[col] = X[col].fillna("missing")

        total_features = variables + continous_as_categorical_variables
        X = pd.get_dummies(X, columns=columns)
        
        return X
    
