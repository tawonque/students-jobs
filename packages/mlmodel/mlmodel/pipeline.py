from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from mlmodel.processing import preprocessors as pp
from mlmodel.processing import estimators
from mlmodel.config import config

autoencoder_pipeline = Pipeline(
    [
        (
            "remove_features",
            pp.RemoveFeatures(variables=config.FEATURES_TO_REMOVE)
        ),
        (
            "numerical_inputer",
            pp.TreatContinuous(variables=config.NUMERICAL_VARS_WITH_NA, max_variables=config.NUMERICAL_WITH_MAX_AS_DEFAULT),
        ),
        (
            "categorical_imputer",
            pp.TreatCategorical(variables=config.CATEGORICAL_VARS_WITH_NA, continous_as_categorical_variables=config.CONTINUOUS_AS_CATEGORICAL), # fix to consider the recreation of categircal variables in test sample
        ),
        (
            'scaler',
            MinMaxScaler((-1,1))
        ),
        (
            estimators.make_estimator(mode='autoencoder') # to review --- to write as a class and instantiate!!! Keep for now as a placeholder
        )

    ]
)

encoder_pipeline = Pipeline(
    [
        (
            "remove_features",
            pp.RemoveFeatures(variables=config.FEATURES_TO_REMOVE)
        ),
        (
            "numerical_inputer",
            pp.TreatContinuous(variables=config.NUMERICAL_VARS_WITH_NA, max_variables=config.NUMERICAL_WITH_MAX_AS_DEFAULT),
        ),
        (
            "categorical_imputer",
            pp.TreatCategorical(variables=config.CATEGORICAL_VARS_WITH_NA, continous_as_categorical_variables=config.CONTINUOUS_AS_CATEGORICAL), # fix to consider the recreation of categorical variables in test sample
        ),
        (
            'scaler',
            MinMaxScaler((-1,1))
        ),
        (
            estimators.make_estimator(mode='encoder') # to review --- to write as a class and instantiate!!! Keep for now as a placeholder
        )

    ]
)