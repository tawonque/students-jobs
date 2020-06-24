import pathlib
import model-a
import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "to be defined" #TODO 


# variables
FEATURES = [ #TODO
]

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = "to be defined" #TODO
 
# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ["to be defined"] #TODO

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [ #TODO
]

TEMPORAL_VARS = "to be def" #TODO

# variables to log transform
NUMERICALS_LOG_VARS = ["to be def"] #TODO

# categorical variables to encode
CATEGORICAL_VARS = [ #TODO

]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]


PIPELINE_NAME = "to be def" #lasso_regression
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
#ACCEPTABLE_MODEL_DIFFERENCE = 0.05 TODO
