import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from ml_model.config import config
from ml_model import __version__ as _version


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data

