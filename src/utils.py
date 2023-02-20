"""
This module includes utility functions for training the model
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def load_data(path):
    "Import data from specified path and return a dataframe"

    df =  pd.read_csv(path)
    return df


def load_artifact(artifact_path):
    "Load artifact"

    return joblib.load(artifact_path)


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline. This is basically done to encode the categorical features. Strings to mathematical values which can be used in the model equations


    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def get_cat_features():
    """ This function Returns a list of categorical features, we will encode them before feeding to the model"""
    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    return cat_features
