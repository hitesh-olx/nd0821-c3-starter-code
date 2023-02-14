"""
This module defines the functions for model, metrics and inference
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
    Training data [Independent data]
    y_train : np.array
    Actual label against training data.
        
    Returns
    -------
    model
    Machine learning model [Basically the equation which is fit by the ML model using the variables of training set]
    """

    # fit model on training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the ML model metrics 

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : [% of Model Positives being actually positives]
    recall : [% of TP covered by model]

    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : pkl
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    return model.predict(X)


