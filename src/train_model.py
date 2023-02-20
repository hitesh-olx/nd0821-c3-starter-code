""" 
Script to train the machine learning model.
"""

import logging
from sklearn.model_selection import train_test_split
from utils import load_data, process_data, get_cat_features
from model import train_model, compute_model_metrics, inference
import joblib


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Import data
logging.info("Importing data")
DATA_PATH = 'data/clean_census.csv'
data = load_data(DATA_PATH)

# Train test split: 80% data is used for training prpose and we validate on 20% test data
train, test = train_test_split(data, test_size=0.20)

# Preprocess the data [Make the data in the format to hit the model. Encode the Categorical Variables]
logging.info("Preprocessing data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=get_cat_features(), label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=get_cat_features(), label="salary",
     training=False, encoder=encoder, lb=lb)


# Train the XGBOOST model: We have used XGB as it performed good as compared to RandomForest
logging.info("Training model")
model = train_model(X_train, y_train)

# Scoring: Getting the Scores/Category on the Test data Set
logging.info("Scoring on test set")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save artifacts: Saving the artifacts: Model, Encoders and 
logging.info("Saving artifacts")
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')
