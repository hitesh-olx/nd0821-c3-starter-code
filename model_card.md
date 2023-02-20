# Model Card


## Model Details

The model is XGBOOST classifier

## What this predicts

Based on Census data like Age, Occupation, education this predicts whether income exceeds $50K/yr based on census data.

## Training Data

We have used the training data (https://archive.ics.uci.edu/ml/datasets/census+income)

## Validation Data

We have used 20% data for validation

## Accuracy Metrics

The model was evaluated on: Precision,Recall and Fbeta.


## Future improvements

To further improve the performance, we can  optimize hyperparameter. Currenty my model seems to be overfit which we can optimize with more data. For now I have only considered a small set of data.
