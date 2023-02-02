# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a Random Forest classifier with default hyperparameters using the Sklearn library.

## Intended Use

Predict whether income exceeds $50K/yr based on census data.

## Training Data

The training [dataset](https://archive.ics.uci.edu/ml/datasets/census+income) cointains information from the 1994 Census database.

## Evaluation Data

After preprocessing the data, the data set has 30162 rows and 15 attributes. A 80-20 split was used to break this into a train and test set.

## Metrics

The model was evaluated on the following metrics: Precision: 0.73. Recall: 0.65. Fbeta: 0.69.

## Ethical Considerations

Given that the data contains attributes about sex, race and so on, special consideration should be given to how the model
performs accros different groups.

## Caveats and Recommendations

To further improve the performance, hyperparameter optimization can be considered. Another helpful recommedation might be doing feature engineering. 
