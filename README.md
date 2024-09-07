# House Price Prediction

Predicting house prices from a tabular dataset. Just a small ML project I built to practice the basics of regression.

## What I did

Loaded the dataset, looked at the columns, cleaned up missing values, made a couple of new features (total square footage, house age, total bathrooms), encoded the categoricals, and split the data 80/20. Then trained three models and compared them.

Models:
- Linear Regression
- Random Forest
- LightGBM

Metrics: RMSE and R² on the test set.

## Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 1,329,392 | 0.650 |
| Random Forest | 1,423,428 | 0.599 |
| LightGBM | 1,364,320 | 0.632 |

Linear Regression actually did best here, with LightGBM close behind and Random Forest trailing. On a small dataset like this one a linear model holding its own against the tree ensembles is not unusual.

## Files

- `house_price_prediction.ipynb` - the whole project

## Dataset

Housing.csv from Kaggle: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

Not pushed to the repo. Download it from the link and drop it in the same folder before running the notebook.
