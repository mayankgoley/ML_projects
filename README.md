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

---

# IMDB Sentiment Analysis

Classifying movie reviews as positive or negative. Wanted to compare a classical NLP baseline against an LSTM and a fine tuned transformer to see how big the jump actually is.

## What I did

Loaded the 50k IMDB reviews, dropped duplicates, cleaned out the HTML tags, mapped sentiment to 0 and 1, and did a stratified 80/20 split. Then trained three models with the same train/test split so the comparison is fair.

Models:
- TF-IDF + Logistic Regression (baseline)
- LSTM from scratch in PyTorch
- DistilBERT fine tuned for 1 epoch on the full dataset

Metrics: accuracy, classification report, confusion matrix on the test set.

## Results

| Model | Test Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 0.905 |
| LSTM | 0.862 |
| DistilBERT | 0.913 |

DistilBERT came out on top, but only by about a point over the TF-IDF baseline — its pretraining is what lets it handle the sarcastic and mixed reviews the other two miss. The from-scratch LSTM trailed both: with no pretrained embeddings it has to learn word meaning from 40k reviews alone, while TF-IDF with bigrams is already a very strong baseline on IMDB.

## Files

- `imdb_sentiment_classification.ipynb` - the whole project

## Dataset

IMDB Dataset of 50K Movie Reviews from Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Not pushed to the repo. Download it from the link and drop it in the same folder before running the notebook.
