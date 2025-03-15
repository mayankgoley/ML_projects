# Car Lane Detection

Detecting lane lines on road images using classical computer vision. No deep learning, just OpenCV.

## What I did

Used the CARLA simulator dataset which has dashcam images and lane masks. Built the standard classical CV pipeline: grayscale, gaussian blur, white brightness mask (the lane markings are faint in this dataset), Canny edges, region of interest mask, then Hough lines to detect line segments. Split the segments by slope into left and right lanes, averaged each side, and extrapolated to a single line per side.

Pipeline steps:
- Grayscale and blur
- White brightness mask to handle faint markings
- Canny edge detection
- Trapezoidal region of interest
- Hough line transform
- Left/right slope split, averaging, extrapolation

Metrics: detection rate (a line found on both sides) and mean IoU against the dilated ground truth lane masks.

## Results

| Metric | Value |
|---|---|
| Detection rate | 14.0% |
| Mean IoU | 0.298 |

These numbers look low at first but they are honest. One of the ego lane boundaries in this dataset is on unpainted asphalt with no visible edges, so a classical pipeline cannot recover it. The pipeline catches the painted dashed centerline and road edges well, but it cannot detect a lane line that has nothing to detect. A learned segmentation model would solve this.

## Files

- `lane_detection.ipynb` - the whole project

## Dataset

Lane Detection for Carla Driving Simulator from Kaggle: https://www.kaggle.com/datasets/thomasfermi/lane-detection-for-carla-driving-simulator

Not pushed to the repo. Download it from the link and drop it in the same folder before running the notebook.

---

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
