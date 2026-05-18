# ML Projects

A bunch of ML and deep learning projects I built while studying and applying for jobs. Each one is a single Jupyter notebook that I tried to keep readable and honest about what worked and what did not.

## Projects in this repo

1. [Energy Forecasting](#energy-forecasting)
2. [Speech Emotion Recognition](#speech-emotion-recognition)
3. [Face Mask Detection](#face-mask-detection)
4. [Traffic Sign Classification](#traffic-sign-classification)
5. [Car Lane Detection](#car-lane-detection)
6. [House Price Prediction](#house-price-prediction)
7. [IMDB Sentiment Analysis](#imdb-sentiment-analysis)

---

# Energy Forecasting

Hourly energy consumption forecasting on the PJME region of the PJM grid. Wanted to do a time series project properly, with chronological splits and a real naive baseline that the deep models actually have to beat.

## What I did

Used about 16 years of hourly PJME energy data (2002 to 2018, around 145k rows). Cleaned up the DST quirks with linear interpolation for the missing hours and averaged the duplicated hours. Did a chronological 80/10/10 split (no shuffling, this is the whole point with time series). Fit the scaler on the train portion only to avoid leaking future info into training.

For features, added hour of day, day of week, month, is weekend, and cyclic sin/cos encodings of hour and day. Used 1 week of past hours (168) as input to predict the next 24 hours.

Models:
- Naive baseline (predict same 24 hours from one week ago)
- LSTM with one or two layers and a linear head outputting 24 values
- Transformer encoder with sinusoidal positional encoding, 2 to 3 encoder layers

Metrics: MAE and RMSE on the inverse scaled values (so the numbers are in actual megawatts, not scaled units).

## Results

Test period: Dec 2016 to Aug 2018 (last 10% of the series).

| Model | MAE (MW) | RMSE (MW) |
|---|---|---|
| Naive | 3519 | 4788 |
| LSTM | 1503 | 2084 |
| Transformer | 1466 | 2050 |

The deep models cut MAE by about 58 percent over naive (3519 down to about 1480 MW). Against a mean load of around 32000 MW that is roughly 4.6 percent error for the deep models versus 11 percent for naive. The naive baseline cannot react to weather, and grid load is heavily weather driven, which is why the gap is this big.

Transformer barely beat LSTM (37 MW MAE difference, about 2.5 percent) which is within run to run noise. Honest take: the extra training time for the Transformer did not buy much over the LSTM on this dataset.

I verified all the time series rules before reporting numbers: chronological split with printed dates, scaler fit on train only, and an explicit look ahead sanity check that prints the actual timestamps of one X window and its corresponding y window with an assertion that y starts after X ends.

## Files

- `energy_forecasting.ipynb` - the whole project

## Dataset

Hourly Energy Consumption from Kaggle: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

Not pushed to the repo. Download it from the link and drop the PJME_hourly.csv in the same folder before running the notebook.

---

# Speech Emotion Recognition

Classifying short speech clips into 8 emotions. Wanted to see how far a small CNN on spectrograms can get versus a classical baseline.

## What I did

Used the RAVDESS speech dataset (1440 clips, 24 actors, 8 emotions). Found a duplicate folder inside the dataset and ignored it. Trimmed leading and trailing silence with librosa, fixed every clip to 3 seconds, and split by actor (18 train, 3 val, 3 test) so the model could not just memorize voices. Stratified the actor split by gender so both are represented in each split.

Models:
- Random Forest on MFCC summary stats (mean and std across time) as a baseline
- Small CNN on log mel spectrograms

Used inverse frequency class weights in the CNN loss since the neutral class is half size (RAVDESS does not have a strong intensity variant for neutral).

Metrics: test accuracy, classification report, confusion matrix.

## Results

| Model | Test Accuracy |
|---|---|
| Random Forest (MFCC stats) | 0.44 |
| CNN (log mel spectrograms) | 0.46 |

Both land around 45 percent for 8 classes (chance is 12.5 percent). Looks low at first glance but it is the honest number. With only 18 training actors and a speaker independent split, the model cannot lean on familiar voices, which is exactly the point of splitting that way.

The most confused pairs from the CNN were happy predicted as angry (both high arousal), and calm and neutral predicted as sad (all three are low energy and flat in pitch). These are the standard confusion patterns in the speech emotion literature, even humans confuse these in unfamiliar languages.

The CNN barely beat the Random Forest by 2 points which is itself an interesting finding. With small datasets like this one, classical features with good ML often nearly match deep learning.

## Files

- `speech_emotion_recognition.ipynb` - the whole project

## Dataset

RAVDESS Emotional Speech Audio from Kaggle: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Not pushed to the repo. Download it from the link and drop it in the same folder before running the notebook.

---

# Face Mask Detection

Classifying face images as with mask or without mask. Two models compared: a small CNN trained from scratch and a frozen MobileNetV2 with a new head.

## What I did

Used a Kaggle face mask dataset with about 7500 face crops split across two folders (with_mask and without_mask). The classes were essentially balanced so no class weights needed. Did a stratified 70/15/15 split for train, val, and test. Resized to 128x128 for the scratch CNN and 224x224 for MobileNetV2 (since it is pretrained on ImageNet at that resolution). Added augmentation on the train set (flip, rotation, color jitter, random resized crop).

Models:
- CNN from scratch (4 conv blocks then an FC head)
- MobileNetV2 with the backbone frozen, new 2 way head trained

Metrics: test accuracy, classification report, confusion matrix.

## Results

| Model | Test Accuracy |
|---|---|
| CNN from scratch | 0.957 |
| MobileNetV2 (transfer) | 0.982 |

MobileNetV2 transfer learning won, 0.982 vs 0.957. Unlike the traffic sign case, transfer learning helps here: the inputs are 224x224 face crops that look like the natural photos ImageNet was trained on, so the frozen backbone features transfer well. The from-scratch CNN is still strong on its own — this is an easy, balanced binary task — but the pretrained features give MobileNetV2 the edge.

Since the images are already tight face crops, the demo cell just classifies the whole image and draws a colored label (green for with_mask, red for without_mask). No separate face detection step needed.

## Files

- `face_mask_detection.ipynb` - the whole project

## Dataset

Face Mask Dataset from Kaggle: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

Not pushed to the repo. Download it from the link and drop it in the same folder before running the notebook.

---

# Traffic Sign Classification

Classifying German traffic signs into 43 categories. Two models compared: a small CNN trained from scratch and a frozen ResNet18 with a new head.

## What I did

Used the GTSRB dataset, loaded train and test from the CSVs (since the test folder is flat and the labels live in Test.csv), built an 80/20 train/val split. Resized images to 64x64, normalized, and added augmentation on the train set (rotation, color jitter, perspective). Used inverse frequency class weights in the loss because GTSRB is moderately imbalanced (about a 10x spread between smallest and largest class).

Models:
- CNN from scratch (4 conv blocks then a small FC head)
- ResNet18 with the backbone frozen, new 43 way head trained

Metrics: test accuracy and a classification report on the most confused classes.

## Results

| Model | Test Accuracy |
|---|---|
| CNN from scratch | 0.969 |
| ResNet18 (transfer) | 0.667 |

The from-scratch CNN won clearly, 0.969 vs 0.667. Freezing an ImageNet-pretrained ResNet18 backbone hurts here: traffic signs are small, low-detail images unlike the natural photos ResNet learned on, so the frozen features do not transfer well. A CNN learning features directly on GTSRB beats it easily. Unfreezing and fine tuning the backbone would be the fix.

The weakest classes were the speed limit signs (20, 30, 50, 70, 80 km/h all look similar), which is the expected failure pattern.

## Files

- `traffic_sign_classification.ipynb` - the whole project

## Dataset

GTSRB German Traffic Sign Recognition Benchmark from Kaggle: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Not pushed to the repo. Download it from the link and drop it in the same folder before running the notebook.

---

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

- `car_lane_detection.ipynb` - the whole project

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
