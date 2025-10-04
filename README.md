# ML Projects

A bunch of ML and deep learning projects I built while studying and applying for jobs. Each one is a single Jupyter notebook that I tried to keep readable and honest about what worked and what did not.

## Projects in this repo

1. [Face Mask Detection](#face-mask-detection)
2. [Traffic Sign Classification](#traffic-sign-classification)
3. [Car Lane Detection](#car-lane-detection)
4. [House Price Prediction](#house-price-prediction)
5. [IMDB Sentiment Analysis](#imdb-sentiment-analysis)

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
