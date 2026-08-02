# 🍽️ Flavorlytics — AI-Powered Restaurant Analytics & Recommendation System

Flavorlytics is a complete multi-page **Streamlit** web application for exploring restaurant trends, generating personalized best-value recommendations, predicting restaurant success with Machine Learning, and performing customer sentiment analysis on reviews.

---

## 📂 Project Architecture

```
Flavourytics/
├── data/
│   └── zomato.csv                 # Raw dataset (downloaded once)
├── models/                        # Persisted model artifacts (joblib)
│   ├── imputer_model.joblib       # Rating imputer model
│   ├── preprocessor.joblib        # Fitted ColumnTransformer (OHE)
│   ├── success_model.joblib       # RandomForestRegressor for rating prediction
│   └── sentiment_data.joblib      # Precomputed sentiment analysis data
├── src/                           # Core Python logic package
│   ├── __init__.py
│   ├── data_processing.py         # Data cleaning & recommendation logic
│   ├── models.py                  # ML model training & Predict_success inference
│   └── sentiment.py               # Review cleaning, TextBlob sentiment & NLTK keyword extraction
├── pages/
│   ├── 1_📊_EDA.py                # Recreated Plotly EDA charts
│   ├── 2_🍽️_Recommendations.py    # Interactive recommendation engine
│   ├── 3_🤖_Success_Prediction.py # ML success prediction interface
│   └── 4_💬_Sentiment_Analysis.py # Sentiment analysis & complaint wordcloud
├── Home.py                        # Streamlit main entrypoint & overview
├── train_models.py                # One-time model training & precomputation script
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## ⚡ Quick Start Guide

### 1. Install Dependencies

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Place the Zomato Bangalore Restaurants dataset at `data/zomato.csv`.

If missing, download it using Python and `kagglehub`:

```python
import kagglehub
import os, shutil

path = kagglehub.dataset_download('himanshupoddar/zomato-bangalore-restaurants')
os.makedirs('data', exist_ok=True)
for item in os.listdir(path):
    if item.endswith('.csv'):
        shutil.copy(os.path.join(path, item), 'data/zomato.csv')
```

### 3. Run One-Time Training Script

Train the rating imputer and Random Forest Success Prediction pipeline, and precompute sentiment metrics:

```bash
python train_models.py
```

> **Note**: Re-run `train_models.py` whenever the underlying dataset is updated.

### 4. Launch Streamlit Web Application

```bash
streamlit run Home.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (ensure `data/zomato.csv` or model artifacts are tracked or downloaded).
2. Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Set main file path to `Home.py`.
4. Deploy! The app automatically manages NLTK resource downloads and loads cached joblib models.
5. Local URL: http://localhost:8501


---

## 🛠️ Features & Modules

1. **📊 Exploratory Data Analysis (EDA)**: Interactive visual dashboards covering ratings, cost, cuisines, online ordering/booking trends, and city breakdown.
2. **🍽️ Restaurant Recommendations**: Rule-based best-value scoring engine with custom budget, rating, and feature importance sliders.
3. **🤖 Success Prediction**: ML classification predicting whether a restaurant will achieve a success rating ($\ge 3.8$).
4. **💬 Sentiment Analysis**: Customer review sentiment polarity and complaint wordcloud for low-rated outlets ($\le 2.5$).
