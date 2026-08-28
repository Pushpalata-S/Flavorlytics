<div align="center">

# 🍽️ Flavorlytics

### AI-Powered Restaurant Analytics & Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat&logo=plotly&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-4B8BBE?style=flat)
![TextBlob](https://img.shields.io/badge/TextBlob-Sentiment-yellow?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

*A multi-page Streamlit web application that turns raw restaurant data into interactive analytics, ML-driven success predictions, personalized recommendations, and customer sentiment insights.*

**[🔗 View Repository](https://github.com/Pushpalata-S/Flavorlytics)**

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features & Modules](#️-features--modules)
- [Screenshots](#️-screenshots)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#️-tech-stack)
- [How It Works](#-how-it-works)
- [Quick Start Guide](#-quick-start-guide)
- [Deployment](#️-deployment-on-streamlit-community-cloud)
- [Dataset](#-dataset)
- [Key Concepts Demonstrated](#-key-concepts-demonstrated)
- [What I Learned](#-what-i-learned)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📌 Project Overview

**Flavorlytics** is an end-to-end data science application built around the Zomato Bangalore Restaurants dataset. Rather than being a single notebook or static dashboard, it's a full **Streamlit web app** with four interconnected modules — exploratory analytics, a recommendation engine, a machine learning success predictor, and NLP-based sentiment analysis — all served from one clean, multi-page interface.

The project demonstrates the complete data product lifecycle:

```text
Raw Data → Cleaning & Imputation → Feature Engineering → Model Training →
Precomputed Artifacts → Interactive Web App → Business Insights
```

> **Why it matters:** it shows the ability to take a messy, real-world dataset all the way from raw CSV to a deployed, interactive product — combining data cleaning, ML modeling, NLP, and front-end delivery in a single cohesive application.

---

## 🛠️ Features & Modules

### 📊 1. Exploratory Data Analysis (EDA)
Interactive Plotly dashboards covering rating distribution, cost patterns, cuisine popularity, online ordering/table booking trends, and city-wise breakdowns — letting users explore the restaurant landscape visually.

### 🍽️ 2. Restaurant Recommendations
A rule-based **best-value scoring engine**. Users set their own budget, minimum rating, and feature-importance sliders, and the engine ranks restaurants that best match their personal preferences.

### 🤖 3. Success Prediction (ML)
A **Random Forest Regressor**, trained on cleaned and encoded restaurant features, predicts whether a restaurant is likely to achieve a **success rating (≥ 3.8)**. Includes a missing-rating imputer model so incomplete records don't get dropped from training.

### 💬 4. Sentiment Analysis
Customer reviews are cleaned and scored for polarity using **TextBlob**, with **NLTK**-based keyword extraction surfacing the most common complaint terms — visualized as a word cloud specifically for low-rated outlets (**≤ 2.5**), making it easy to spot recurring pain points.

---

## 🖼️ Screenshots

<p align="center">
<img width="1182" alt="Flavorlytics home / overview" src="https://github.com/user-attachments/assets/1e4ca0b9-93af-4936-a049-a788d8881f51" />
<br/><br/>
<img width="1467" alt="Exploratory Data Analysis dashboard" src="https://github.com/user-attachments/assets/b2b57012-af55-4f4a-a4dc-7c41eedb8c15" />
<br/><br/>
<img width="1041" alt="Restaurant recommendation engine" src="https://github.com/user-attachments/assets/991e3908-e551-48ff-86a4-a4e55f0e053e" />
<br/><br/>
<img width="1653" alt="ML success prediction interface" src="https://github.com/user-attachments/assets/fa84a3b7-c9fc-495f-a446-75e63a266e53" />
<br/><br/>
<img width="1477" alt="Sentiment analysis and complaint wordcloud" src="https://github.com/user-attachments/assets/d4a7300d-7028-4ff6-bf7e-dc8aacb3396c" />
</p>

---

## 📂 Project Architecture

```text
Flavorlytics/
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
│   ├── models.py                  # ML model training & predict_success inference
│   └── sentiment.py               # Review cleaning, TextBlob sentiment & NLTK keyword extraction
├── pages/
│   ├── 1_📊_EDA.py                # Recreated Plotly EDA charts
│   ├── 2_🍽️_Recommendations.py    # Interactive recommendation engine
│   ├── 3_🤖_Success_Prediction.py # ML success prediction interface
│   └── 4_💬_Sentiment_Analysis.py # Sentiment analysis & complaint wordcloud
├── Home.py                        # Streamlit main entrypoint & overview
├── train_models.py                # One-time model training & precomputation script
├── requirements.txt                # Dependencies
└── README.md                      # Documentation
```

> 💡 As with any repo, keep this structure in sync with what's actually committed — rename or prune entries here if files move.

---

## ⚙️ Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Web App** | Streamlit | Multi-page interactive front end |
| **Data Handling** | Pandas, NumPy | Cleaning, transformation, feature engineering |
| **Visualization** | Plotly | Interactive EDA charts |
| **Machine Learning** | scikit-learn (RandomForestRegressor, ColumnTransformer) | Success prediction & preprocessing pipeline |
| **NLP / Sentiment** | TextBlob, NLTK | Review polarity scoring & keyword/complaint extraction |
| **Model Persistence** | joblib | Saving/loading trained models & precomputed artifacts |
| **Data Source** | kagglehub | Programmatic dataset download |
| **Deployment** | Streamlit Community Cloud | Hosting the live app |

---

## 🧠 How It Works

```text
┌───────────────────────────┐
│      RAW DATASET          │
│  Zomato Bangalore CSV     │
└─────────────┬──────────────┘
              ↓
┌───────────────────────────┐
│    DATA PROCESSING        │
│  Cleaning + Imputation    │
│  (src/data_processing.py) │
└─────────────┬──────────────┘
              ↓
      ┌───────┴────────┐
      ↓                ↓
┌───────────────┐ ┌─────────────────────┐
│  ML PIPELINE   │ │  SENTIMENT PIPELINE  │
│ ColumnTransform│ │ TextBlob + NLTK      │
│ RandomForest   │ │ Keyword Extraction   │
│ (src/models.py)│ │ (src/sentiment.py)   │
└───────┬────────┘ └──────────┬───────────┘
        ↓                     ↓
┌───────────────────────────────────────┐
│   PRECOMPUTED ARTIFACTS (joblib)      │
│   models/*.joblib                     │
└─────────────────┬──────────────────────┘
                   ↓
┌───────────────────────────────────────┐
│         STREAMLIT WEB APP             │
│  EDA · Recommendations · Prediction   │
│  · Sentiment Analysis                 │
└───────────────────────────────────────┘
```

Models and sentiment metrics are **trained/precomputed once** via `train_models.py` and persisted with `joblib`, so the live app loads cached artifacts instantly instead of retraining on every run.

---

## ⚡ Quick Start Guide

### 1. Install Dependencies

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

Place the Zomato Bangalore Restaurants dataset at `data/zomato.csv`.

If it's missing, download it programmatically:

```python
import kagglehub
import os, shutil

path = kagglehub.dataset_download('himanshupoddar/zomato-bangalore-restaurants')
os.makedirs('data', exist_ok=True)
for item in os.listdir(path):
    if item.endswith('.csv'):
        shutil.copy(os.path.join(path, item), 'data/zomato.csv')
```

### 3. Run the One-Time Training Script

Trains the rating imputer and Random Forest success-prediction pipeline, and precomputes sentiment metrics:

```bash
python train_models.py
```

> **Note:** Re-run `train_models.py` whenever the underlying dataset is updated.

### 4. Launch the Streamlit App

```bash
streamlit run Home.py
```

The app will be available locally at **http://localhost:8501**.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (ensure `data/zomato.csv` and/or model artifacts are tracked or downloadable at runtime).
2. Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Set the main file path to `Home.py`.
4. Deploy — the app automatically manages NLTK resource downloads and loads the cached `joblib` models.

---

## 📂 Dataset

This project uses the **[Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)** dataset from Kaggle, containing restaurant-level data on location, cuisine, cost, ratings, online ordering, table booking, and customer reviews across Bangalore.

---

## 📚 Key Concepts Demonstrated

<table>
<tr><td valign="top" width="33%">

**Data Engineering**
- Data cleaning & validation
- Missing-value imputation
- Feature engineering
- ColumnTransformer / One-Hot Encoding
- Reusable preprocessing pipelines

</td><td valign="top" width="33%">

**Machine Learning**
- RandomForestRegressor
- Train/precompute-once architecture
- Model persistence with joblib
- Rule-based recommendation scoring
- Threshold-based classification (≥ 3.8 success)

</td><td valign="top" width="34%">

**NLP & App Development**
- TextBlob sentiment polarity
- NLTK keyword/complaint extraction
- Word cloud visualization
- Multi-page Streamlit architecture
- Plotly interactive dashboards

</td></tr>
</table>

---

## 🎓 What I Learned

**Data Engineering** — building a clean, reusable preprocessing pipeline with imputation and `ColumnTransformer`-based encoding that could be shared between the recommendation engine and the ML model without duplication.

**Machine Learning** — training and persisting a `RandomForestRegressor` for a real prediction task, and structuring the project so training happens once (`train_models.py`) rather than on every app run.

**NLP** — applying TextBlob for sentiment polarity and NLTK for keyword extraction to turn unstructured review text into an actionable complaint word cloud.

**Product Thinking** — designing a multi-page Streamlit app that ties EDA, recommendations, prediction, and sentiment analysis together into one coherent user experience rather than four disconnected scripts.

---

## 🔮 Future Improvements

- Swap the rule-based recommender for a learned ranking/collaborative-filtering model
- Add model explainability (SHAP/feature importance) to the success predictor
- Replace TextBlob with a transformer-based sentiment model for higher accuracy
- Add automated retraining on a schedule as new review data arrives
- Add user authentication and saved preference profiles
- Containerize with Docker for more portable deployment

---

## 👩‍💻 Author

**Pushpalata S**
B.Tech in Chemical Engineering, Motilal Nehru National Institute of Technology Allahabad

[![GitHub](https://img.shields.io/badge/GitHub-Pushpalata--S-181717?style=flat&logo=github)](https://github.com/Pushpalata-S)

---

<div align="center">

⭐ **[View the full repository](https://github.com/Pushpalata-S/Flavorlytics)** — explore the Streamlit app, ML pipeline, and sentiment analysis modules.

</div>
