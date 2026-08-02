# 🍽️ Flavorlytics

## AI-Powered Restaurant Analytics & Recommendation System

Flavorlytics is an end-to-end data analytics project that explores restaurant trends using the Zomato Bangalore dataset. It combines exploratory data analysis, natural language processing, and machine learning to uncover customer preferences, evaluate restaurant performance, and deliver intelligent restaurant recommendations.

The project demonstrates how data-driven insights can support both customers in choosing restaurants and businesses in understanding factors that influence restaurant success.

---

## 📂 Dataset

This project uses the **Zomato Bangalore Restaurants Dataset** available on Kaggle.

The dataset contains information such as:

- Restaurant name and location
- Cuisine served
- Customer ratings and votes
- Average cost for two people
- Online ordering availability
- Table booking availability
- Customer reviews
- Restaurant type and listed category

---

## 🚀 Project Modules

### 📊 1. Exploratory Data Analysis (EDA)

Performed extensive exploratory analysis to understand restaurant performance and customer behavior.

Key analyses include:

- Rating distribution across restaurants
- Cost vs Rating relationship
- Popular cuisines and restaurant types
- Top restaurant locations
- Online ordering and table booking trends
- Customer voting patterns
- Premium restaurant analysis

---

### 🍴 2. Restaurant Recommendation System

Developed a recommendation engine that suggests restaurants based on user preferences.

Recommendation parameters include:

- City / Location
- Cuisine
- Budget
- Customer Rating

This helps users discover restaurants that best match their requirements.

---

### 💬 3. Sentiment Analysis

Applied Natural Language Processing (NLP) techniques on customer reviews to understand public opinion.

The analysis includes:

- Positive and negative sentiment classification
- Frequently used words in customer reviews
- Common complaints in low-rated restaurants
- Customer satisfaction trends

---

### 🤖 4. Restaurant Success Prediction

Built machine learning models to predict the likelihood of a restaurant performing successfully based on historical restaurant data.

Models explored include:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Model performance is evaluated using multiple classification metrics.

---

## 📈 Business Insights

The project answers questions such as:

- Which locations have the highest-rated restaurants?
- Which cuisines are most preferred by customers?
- Does online ordering influence ratings?
- Are expensive restaurants always highly rated?
- What factors contribute to restaurant success?
- Why do customers leave negative reviews?

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn
- XGBoost

### Natural Language Processing

- NLTK
- TextBlob
- WordCloud

### Development Environment

- Google Colab
- Jupyter Notebook
- VS Code

---

## 📁 Project Structure

```
Flavorlytics/
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_recommendation_system.ipynb
│   └── 05_success_prediction.ipynb
│
├── visuals/
│
├── reports/
│
├── README.md
└── requirements.txt
```

---

## 🎯 Project Highlights

- Cleaned and analyzed a restaurant dataset containing **50,000+ records**
- Built an interactive recommendation system using customer preferences
- Performed sentiment analysis on thousands of customer reviews
- Compared multiple machine learning models for restaurant success prediction
- Generated actionable business insights through visual analytics

---

## 🔮 Future Enhancements

- Deploy the recommendation engine using Streamlit
- Add interactive dashboards
- Integrate real-time restaurant data
- Improve recommendation accuracy using hybrid recommendation techniques
- Perform deep learning-based sentiment analysis

---

## 📌 Conclusion

Flavorlytics demonstrates how data analytics, machine learning, and natural language processing can be combined to solve real-world business problems in the restaurant industry. The project transforms raw restaurant data into meaningful insights that support better customer decisions and business strategies.
