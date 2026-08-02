import pathlib
import sys
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_raw_data, clean_data_for_eda
from src.sentiment import ensure_nltk_corpora

# Page Configuration
st.set_page_config(
    page_title="Flavorlytics — Restaurant Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .feature-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Ensure NLTK resources
@st.cache_resource
def init_app_resources():
    ensure_nltk_corpora()
    return True

init_app_resources()

# Load Dataset
@st.cache_data
def get_cached_eda_data():
    data_path = PROJECT_ROOT / "data" / "zomato.csv"
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_data_for_eda(raw_df)
    return raw_df, cleaned_df

try:
    raw_df, eda_df = get_cached_eda_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    data_error = str(e)

# Sidebar Navigation / Info
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/000000/fork-and-knife-with-plate-emoji.png", width=70)
    st.title("Flavorlytics")
    st.caption("AI-Powered Restaurant Analytics")
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    st.markdown("- **Home**: Overview & Dataset Stats")
    st.markdown("- **📊 EDA**: Visual Analytics & Trends")
    st.markdown("- **🍽️ Recommendations**: Best Value Engine")
    st.markdown("- **🤖 Success Prediction**: ML Model")
    st.markdown("- **💬 Sentiment Analysis**: Customer Complaints")
    st.markdown("---")
    st.info("💡 **Tip**: Select a module from the sidebar to begin exploring.")

# Main Header
st.markdown('<div class="main-title">🍽️ Flavorlytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End AI-Powered Restaurant Analytics, Recommendation & Success Prediction System</div>', unsafe_allow_html=True)

if not data_loaded:
    st.error(f"⚠️ Error loading dataset: {data_error}")
    st.warning("Please verify that `data/zomato.csv` exists or run `py -3 train_models.py`.")
    st.stop()

# Key Dataset Overview Metrics
st.markdown("### 📊 Dataset Summary Statistics")
col1, col2, col3, col4, col5 = st.columns(5)

total_raw = len(raw_df)
total_analyzed = len(eda_df)
avg_rating = round(eda_df['rate'].mean(), 2)
online_pct = round((eda_df['online_order'] == 'Yes').mean() * 100, 1)
booking_pct = round((eda_df['book_table'] == 'Yes').mean() * 100, 1)

col1.metric("Total Raw Entries", f"{total_raw:,}")
col2.metric("Analyzed Outlets", f"{total_analyzed:,}")
col3.metric("Average Rating", f"{avg_rating} / 5.0")
col4.metric("Online Order %", f"{online_pct}%")
col5.metric("Table Booking %", f"{booking_pct}%")

st.markdown("---")

# Feature Highlights Grid
st.markdown("### 🚀 Core Platform Modules")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    #### 📊 1. Exploratory Data Analysis (EDA)
    Explore interactive visual dashboards covering:
    - Rating distributions & cost vs. rating correlations
    - Popular cuisines, restaurant types & chain analysis
    - Online ordering vs. table booking impacts
    - City-wise restaurant density & high-rated ratios
    """)

    st.markdown("""
    #### 🍽️ 2. Restaurant Recommendation Engine
    Rule-based recommendation system tailoring choices by:
    - Target city & restaurant category
    - Maximum budget & rating preferences
    - Customizable weighting (Cost, Rating, Booking, Online Order)
    """)

with c2:
    st.markdown("""
    #### 🤖 3. Restaurant Success Prediction (ML)
    Predict whether a proposed restaurant will succeed ($\ge 3.8$ rating) using our trained **Random Forest Regressor** model based on:
    - Location, restaurant type & listed category
    - Online order & table booking availability
    - Estimated cost for two people
    """)

    st.markdown("""
    #### 💬 4. Sentiment Analysis & Customer Complaints
    NLP pipeline examining customer reviews of low-rated outlets ($\le 2.5$):
    - TextBlob sentiment polarity distributions
    - Top complaint keywords & frequency breakdown
    - Customer dissatisfaction wordcloud visualization
    """)

st.markdown("---")
st.markdown("### 📂 Dataset Source & Tech Stack")
st.caption("Powered by **Zomato Bangalore Dataset**, Scikit-learn, XGBoost, Plotly, TextBlob, NLTK, and Streamlit.")
