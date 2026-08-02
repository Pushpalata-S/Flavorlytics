import pathlib
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.sentiment import load_sentiment_artifact

st.set_page_config(page_title="Sentiment Analysis — Flavorlytics", page_icon="💬", layout="wide")

st.title("💬 Customer Sentiment Analysis & Complaint Mining")
st.caption("NLP Natural Language Processing on customer reviews of low-rated restaurants (Rating ≤ 2.5)")

@st.cache_data
def get_cached_sentiment():
    models_dir = PROJECT_ROOT / "models"
    data = load_sentiment_artifact(models_dir)
    return data

try:
    sentiment_data = get_cached_sentiment()
    stats = sentiment_data["summary_stats"]
    top_words = sentiment_data["top_words"]
    word_freq_dict = sentiment_data["word_freq_dict"]
    sample_reviews = pd.DataFrame(sentiment_data["reviews_sample"])
    polarity_series = sentiment_data["polarity_series"]
    loaded = True
except Exception as e:
    loaded = False
    error_msg = str(e)

if not loaded:
    st.error(f"⚠️ Failed to load sentiment analysis artifacts: {error_msg}")
    st.warning("Please run `py -3 train_models.py` in your terminal to process and persist sentiment data.")
    st.stop()

# ---------------------------------------------------------
# METRIC SUMMARY CARDS
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

total_rev = stats["total_reviews"]
avg_pol = stats["avg_polarity"]
pos_cnt = stats["positive_count"]
neg_cnt = stats["negative_count"]
neu_cnt = stats["neutral_count"]

pos_pct = round((pos_cnt / total_rev) * 100, 1) if total_rev > 0 else 0
neg_pct = round((neg_cnt / total_rev) * 100, 1) if total_rev > 0 else 0

col1.metric("Low-Rated Reviews Analyzed", f"{total_rev:,}")
col2.metric("Average Polarity", f"{avg_pol:.3f}")
col3.metric("Negative Reviews %", f"{neg_pct}%", delta="- Negative Tone")
col4.metric("Positive Reviews %", f"{pos_pct}%", delta="+ Positive Tone")
col5.metric("Neutral Reviews Count", f"{neu_cnt:,}")

st.markdown("---")

# ---------------------------------------------------------
# VISUAL DASHBOARDS: POLARITY & WORDCLOUD
# ---------------------------------------------------------
st.markdown("### 📊 Review Sentiment Polarity Distribution")

fig_pol = px.histogram(
    x=polarity_series,
    nbins=40,
    title="Distribution of TextBlob Sentiment Polarity Scores (-1.0 Negative to +1.0 Positive)",
    color_discrete_sequence=['#FF4B4B'],
    labels={'x': 'Polarity Score', 'y': 'Review Count'}
)
fig_pol.add_vline(x=0.0, line_dash="dash", line_color="black", annotation_text="Neutral Line")
st.plotly_chart(fig_pol, use_container_width=True)

st.markdown("---")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("### ☁️ Customer Complaint WordCloud")
    st.caption("Visual representation of most frequent terms in negative reviews")
    
    if word_freq_dict:
        wc = WordCloud(
            width=800,
            height=500,
            background_color='#1E1E1E' if st.get_option("theme.base") == "dark" else '#FFFFFF',
            colormap='Reds',
            max_words=100
        ).generate_from_frequencies(word_freq_dict)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig)
    else:
        st.info("No complaint word frequency data found.")

with row1_col2:
    st.markdown("### 🚨 Top 20 Negative Complaint Keywords")
    st.caption("Most recurring terms extracted from low-rated restaurant reviews")
    
    top_20_df = pd.DataFrame(top_words[:20], columns=['Keyword', 'Frequency'])
    fig_bar = px.bar(
        top_20_df.sort_values('Frequency', ascending=True),
        x='Frequency',
        y='Keyword',
        orientation='h',
        color='Frequency',
        color_continuous_scale='reds',
        title="Top 20 Complaint Word Frequencies"
    )
    fig_bar.update_layout(height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# INTERACTIVE REVIEW INSPECTOR TABLE
# ---------------------------------------------------------
st.markdown("### 🔍 Sample Customer Review Inspector")

search_term = st.text_input("Filter sample reviews by keyword (e.g. 'delivery', 'cold', 'worst', 'taste'):", "")

if not sample_reviews.empty:
    filtered_reviews = sample_reviews.copy()
    if search_term.strip():
        filtered_reviews = filtered_reviews[
            filtered_reviews['clean_reviews'].str.contains(search_term.strip().lower(), case=False, na=False)
        ]
        
    st.dataframe(
        filtered_reviews.rename(columns={
            'name': 'Restaurant Name',
            'rate': 'Rating',
            'clean_reviews': 'Cleaned Review Content',
            'sentiment': 'Polarity Score'
        }),
        use_container_width=True,
        column_config={
            "Rating": st.column_config.NumberColumn(format="⭐ %.1f"),
            "Polarity Score": st.column_config.NumberColumn(format="%.3f")
        },
        hide_index=True
    )
