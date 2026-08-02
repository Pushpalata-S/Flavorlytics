import pathlib
import ast
import re
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from collections import Counter

def ensure_nltk_corpora():
    """Ensure required NLTK resources are downloaded."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def clean_text(text) -> str:
    """Clean review text matching Notebook 1 logic, handling tuple entries from literal_eval."""
    if isinstance(text, (list, tuple)):
        text = text[1] if len(text) > 1 else (text[0] if len(text) == 1 else "")
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bratedn?\b', '', text, flags=re.IGNORECASE)
    return text

def extract_keywords(text: str, stop_words: set) -> list:
    """Extract keywords excluding stopwords and short words (<3 chars)."""
    words = text.lower().split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords

def safe_convert_reviews(x):
    """Safely convert string representation of list to actual Python list."""
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == '[]':
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

def process_sentiment_analysis(df: pd.DataFrame):
    """
    Process low-rated restaurant reviews (rate <= 2.5), explode reviews,
    compute TextBlob polarity, extract top complaint words and sentiment metrics.
    """
    ensure_nltk_corpora()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    # Add common non-informative restaurant review words to stop words
    additional_stops = {'restaurant', 'food', 'place', 'service', 'good', 'bad', 'ordered', 'order', 'one', 'get', 'got', 'taste', 'like', 'back'}
    stop_words.update(additional_stops)

    # Filter low-rated restaurants (rate <= 2.5)
    sentiment_df = df[df['rate'] <= 2.5][['name', 'rate', 'reviews_list']].copy()
    sentiment_df = sentiment_df.dropna(subset=['reviews_list'])
    
    # Safe convert review strings
    sentiment_df['reviews_list'] = sentiment_df['reviews_list'].apply(safe_convert_reviews)
    sentiment_df = sentiment_df[sentiment_df['reviews_list'].str.len() > 0]

    if sentiment_df.empty:
        return {
            "summary_stats": {"total_reviews": 0, "avg_polarity": 0.0, "positive_count": 0, "negative_count": 0, "neutral_count": 0},
            "reviews_df": pd.DataFrame(),
            "top_words": [],
            "word_freq_dict": {}
        }

    # Explode review list
    expanded_df = sentiment_df.explode('reviews_list').reset_index(drop=True)
    
    # Clean review text
    expanded_df['clean_reviews'] = expanded_df['reviews_list'].apply(clean_text)
    expanded_df = expanded_df[expanded_df['clean_reviews'].str.len() > 3].copy()

    # Compute TextBlob sentiment polarity
    expanded_df['sentiment'] = expanded_df['clean_reviews'].apply(lambda txt: TextBlob(txt).sentiment.polarity)

    # Extract keywords
    expanded_df['keywords'] = expanded_df['clean_reviews'].apply(lambda txt: extract_keywords(txt, stop_words))

    # Keyword frequency counter
    all_keywords = [kw for kw_list in expanded_df['keywords'] for kw in kw_list]
    word_freq = Counter(all_keywords)
    top_words = word_freq.most_common(30)
    word_freq_dict = dict(top_words)

    total_reviews = len(expanded_df)
    avg_polarity = float(expanded_df['sentiment'].mean()) if total_reviews > 0 else 0.0
    pos_count = int((expanded_df['sentiment'] > 0.05).sum())
    neg_count = int((expanded_df['sentiment'] < -0.05).sum())
    neu_count = total_reviews - pos_count - neg_count

    results = {
        "summary_stats": {
            "total_reviews": total_reviews,
            "avg_polarity": round(avg_polarity, 3),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count
        },
        "top_words": top_words,
        "word_freq_dict": word_freq_dict,
        "reviews_sample": expanded_df[['name', 'rate', 'clean_reviews', 'sentiment']].head(100).to_dict(orient='records'),
        "polarity_series": expanded_df['sentiment'].tolist()
    }

    return results

def save_sentiment_artifact(models_dir: pathlib.Path, sentiment_results: dict):
    """Save precomputed sentiment analysis results using joblib."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(sentiment_results, models_dir / "sentiment_data.joblib")

def load_sentiment_artifact(models_dir: pathlib.Path) -> dict:
    """Load precomputed sentiment data from models_dir."""
    filepath = models_dir / "sentiment_data.joblib"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Sentiment artifact not found at {filepath}. "
            "Please run 'py -3 train_models.py' first."
        )
    return joblib.load(filepath)
