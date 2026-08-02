import pathlib
import pandas as pd
import numpy as np

def load_raw_data(data_path: pathlib.Path) -> pd.DataFrame:
    """Load raw zomato csv file."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}. "
            "Please follow README instructions to download data/zomato.csv."
        )
    df = pd.read_csv(data_path)
    return df

def clean_data_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data for EDA and Recommendations module:
    - Exclude 'Peenya' location
    - Remove duplicates
    - Clean 'rate' column (e.g. '4.1/5' -> 4.1, missing/NEW/- -> NaN)
    - Clean 'approx_cost(for two people)' column -> numeric 'cost' and 'avg_cost'
    - Standardize column names: listed_in(city) -> city, listed_in(type) -> type
    """
    cleaned_df = df.copy()
    cleaned_df = cleaned_df[cleaned_df['location'] != 'Peenya'].drop_duplicates()
    
    # Standardize column names
    cleaned_df = cleaned_df.rename(columns={
        'approx_cost(for two people)': 'cost',
        'listed_in(type)': 'type',
        'listed_in(city)': 'city'
    })
    
    # Process rate column
    cleaned_df['rate_raw'] = cleaned_df['rate']
    cleaned_df['rate'] = cleaned_df['rate'].astype(str).str.replace('/5', '', regex=False)
    cleaned_df['rate'] = pd.to_numeric(cleaned_df['rate'], errors='coerce')
    
    # Process cost column
    cleaned_df['cost'] = cleaned_df['cost'].astype(str).str.replace(',', '', regex=False)
    cleaned_df['cost'] = pd.to_numeric(cleaned_df['cost'], errors='coerce')
    cleaned_df['avg_cost'] = cleaned_df['cost']
    
    # Drop rows with missing cost for EDA consistency
    cleaned_df = cleaned_df.dropna(subset=['cost']).reset_index(drop=True)
    
    return cleaned_df

def best_value_restaurants(
    df: pd.DataFrame,
    city: str,
    rest_type: str,
    max_cost: float = 1000.0,
    rating_range: tuple = (3.9, 4.2),
    cost_weight: float = 0.5,
    rating_weight: float = 0.4,
    booking_weight: float = 0.05,
    online_weight: float = 0.05
) -> pd.DataFrame:
    """
    Rule-based recommendation function matching Notebook 1 implementation.
    Filters by city, rest_type, rating_range (+/- 0.3 tolerance), max_cost (*1.2 tolerance).
    Returns weighted score sorted top 10 DataFrame.
    """
    if df.empty or 'city' not in df.columns or 'rest_type' not in df.columns:
        return pd.DataFrame()

    filtered_df = df[
        (df['city'].astype(str).str.lower() == city.lower()) &
        (df['rest_type'].astype(str).str.contains(rest_type, case=False, na=False))
    ].copy()

    min_rating, max_rating = rating_range
    expanded_df = filtered_df[
        (filtered_df['rate'] >= min_rating - 0.3) &
        (filtered_df['rate'] <= max_rating + 0.3) &
        (filtered_df['cost'] <= max_cost * 1.2)
    ].copy()

    if expanded_df.empty:
        return pd.DataFrame()

    # Cost score (lower cost is better)
    expanded_df['cost_score'] = 1 - (expanded_df['cost'] / max_cost)
    
    # Rating score
    rating_range_span = (max_rating - min_rating) if max_rating != min_rating else 1.0
    expanded_df['rating_score'] = (expanded_df['rate'] - min_rating) / rating_range_span
    
    # Booking and Online scores
    expanded_df['booking_score'] = expanded_df['book_table'].map({'Yes': 1, 'No': 0}).fillna(0)
    expanded_df['online_score'] = expanded_df['online_order'].map({'Yes': 1, 'No': 0}).fillna(0)

    # Calculate weighted final score
    expanded_df['score'] = (
        cost_weight * expanded_df['cost_score'] +
        rating_weight * expanded_df['rating_score'] +
        booking_weight * expanded_df['booking_score'] +
        online_weight * expanded_df['online_score']
    )

    best_value = expanded_df.sort_values(['score', 'cost'], ascending=[False, True])
    
    output_cols = ['name', 'address', 'rate', 'cost', 'online_order', 'book_table', 'cuisines', 'score']
    available_cols = [col for col in output_cols if col in best_value.columns]
    
    result = best_value[available_cols].copy()
    if 'cost' in result.columns:
        result = result.rename(columns={'cost': 'avg_cost'})
        
    return result.head(10).reset_index(drop=True)
