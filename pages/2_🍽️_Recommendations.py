import pathlib
import sys
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_raw_data, clean_data_for_eda, best_value_restaurants

st.set_page_config(page_title="Recommendations — Flavorlytics", page_icon="🍽️", layout="wide")

st.title("🍽️ Restaurant Recommendation Engine")
st.caption("Personalized rule-based restaurant discovery system based on customer preferences & weights")

@st.cache_data
def get_rec_data():
    data_path = PROJECT_ROOT / "data" / "zomato.csv"
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_data_for_eda(raw_df)
    return cleaned_df

df = get_rec_data()

# Extract unique filter options from dataset
available_cities = sorted([str(c) for c in df['city'].dropna().unique()])

# Extract common rest_types
all_rest_types = df['rest_type'].dropna().astype(str).str.split(',').explode().str.strip().unique()
available_rest_types = sorted(list(set(all_rest_types)))

# Sidebar Controls / Form
with st.sidebar:
    st.header("⚙️ Preference Filters")
    
    selected_city = st.selectbox("Select Target City / Area", available_cities, index=available_cities.index("BTM") if "BTM" in available_cities else 0)
    
    selected_rest_type = st.selectbox("Select Restaurant Category", available_rest_types, index=available_rest_types.index("Casual Dining") if "Casual Dining" in available_rest_types else 0)
    
    max_cost_input = st.slider("Maximum Budget for Two (₹)", min_value=100, max_value=6000, value=1000, step=100)
    
    rating_range_input = st.slider("Target Rating Range", min_value=1.0, max_value=5.0, value=(3.9, 4.2), step=0.1)
    
    st.markdown("---")
    st.subheader("⚖️ Factor Importance Weights")
    st.caption("Adjust sliders to customize your decision criteria (0 = ignore, 1 = maximum priority)")
    
    w_cost = st.slider("Cost Efficiency Weight", 0.0, 1.0, 0.50, 0.05)
    w_rating = st.slider("Rating Score Weight", 0.0, 1.0, 0.40, 0.05)
    w_booking = st.slider("Table Booking Facility Weight", 0.0, 1.0, 0.05, 0.05)
    w_online = st.slider("Online Delivery Facility Weight", 0.0, 1.0, 0.05, 0.05)

# Calculate sum of weights for automatic normalization if sum > 0
total_weight = w_cost + w_rating + w_booking + w_online
if total_weight > 0:
    norm_w_cost = w_cost / total_weight
    norm_w_rating = w_rating / total_weight
    norm_w_booking = w_booking / total_weight
    norm_w_online = w_online / total_weight
else:
    norm_w_cost, norm_w_rating, norm_w_booking, norm_w_online = 0.25, 0.25, 0.25, 0.25

# Main Content Area
st.markdown(f"### 📍 Top Restaurant Recommendations in **{selected_city}** ({selected_rest_type})")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Selected City", selected_city)
col_b.metric("Category", selected_rest_type)
col_c.metric("Max Budget", f"₹{max_cost_input}")
col_d.metric("Target Rating", f"{rating_range_input[0]} - {rating_range_input[1]}")

st.markdown("---")

with st.spinner("Computing best value recommendations..."):
    results_df = best_value_restaurants(
        df=df,
        city=selected_city,
        rest_type=selected_rest_type,
        max_cost=max_cost_input,
        rating_range=rating_range_input,
        cost_weight=norm_w_cost,
        rating_weight=norm_w_rating,
        booking_weight=norm_w_booking,
        online_weight=norm_w_online
    )

if results_df.empty:
    st.warning("⚠️ **No restaurants match your exact search criteria.**")
    st.info("💡 **Suggestions**: Try increasing your budget slider, broadening the rating range, or choosing a different restaurant category.")
else:
    st.success(f"🎉 Found **{len(results_df)}** matching top-value outlets!")
    
    # Format display table
    display_df = results_df.copy()
    display_df['score'] = display_df['score'].apply(lambda s: f"{s:.3f}")
    display_df = display_df.rename(columns={
        'name': 'Restaurant Name',
        'address': 'Address',
        'rate': 'Rating',
        'avg_cost': 'Cost for Two (₹)',
        'online_order': 'Online Order',
        'book_table': 'Table Booking',
        'cuisines': 'Cuisines Offered',
        'score': 'Weighted Score'
    })
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Rating": st.column_config.NumberColumn(format="⭐ %.1f"),
            "Cost for Two (₹)": st.column_config.NumberColumn(format="₹ %d"),
            "Weighted Score": st.column_config.ProgressColumn("Match Score", min_value=0.0, max_value=1.0, format="%.3f")
        },
        hide_index=True
    )
