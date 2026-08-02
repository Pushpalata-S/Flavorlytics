import pathlib
import sys
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_raw_data, clean_data_for_eda
from src.models import load_model_artifacts, predict_success

st.set_page_config(page_title="Success Prediction — Flavorlytics", page_icon="🤖", layout="wide")

st.title("🤖 Restaurant Success Prediction")
st.caption("Evaluate proposed restaurant parameters using our Machine Learning model (Random Forest Regressor)")

# Load persisted model artifacts via cache
@st.cache_resource
def get_cached_models():
    models_dir = PROJECT_ROOT / "models"
    preprocessor, model = load_model_artifacts(models_dir)
    return preprocessor, model

# Load dataset to populate dropdown selections
@st.cache_data
def get_prediction_dropdown_options():
    data_path = PROJECT_ROOT / "data" / "zomato.csv"
    raw_df = load_raw_data(data_path)
    df = clean_data_for_eda(raw_df)
    
    locations = sorted([str(l) for l in df['location'].dropna().unique()])
    rest_types = sorted([str(r) for r in df['rest_type'].dropna().unique()])
    categories = sorted([str(t) for t in df['type'].dropna().unique()])
    cities = sorted([str(c) for c in df['city'].dropna().unique()])
    
    return locations, rest_types, categories, cities

try:
    preprocessor, model = get_cached_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)

try:
    locations, rest_types, categories, cities = get_prediction_dropdown_options()
except Exception as e:
    locations, rest_types, categories, cities = [], [], [], []

if not models_loaded:
    st.error(f"⚠️ Failed to load machine learning models: {model_error}")
    st.warning("Please run `py -3 train_models.py` in your terminal to build and save model artifacts.")
    st.stop()

st.markdown("### 📝 Enter Proposed Restaurant Configuration")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_online = st.selectbox("Online Order Availability", ["Yes", "No"], index=0)
        input_booking = st.selectbox("Table Booking Availability", ["Yes", "No"], index=1)
        input_cost = st.number_input("Estimated Cost for Two (₹)", min_value=50, max_value=10000, value=500, step=50)

    with col2:
        input_location = st.selectbox("Location / Area", locations, index=locations.index("Koramangala 5th Block") if "Koramangala 5th Block" in locations else 0)
        input_city = st.selectbox("Listed City", cities, index=cities.index("Koramangala 5th Block") if "Koramangala 5th Block" in cities else 0)

    with col3:
        input_rest_type = st.selectbox("Restaurant Type", rest_types, index=rest_types.index("Casual Dining") if "Casual Dining" in rest_types else 0)
        input_type = st.selectbox("Listed Category Type", categories, index=categories.index("Dine-out") if "Dine-out" in categories else 0)

    submit_btn = st.form_submit_button("🚀 Predict Restaurant Success Rate", use_container_width=True)

if submit_btn:
    with st.spinner("Executing Random Forest model inference..."):
        result = predict_success(
            preprocessor=preprocessor,
            model=model,
            online=input_online,
            bookings=input_booking,
            location=input_location,
            rest_type=input_rest_type,
            cost=input_cost,
            type=input_type,
            city=input_city
        )
        
    predicted_rate = result["predicted_rate"]
    status = result["status"]
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Result Summary")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="Predicted Customer Rating",
            value=f"{predicted_rate:.2f} / 5.0",
            delta=f"{predicted_rate - 3.8:+.2f} vs Success Benchmark (3.8)"
        )
        
    with res_col2:
        if status == "Success":
            st.success("### 🎉 OUTCOME: SUCCESS")
            st.markdown("The model predicts this restaurant configuration will achieve a high rating ($\ge 3.8$) and perform successfully in Bangalore!")
        else:
            st.error("### ⚠️ OUTCOME: FAILURE")
            st.markdown("The model predicts this restaurant configuration may struggle to reach the 3.8 success benchmark. Consider enabling online ordering, table booking, or tweaking pricing.")

    st.markdown("---")
    st.markdown("#### 🔍 Input Configuration Breakdown")
    input_summary = pd.DataFrame([{
        "Online Order": input_online,
        "Table Booking": input_booking,
        "Cost for Two": f"₹{input_cost}",
        "Location": input_location,
        "Listed City": input_city,
        "Restaurant Type": input_rest_type,
        "Category": input_type
    }])
    st.table(input_summary)
