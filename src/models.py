import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = ['online_order', 'book_table', 'location', 'rest_type', 'cost', 'type', 'city']

def clean_data_for_prediction(df: pd.DataFrame):
    """
    Cleans data and imputes missing rate values using RandomForestRegressor(votes, cost)
    matching the flavourytics_prediction.ipynb notebook.
    Returns:
      olddf: DataFrame with all rates (including imputed ones) and requisite features.
      imputer_model: Fitted imputer RandomForestRegressor.
    """
    cleaned_df = df[df['location'] != 'Peenya'].drop_duplicates().copy()
    
    # Rename columns to standard names
    cleaned_df = cleaned_df.rename(columns={
        'approx_cost(for two people)': 'cost',
        'listed_in(type)': 'type',
        'listed_in(city)': 'city'
    })
    
    # Process cost column
    cleaned_df['cost'] = cleaned_df['cost'].astype(str).str.replace(',', '', regex=False)
    cleaned_df['cost'] = pd.to_numeric(cleaned_df['cost'], errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['cost']).reset_index(drop=True)
    
    # Separate olddf (non 'NEW' and non '-')
    olddf = cleaned_df[(cleaned_df['rate'] != 'NEW') & (cleaned_df['rate'] != '-')].copy()
    
    # Parse rate
    olddf['rate'] = olddf['rate'].astype(str).str.replace('/5', '', regex=False)
    olddf['rate'] = pd.to_numeric(olddf['rate'], errors='coerce')
    
    # Train rating imputer model for rows with missing rate
    df_complete = olddf[olddf['rate'].notna()].copy()
    df_missing = olddf[olddf['rate'].isna()].copy()
    
    imputer_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    if not df_missing.empty and not df_complete.empty:
        X_imp_train = df_complete[['votes', 'cost']]
        y_imp_train = df_complete['rate']
        imputer_model.fit(X_imp_train, y_imp_train)
        
        df_missing['rate'] = imputer_model.predict(df_missing[['votes', 'cost']])
        olddf = pd.concat([df_complete, df_missing], ignore_index=True)
    else:
        olddf = df_complete
        
    return olddf, imputer_model

def train_success_prediction_model(olddf: pd.DataFrame):
    """
    Fits ColumnTransformer with OneHotEncoder on categorical features
    and RandomForestRegressor on rate target.
    """
    X = olddf[FEATURE_COLUMNS].copy()
    y = olddf['rate']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
             ['online_order', 'book_table', 'location', 'rest_type', 'type', 'city']),
        ],
        remainder='passthrough'
    )
    
    X_prepared = preprocessor.fit_transform(X)
    
    rf_model = RandomForestRegressor(min_samples_leaf=0.0001, random_state=42)
    rf_model.fit(X_prepared, y)
    
    return preprocessor, rf_model

def predict_success(preprocessor, model, online, bookings, location, rest_type, cost, type, city):
    """
    Predicts rating and success/failure status based on inputs.
    Returns: dict with 'predicted_rate' and 'status' ("Success" if rate >= 3.8 else "Failure").
    """
    test_input = pd.DataFrame(
        [[online, bookings, location, rest_type, cost, type, city]],
        columns=FEATURE_COLUMNS
    )
    
    test_input_processed = preprocessor.transform(test_input)
    predicted_rate = float(model.predict(test_input_processed)[0])
    
    status = "Success" if predicted_rate >= 3.8 else "Failure"
    return {
        "predicted_rate": round(predicted_rate, 2),
        "status": status
    }

def save_model_artifacts(models_dir: pathlib.Path, preprocessor, model, imputer_model):
    """Saves fitted preprocessor and models into models_dir using joblib."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(model, models_dir / "success_model.joblib")
    joblib.dump(imputer_model, models_dir / "imputer_model.joblib")

def load_model_artifacts(models_dir: pathlib.Path):
    """Loads preprocessor and success model from models_dir."""
    preprocessor_path = models_dir / "preprocessor.joblib"
    model_path = models_dir / "success_model.joblib"
    
    if not preprocessor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {models_dir}. "
            "Please run 'py -3 train_models.py' first to build and save models."
        )
        
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model
