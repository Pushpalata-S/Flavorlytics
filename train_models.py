import pathlib
import sys
import time

# Ensure src package is in Python path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_raw_data, clean_data_for_eda
from src.models import clean_data_for_prediction, train_success_prediction_model, save_model_artifacts
from src.sentiment import process_sentiment_analysis, save_sentiment_artifact

def main():
    print("=" * 60)
    print("Flavorlytics — Offline Model Training & Data Precomputation")
    print("=" * 60)
    
    data_path = PROJECT_ROOT / "data" / "zomato.csv"
    models_dir = PROJECT_ROOT / "models"
    
    start_time = time.time()
    
    # 1. Load Raw Data
    print(f"\n[1/4] Loading dataset from: {data_path}")
    raw_df = load_raw_data(data_path)
    print(f"      Loaded {raw_df.shape[0]:,} rows and {raw_df.shape[1]} columns.")
    
    # 2. Train Success Prediction Model & Imputer
    print("\n[2/4] Cleaning data & training Success Prediction ML Models...")
    olddf, imputer_model = clean_data_for_prediction(raw_df)
    print(f"      Prepared dataset: {olddf.shape[0]:,} rows.")
    
    preprocessor, success_model = train_success_prediction_model(olddf)
    print("      Model training complete.")
    
    # 3. Save Model Artifacts
    print(f"\n[3/4] Saving model artifacts to: {models_dir}")
    save_model_artifacts(models_dir, preprocessor, success_model, imputer_model)
    print("      Saved: preprocessor.joblib, success_model.joblib, imputer_model.joblib")
    
    # 4. Precompute Sentiment Analysis Artifacts
    print("\n[4/4] Processing low-rated restaurant reviews sentiment analysis...")
    eda_df = clean_data_for_eda(raw_df)
    sentiment_results = process_sentiment_analysis(eda_df)
    save_sentiment_artifact(models_dir, sentiment_results)
    print(f"      Analyzed {sentiment_results['summary_stats']['total_reviews']:,} reviews.")
    print("      Saved: sentiment_data.joblib")
    
    elapsed = time.time() - start_time
    print(f"\nSUCCESS: Pipeline finished successfully in {elapsed:.2f} seconds.")
    print("=" * 60)

if __name__ == "__main__":
    main()
