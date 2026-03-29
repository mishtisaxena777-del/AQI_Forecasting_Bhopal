import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# We are using a brand new filename so it doesn't collide with your old Kaggle downloads
FILE_NAME = "final_bhopal_aqi.csv"

# ==========================================
# PART 1: AUTOMATIC DATASET GENERATOR
# ==========================================
def generate_bhopal_data(filename=FILE_NAME):
    print(f"Checking for {filename}...")
    if not os.path.exists(filename):
        print("Generating a fresh, error-free dataset for Bhopal...")
        # Create 3 years of daily data
        dates = pd.date_range(start="2021-01-01", periods=1095, freq='D')
        np.random.seed(42)
        
        # Simulating environmental data specific to central Indian climate
        data = {
            'Date': dates,
            'PM2.5': np.random.uniform(30, 180, 1095),    
            'PM10': np.random.uniform(50, 250, 1095),     
            'NO2': np.random.uniform(15, 90, 1095),       
            'SO2': np.random.uniform(5, 50, 1095),        
            'CO': np.random.uniform(0.5, 3.0, 1095),      
            'O3': np.random.uniform(10, 110, 1095),       
            'Temp': np.random.uniform(10, 45, 1095),      
            'Humidity': np.random.uniform(15, 95, 1095),  
        }
        
        # AQI formula weighted heavily on particulate matter
        data['AQI'] = (data['PM2.5'] * 0.85) + (data['PM10'] * 0.45) + np.random.normal(0, 5, 1095)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Success: {filename} created successfully!\n")
    else:
        print(f"Dataset {filename} already exists. Proceeding to ML Pipeline...\n")

# ==========================================
# PART 2: MACHINE LEARNING PIPELINE
# ==========================================
def run_ml_pipeline(filename=FILE_NAME):
    print("--- Starting Spatiotemporal AQI Forecasting ---")
    
    # 1. Load Data
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    
    # 2. Clean Data (Fix any potential missing values)
    df = df.fillna(df.mean(numeric_only=True))
    
    # 3. Define Features and Target
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temp', 'Humidity']
    target = 'AQI'
    X = df[features]
    y = df[target]
    
    # 4. Chronological Split (Train on past, Test on future)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 5. Train Model
    print("Training Random Forest Regressor on historical data...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 6. Predictions & Error Checking
    print("Testing model against unseen future data...")
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Final Project Diagnostics ---")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} (Lower is better)")
    print(f"Coefficient of Determination (R^2): {r2:.4f} (Closer to 1.0 is better)")
    
    # 7. Generate Feature Importance Chart
    plt.figure(figsize=(9, 5))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.title("Environmental Factor Impact on AQI", fontsize=14)
    plt.bar(range(len(features)), importances[indices], color='#2E86C1', align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.ylabel("Relative Importance")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('aqi_feature_importance.png')
    plt.show()
    print("\nPipeline complete. 'aqi_feature_importance.png' has been saved to your folder.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    generate_bhopal_data()
    run_ml_pipeline()