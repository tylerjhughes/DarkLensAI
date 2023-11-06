import pandas as pd
import numpy as np

def preprocess_data(data_path):
    # Load data
    data = pd.read_csv(data_path)
    
    # Drop duplicates
    data.drop_duplicates(inplace=True)
    
    # Drop missing values
    data.dropna(inplace=True)
    
    # Convert categorical variables to numerical
    data = pd.get_dummies(data, columns=['category'])
    
    # Normalize numerical variables
    numerical_cols = ['age', 'income']
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()
    
    # Save preprocessed data
    data.to_csv('data/processed/preprocessed_data.csv', index=False)
    
    return data