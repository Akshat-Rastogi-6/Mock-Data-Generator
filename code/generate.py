# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (30 September 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0

from load import data_load
from ingest_transform import delabel, scaler
import numpy as np
import pandas as pd

# Generate synthetic data using the loaded generator model
def generate_synthetic_data(generator, latent_dim, num_samples=1000):
    noise = np.random.normal(0, 1, size=(num_samples, latent_dim))
    synthetic_data = generator.predict(noise)
    return scaler.inverse_transform(synthetic_data)  # Rescale back to original data range

def generate_data(data_path, generator_loaded, num_samples):
    # Generate a new synthetic dataset

    _, columns = data_load(data_path)

    latent_dim = 10

    synthetic_data = generate_synthetic_data(generator_loaded, latent_dim, num_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=columns)

    # Convert labels back to original categories if needed
    synthetic_df = delabel(synthetic_df)  # Assuming delabel function is defined
    return synthetic_df, synthetic_df.head()

def save_data(data):
    csv_data = data.to_csv(index=False)
    return csv_data
