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
# Import necessary modules
from load import data_load  # Custom module for data loading
from ingest_transform import delabel, scaler  # Custom functions for label transformation and scaling
import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation

# Function to generate synthetic data using a pre-trained generator model
def generate_synthetic_data(generator, latent_dim, num_samples=1000):
    """
    Generate synthetic data samples using a pre-trained generator model.

    Parameters:
    - generator (model): The pre-trained generator model for generating synthetic data.
    - latent_dim (int): The dimension of the latent space for generating input noise.
    - num_samples (int): The number of synthetic data samples to generate (default is 1000).

    Returns:
    - synthetic_data (array): The generated synthetic data, rescaled to the original data range.
    """
    # Generate random noise as input for the generator model
    # Noise is sampled from a normal distribution with mean=0 and std=1
    noise = np.random.normal(0, 1, size=(num_samples, latent_dim))
    
    # Use the generator model to generate synthetic data from the noise
    synthetic_data = generator.predict(noise)
    
    # Rescale the generated synthetic data back to the original data range using the scaler
    return scaler.inverse_transform(synthetic_data)  # Ensure data aligns with original range

# Main function to generate a synthetic dataset
def generate_data(data_path, generator_loaded, num_samples):
    """
    Generate a synthetic dataset by loading real data, generating synthetic data,
    and transforming it back to its original format if needed.

    Parameters:
    - data_path (str): The file path to the real dataset (for loading column names).
    - generator_loaded (model): The loaded generator model for synthetic data generation.
    - num_samples (int): The number of synthetic data samples to generate.

    Returns:
    - synthetic_df (DataFrame): The complete synthetic dataset as a DataFrame.
    - synthetic_df.head() (DataFrame): The first few rows of the synthetic dataset for preview.
    """
    # Load the dataset to retrieve column names only (we don't need the data here)
    _, columns = data_load(data_path)
    
    # Define the dimension of the latent space for generating synthetic data
    latent_dim = 10

    # Generate synthetic data using the pre-trained generator and latent dimensions
    synthetic_data = generate_synthetic_data(generator_loaded, latent_dim, num_samples)
    
    # Convert synthetic data array into a DataFrame with the original dataset's column names
    synthetic_df = pd.DataFrame(synthetic_data, columns=columns)

    # If synthetic data includes encoded labels, convert them back to original categories
    # This assumes 'delabel' function transforms encoded labels back to their original format
    synthetic_df = delabel(synthetic_df)
    
    # Return the complete synthetic dataset and a preview of the first few rows
    return synthetic_df, synthetic_df.head()

# Function to save the synthetic dataset to a CSV format
def save_data(data):
    """
    Convert a DataFrame to CSV format for easy storage or download.

    Parameters:
    - data (DataFrame): The DataFrame containing the synthetic data to be saved.

    Returns:
    - csv_data (str): The CSV data as a string with no index column (for streamlined storage).
    """
    # Convert DataFrame to CSV format without the index column and return it as a string
    csv_data = data.to_csv(index=False)
    return csv_data
