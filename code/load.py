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

# Import necessary libraries and functions for data loading, preprocessing, model retrieval, and training
import os
from ingest_transform import preprocess  # Function for preprocessing data
from ingest_transform import retrieve_model_path as rmpsql  # Function to retrieve model path from PostgreSQL
from ingest_transform_couchdb import retrieve_model_path as rmpcdb  # Function to retrieve model path from CouchDB
from tensorflow.keras.models import load_model  # Keras function to load pre-trained models
from tensorflow.keras.optimizers import Adam  # Optimizer to recompile models if needed
import pandas as pd  # Data manipulation library
import streamlit as st

# Function to load and preprocess the dataset
def data_load(df):
    # Read data from the provided file path (expects a CSV file)
    data = pd.read_csv(df)
    # Store column names for future reference if needed
    columns = data.columns
    # Apply preprocessing to the dataset (the preprocess function should be defined in ingest_transform)
    df = preprocess(data)
    # Return the preprocessed data and the original columns for further processing
    return df, columns

# Function to load pre-trained models stored locally
def load_pretrained():
    # Define the directory where the models are saved
    model_dir = 'data/saved_models'
    st.write("Model Accuracy for pre-trained model is:", 89.06)
    # Load the generator model using the saved .h5 file
    generator_loaded = load_model(os.path.join(model_dir, 'generator_model.h5'))
    # Load the discriminator model similarly
    discriminator_loaded = load_model(os.path.join(model_dir, 'discriminator_model.h5'))
    # Load the full GAN model if needed for continued training or evaluation
    gan_loaded = load_model(os.path.join(model_dir, 'gan_model.h5'))

    # Recompile the discriminator model if further training is intended
    # This setup is often used in GANs where the discriminator is trained independently
    discriminator_loaded.compile(loss='binary_crossentropy', 
                                 optimizer=Adam(learning_rate=0.0002), 
                                 metrics=['accuracy'])  # Loss function and optimizer
    
    # Set the discriminator as non-trainable within the GAN model to maintain GAN training structure
    discriminator_loaded.trainable = False
    # Recompile the GAN model to ensure it has the correct architecture and optimizer settings
    gan_loaded.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))

    # Return a success message and the loaded generator model for further use in the application
    return "Models loaded successfully!", generator_loaded

# Function to load models based on the user's choice of database (PostgreSQL or CouchDB)
def load_selftrained(database_choice):
    # Retrieve model paths from the selected database
    # If "PostgreSQL" is chosen, rmpsql() is called; otherwise, rmpcdb() is called for CouchDB
    generator_loaded, discriminator_loaded, gan_loaded = rmpsql() if database_choice == "PostgreSQL" else rmpcdb()
    
    # Load the generator model using the retrieved path from the database
    generator_loaded = load_model(generator_loaded)
    
    # Return a success message and the loaded generator model for further processing or usage
    return "Model loaded successfully!", generator_loaded
