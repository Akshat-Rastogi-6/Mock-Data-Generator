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

import os
from ingest_transform import preprocess
from ingest_transform import retrieve_model_path as rmpsql
from ingest_transform_couchdb import retrieve_model_path as rmpcdb
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd

def data_load(df):
    data = pd.read_csv(df)
    columns = data.columns
    df = preprocess(data)
    return df, columns

def load_pretrained():
    model_dir = 'data/saved_models'
    # Load the models
    generator_loaded = load_model(os.path.join(model_dir, 'generator_model.h5'))
    discriminator_loaded = load_model(os.path.join(model_dir, 'discriminator_model.h5'))
    gan_loaded = load_model(os.path.join(model_dir, 'gan_model.h5'))

    # Recompile discriminator if needed for further training
    discriminator_loaded.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

    # Freeze the discriminator within the GAN model and recompile the GAN
    discriminator_loaded.trainable = False
    gan_loaded.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))

    return "Models loaded successfully!", generator_loaded

def load_selftrained(database_choice):
    generator_loaded, discriminator_loaded, gan_loaded = rmpsql() if database_choice == "PostgreSQL" else rmpcdb()
    generator_loaded = load_model(generator_loaded)
    return "Model loaded successfully!",generator_loaded