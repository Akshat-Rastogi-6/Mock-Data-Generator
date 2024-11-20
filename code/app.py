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
        # Python 3.10.11
        # Streamlit 1.36.0
# Importing required libraries for Streamlit app functionality and data manipulation
import os
from generate import generate_data, save_data  # Functions for generating and saving data
from load import load_pretrained, load_selftrained  # Load pretrained/self-trained models
from train import train_  # Function for model training
import numpy as np
import pandas as pd
import streamlit as st
from ingest_transform import store_data_path_in_postgresql, retrieve_data_path_from_postgresql  # PostgreSQL data handling
from ingest_transform_couchdb import store_data_path_in_couchdb, retrieve_data_path_from_couchdb  # CouchDB data handling

# Configuring the Streamlit app display settings
st.set_page_config(page_title="Fake Data Generator", page_icon=":cash:", layout="centered")
# Custom title in HTML format for styled header
st.markdown("<h1 style='text-align: center; color: white;'>Fake Data Generator</h1>", unsafe_allow_html=True)
st.divider()  # Adding a horizontal divider for visual organization

# Creating tabs for different functionalities: Model Config, Model Training, Data Generation
tab1, tab2, tab3 = st.tabs(["Model Config", "Model Training", "Data Generation"])
# Default path for data storage; can be customized by the user
default_path = "data/master"
extraction_dir = None  # Initializing extraction directory variable

# Model Config Tab
with tab1:
    st.title("Data Folder Path Storage")  # Title for the first tab

    # Text input for user to specify directory path for data storage
    data_path = st.text_input("Enter the path to the folder ", value=default_path)
    # Dropdown for database selection to store data path
    database_choice = st.selectbox("Select the database to store the data path:", ("PostgreSQL", "CouchDB"))

    # Checking if the entered path exists before storing
    if os.path.exists(data_path):
        if st.button("Store Data Path"):  # Button to confirm and store data path
            # Storing the path based on selected database
            if database_choice == "PostgreSQL":
                store_data_path_in_postgresql(data_path)  # PostgreSQL storage function
            elif database_choice == "CouchDB":
                store_data_path_in_couchdb(data_path)  # CouchDB storage function
    else:
        # Display an alert if the path does not exist
        st.write("The specified path does not exist. Please enter a valid path.")

# Model Training Tab
with tab2:
    st.subheader("Model Training")  # Subheader for model training section
    st.write("This is where you can train the model.")
    st.divider()  # Visual divider

    # Model-specific settings
    model_name = 'GAN'  # Display name of the model
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)
    epochs = st.number_input('Number of Epochs:', min_value=1, max_value=10000, value=100, step=1)  # Number of epochs

    # Button to initiate model training
    if st.button(f"Train {model_name} Model", use_container_width=True):
        with st.status(f"Training {model_name} Model..."):  # Status update during training
            # Retrieve data path from the selected database
            extraction_dir = retrieve_data_path_from_postgresql() if database_choice == "PostgreSQL" else retrieve_data_path_from_couchdb()
            # Locate the data file in the directory and pass it to the training function
            df = data_path + '/' + str([file for file in os.listdir(extraction_dir) if file.endswith('.csv')][0])
            model, score = train_(df, epochs, database_choice)  # Model training function
            st.write(model)  # Display trained model details

        st.success(f"{model}")  # Success message after training completion
        st.write(f"Accuracy: {score}")  # Display the model's accuracy

# Data Generation Tab
with tab3:
    st.subheader("Data Generation")  # Subheader for data generation section
    st.write("Data Generation is here.")
    st.divider()

    # Dropdown for selecting the GAN model for data generation
    st.write("Choose Model for Prediction:")
    num_samples = st.number_input('Number of Samples:', min_value=1, max_value=10000, value=100, step=1)  # Number of samples to generate
    model_choice = st.selectbox("Model", ["Self-trained GAN", "Pretrained GAN"])

    # Button to initiate data generation
    if st.button(f"Generate Fake Data", use_container_width=True):
        with st.status(f"Generating Fake Data..."):  # Status update during data generation
            # Load model based on user choice and retrieve data path from database
            if model_choice == "Pretrained GAN":
                message, generator = load_pretrained()
                st.write(message)
            else:
                message, generator = load_selftrained(database_choice)
            extraction_dir = retrieve_data_path_from_postgresql() if database_choice == "PostgreSQL" else retrieve_data_path_from_couchdb()
            data_path = default_path + '/' + str([file for file in os.listdir(extraction_dir) if file.endswith('.csv')][0])
            st.write(data_path)  # Display selected data path
            generated_data, generate = generate_data(data_path, generator, num_samples)  # Data generation function call

        st.write(generate)  # Display generated data
    try:
        # Button to download the generated data as a CSV file
        st.download_button(
            label="Download Data",
            data=save_data(generated_data),  # Saves data to be downloadable
            file_name="generated_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    except NameError:
        st.text("Please Generate the Data.")  # Message prompting user to generate data if not already done
