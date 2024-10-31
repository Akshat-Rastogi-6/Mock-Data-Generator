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
from generate import generate_data, save_data
from load import load_pretrained, load_selftrained
from train import train_
import numpy as np
import pandas as pd
import streamlit as st
from ingest_transform import store_data_path_in_postgresql, retrieve_data_path_from_postgresql  # PostgreSQL data handling
from ingest_transform_couchdb import store_data_path_in_couchdb, retrieve_data_path_from_couchdb 

st.set_page_config(page_title="Fake Data Generator", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Fake Data Generator</h1>", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 =  st.tabs(["Model Config", "Model Training", "Data Generation"])
default_path = "data/master"
extraction_dir = None

with tab1:
    st.title("Data Folder Path Storage")  # Title for the first tab

    # Input field to take the directory path for image storage
    data_path = st.text_input("Enter the path to the folder ", value=default_path)  # User input for directory
    # Dropdown to select the database for storing the data path
    database_choice = st.selectbox("Select the database to store the data path:", ("PostgreSQL", "CouchDB"))

    # Check if the provided path exists
    if os.path.exists(data_path):
        # Button to store the data path in the selected database
        if st.button("Store Data Path"):
            # Based on user's choice, store data path in the respective database
            if database_choice == "PostgreSQL":
                store_data_path_in_postgresql(data_path)  # Function to store path in PostgreSQL
            elif database_choice == "CouchDB":
                store_data_path_in_couchdb(data_path)  # Function to store path in CouchDB
    else:
        # Message if the specified path does not exist
        st.write("The specified path does not exist. Please enter a valid path.")  # Alert for invalid path

# Second tab: Model Training
with tab2:
    st.subheader("Model Training")  # Subheader for the Model Training section
    st.write("This is where you can train the model.")  # Brief description of the section
    st.divider()  # Adding a horizontal divider for visual separation

    # Setting the model name for display purposes
    model_name = 'GAN Model Training'  # The name of the model being trained
    # Displaying the model name as a header
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)
    epochs = st.number_input('Number of Epochs:', min_value=1, max_value=10000, value=100, step=1)

    if st.button(f"Train {model_name} Model", use_container_width=True):
        with st.status(f"Training {model_name} Model..."):
            # Retrieving the extraction directory based on the selected database
            # Use PostgreSQL or CouchDB to retrieve the path where images are stored
            extraction_dir = retrieve_data_path_from_postgresql() if database_choice == "PostgreSQL" else retrieve_data_path_from_couchdb()
            df = data_path + '/' + str([file for file in os.listdir(extraction_dir) if file.endswith('.csv')][0])
            model, score = train_(df, epochs, database_choice)
            st.write(model)

        st.success(f"{model}")

        st.write(f"Accuracy: {score}")

with tab3:
    st.subheader("Data Generation")
    st.write("Data Generation is here.")
    st.divider()

    st.write("Choose Model for Prediction:")  # Instruction for the user regarding model selection
    # Dropdown for selecting the model to use for prediction
    num_samples = st.number_input('Number of Samples:', min_value=1, max_value=10000, value=100, step=1)
    model_choice = st.selectbox("Model", ["Self-trained GAN", "Pretrained GAN"])

    if st.button(f"Generate Fake Data", use_container_width=True):
        with st.status(f"Generating Fake Data..."):
            if model_choice == "Pretrained GAN":
                message, generator = load_pretrained()
                st.write(message)
            else:
                message, generator = load_selftrained(database_choice)
            extraction_dir = retrieve_data_path_from_postgresql() if database_choice == "PostgreSQL" else retrieve_data_path_from_couchdb()
            data_path = default_path + '/' + str([file for file in os.listdir(extraction_dir) if file.endswith('.csv')][0])
            st.write(data_path)
            generated_data, generate = generate_data(data_path, generator, num_samples)
        
        st.write(generate)
    try:
        st.download_button(
            label="Download Data",
            data=save_data(generated_data),
            file_name="generated_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    except NameError:
        st.text("Please Generate the Data.")
    


