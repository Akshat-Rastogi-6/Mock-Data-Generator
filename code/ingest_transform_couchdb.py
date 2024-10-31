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

import couchdb
import streamlit as st

def connect_couchdb(db_name="data_paths"):
    """
    Connect to a CouchDB database with the specified name.
    If the database does not exist, it will be created.

    Parameters:
    - db_name (str): The name of the CouchDB database to connect to.

    Returns:
    - db (Database object or None): A CouchDB database instance if connection is successful; None otherwise.
    """
    try:
        # Connect to CouchDB server with admin credentials
        couch = couchdb.Server("http://admin:123456@localhost:5984/")
        
        # Check if the specified database exists; if not, create it
        if db_name not in couch:
            db = couch.create(db_name)  # Create database
        else:
            db = couch[db_name]  # Access existing database
        return db
    except Exception as e:
        # Display an error message if connection fails
        st.write(f"CouchDB connection error: {e}")
        return None
    
def store_data_path_in_couchdb(data_path):
    """
    Store a data path in the CouchDB database named "data_paths".

    Parameters:
    - data_path (str): The file path of the data to be stored.

    This function connects to the "data_paths" database and saves a new document 
    containing the specified data path. If the operation fails, an error message 
    is displayed.
    """
    # Connect to CouchDB and access the 'data_paths' database
    db = connect_couchdb("data_paths")
    if db is not None:
        try:
            # Save the document with data path information
            db.save({"type": "data_path", "path": data_path})
            st.write("Data path stored in CouchDB successfully.")
        except Exception as e:
            # Display an error message if storing fails
            st.write(f"Error storing data path in CouchDB: {e}")

# Function to retrieve the most recently stored data path from CouchDB
def retrieve_data_path_from_couchdb():
    """
    Retrieve the most recent data path from the "data_paths" CouchDB database.

    Returns:
    - data_path (str or None): The most recent data path stored, or None if no data path is found.
    
    This function connects to the "data_paths" database, retrieves the last stored 
    data path document based on document ID sorting, and displays it. If the operation fails, 
    an error message is displayed.
    """
    # Connect to CouchDB and access the 'data_paths' database
    db = connect_couchdb("data_paths")
    if db is not None:
        try:
            # Retrieve all documents with the type "data_path"
            docs = list(db.find({"selector": {"type": "data_path"}}))
            
            # Check if any documents are found
            if docs:
                # Sort documents by ID to get the most recent path (CouchDB uses sequential IDs)
                latest_doc = sorted(docs, key=lambda d: d["_id"], reverse=True)[0]
                
                # Extract the data path from the latest document
                data_path = latest_doc.get("path", "")
                st.write(f"Retrieved data path from CouchDB: {data_path}")
                return data_path
            else:
                # Inform the user if no data paths are found
                st.write("No data path found in CouchDB.")
                return None
        except Exception as e:
            # Display an error message if retrieval fails
            st.write(f"Error retrieving data path from CouchDB: {e}")
            return None
        

def store_model_path(generator, discriminator, gan):
    """
    Store paths for generator, discriminator, and GAN in CouchDB.
    
    Parameters:
    - generator (str): Path to the generator model.
    - discriminator (str): Path to the discriminator model.
    - gan (str): Path to the GAN model.
    """
    db = connect_couchdb()
    if db is not None:
        try:
            # Create a document to store the paths for each model
            model_doc = {
                "generator": generator,
                "discriminator": discriminator,
                "gan": gan,
                "type": "model_paths"
            }
            db.save(model_doc)
            st.write("Model paths stored in CouchDB successfully.")
        except Exception as e:
            st.write(f"Error storing model paths in CouchDB: {e}")

def retrieve_model_path():
    """
    Retrieve the latest model paths for generator, discriminator, and GAN from CouchDB.
    
    Returns:
    - A tuple containing paths for the generator, discriminator, and GAN models.
    """
    db = connect_couchdb()
    if db is not None:
        try:
            # Query to retrieve documents with 'type' set as 'model_paths'
            docs = [doc for doc in db.view('_all_docs', include_docs=True) if doc.doc.get("type") == "model_paths"]
            
            # Retrieve the most recent model paths document based on the latest document
            if docs:
                latest_doc = docs[-1].doc  # Access the latest document
                model_paths = (
                    latest_doc["generator"],
                    latest_doc["discriminator"],
                    latest_doc["gan"]
                )
                st.write("Model paths retrieved successfully.")
                return model_paths
            else:
                st.write("No model paths found in CouchDB.")
                return None, None, None
        except Exception as e:
            st.write(f"Error retrieving model paths from CouchDB: {e}")
            return None, None, None