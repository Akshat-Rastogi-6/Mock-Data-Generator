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
# Import necessary libraries
import couchdb  # CouchDB adapter for Python
import streamlit as st  # Streamlit library for web applications

# Function to connect to CouchDB
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
        # Establish a connection to the CouchDB server, using admin credentials
        couch = couchdb.Server("http://admin:123456@localhost:5984/")
        
        # Check if the specified database exists in CouchDB
        # If it does not exist, create a new database with that name
        if db_name not in couch:
            db = couch.create(db_name)  # Create the database
        else:
            db = couch[db_name]  # Connect to the existing database
        return db
    except Exception as e:
        # Display an error message on the Streamlit app if connection fails
        st.write(f"CouchDB connection error: {e}")
        return None

# Function to store a file path in CouchDB
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
            # Create a new document in the database with a specific structure
            # Document type is "data_path" and it stores the 'path' field
            db.save({"type": "data_path", "path": data_path})
            st.write("Data path stored in CouchDB successfully.")
        except Exception as e:
            # Display an error message on the Streamlit app if storing fails
            st.write(f"Error storing data path in CouchDB: {e}")

# Function to retrieve the most recent data path from CouchDB
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
            # Query all documents where 'type' field is 'data_path'
            docs = list(db.find({"selector": {"type": "data_path"}}))
            
            # Check if any documents were found in the database
            if docs:
                # Sort documents by document ID (newest ID first) to get the latest entry
                latest_doc = sorted(docs, key=lambda d: d["_id"], reverse=True)[0]
                
                # Extract and return the 'path' from the latest document
                data_path = latest_doc.get("path", "")
                st.write(f"Retrieved data path from CouchDB: {data_path}")
                return data_path
            else:
                # Inform the user if no data paths were found in the database
                st.write("No data path found in CouchDB.")
                return None
        except Exception as e:
            # Display an error message on the Streamlit app if retrieval fails
            st.write(f"Error retrieving data path from CouchDB: {e}")
            return None

# Function to store model paths in CouchDB
def store_model_path(generator, discriminator, gan):
    """
    Store paths for generator, discriminator, and GAN in CouchDB.
    
    Parameters:
    - generator (str): Path to the generator model.
    - discriminator (str): Path to the discriminator model.
    - gan (str): Path to the GAN model.
    """
    # Connect to the default CouchDB database (or "data_paths" if specified)
    db = connect_couchdb()
    if db is not None:
        try:
            # Create a document with paths for each model (generator, discriminator, GAN)
            # Set 'type' field to "model_paths" to identify it in the database
            model_doc = {
                "generator": generator,
                "discriminator": discriminator,
                "gan": gan,
                "type": "model_paths"
            }
            # Save the document in the database
            db.save(model_doc)
            st.write("Model paths stored in CouchDB successfully.")
        except Exception as e:
            # Display an error message if storing fails
            st.write(f"Error storing model paths in CouchDB: {e}")

# Function to retrieve the most recent model paths from CouchDB
def retrieve_model_path():
    """
    Retrieve the latest model paths for generator, discriminator, and GAN from CouchDB.
    
    Returns:
    - A tuple containing paths for the generator, discriminator, and GAN models.
    
    This function connects to the database, finds documents with type 'model_paths',
    and retrieves paths for each model (generator, discriminator, GAN) from the latest document.
    """
    # Connect to CouchDB and access the default database (or "data_paths" if specified)
    db = connect_couchdb()
    if db is not None:
        try:
            # Retrieve documents with 'type' set to 'model_paths'
            docs = [doc for doc in db.view('_all_docs', include_docs=True) if doc.doc.get("type") == "model_paths"]
            
            # Check if model path documents were found
            if docs:
                # Get the last document in the list (latest model paths)
                latest_doc = docs[-1].doc  # Access the most recent document
                
                # Extract the paths for generator, discriminator, and GAN models
                model_paths = (
                    latest_doc["generator"],
                    latest_doc["discriminator"],
                    latest_doc["gan"]
                )
                st.write("Model paths retrieved successfully.")
                return model_paths
            else:
                # Inform the user if no model paths were found
                st.write("No model paths found in CouchDB.")
                return None, None, None
        except Exception as e:
            # Display an error message on the Streamlit app if retrieval fails
            st.write(f"Error retrieving model paths from CouchDB: {e}")
            return None, None, None
