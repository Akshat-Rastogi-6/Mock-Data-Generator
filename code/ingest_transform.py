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


import psycopg2
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

contract_mapping = {'One year' : 0, 'Two year' : 1, 'Month-to-month' : 2}
internet_service_mapping = {'DSL' : 0, 'Fiber optic' : 1}
payment_mapping = {'Mailed check' : 0, 'Bank transfer' : 1, 'Credit card' : 2, 'Electronic check' : 3}
agree_mapping = {'Yes': 0, 'No' :1}
scaler = MinMaxScaler()

def labelling(data):
    global contract_mapping, internet_service_mapping, payment_mapping, agree_mapping

    data['Contract'] = data['Contract'].map(contract_mapping)
    data['InternetService'] = data['InternetService'].map(internet_service_mapping)
    data['PaymentMethod'] = data['PaymentMethod'].map(payment_mapping)
    data['OnlineSecurity'] = data['OnlineSecurity'].map(agree_mapping)
    data['TechSupport'] = data['TechSupport'].map(agree_mapping)
    data['StreamingTV'] = data['StreamingTV'].map(agree_mapping)
    data['StreamingMovies'] = data['StreamingMovies'].map(agree_mapping)
    data['InternetService'] = data['InternetService'].fillna(data['InternetService'].mean())

    return data

def delabel(df):
    # Reverse mappings
    contract_reverse_mapping = {v: k for k, v in contract_mapping.items()}
    internet_service_reverse_mapping = {v: k for k, v in internet_service_mapping.items()}
    payment_reverse_mapping = {v: k for k, v in payment_mapping.items()}
    agree_reverse_mapping = {v: k for k, v in agree_mapping.items()}
    
    # Apply the reverse mappings column by column, after rounding values
    df['Contract'] = df['Contract'].round().astype('int').map(contract_reverse_mapping)
    df['InternetService'] = df['InternetService'].round().astype('int').map(internet_service_reverse_mapping)
    df['PaymentMethod'] = df['PaymentMethod'].round().astype('int').map(payment_reverse_mapping)
    df['OnlineSecurity'] = df['OnlineSecurity'].round().astype('int').map(agree_reverse_mapping)
    df['TechSupport'] = df['TechSupport'].round().astype('int').map(agree_reverse_mapping)
    df['StreamingTV'] = df['StreamingTV'].round().astype('int').map(agree_reverse_mapping)
    df['StreamingMovies'] = df['StreamingMovies'].round().astype('int').map(agree_reverse_mapping)
    df['SeniorCitizen'] = df['SeniorCitizen'].round().astype('int')
    df['PaperlessBilling'] = df['PaperlessBilling'].round().astype('int')
    df['Churn'] = df['Churn'].round().astype('int')
    df['CustomerID'] = df['CustomerID'].round().astype('int')
    df['Tenure'] = df['Tenure'].round().astype('int')
    
    return df

def preprocess(data):
    data = labelling(data)
    scaled_data = scaler.fit_transform(data)
    return scaled_data
    

def connect_postgresql():
    """
    Establish a connection to the PostgreSQL database.
    
    Returns:
    - conn (Connection object or None): PostgreSQL connection object if successful; None otherwise.
    """
    try:
        # Connect to the PostgreSQL database with host, database, user, and password details
        conn = psycopg2.connect(
            host="localhost",
            database="churn",
            user="postgres",
            password="123456"
        )
        return conn
    except Exception as e:
        # Show an error message on the Streamlit app if connection fails
        
        return None
    
def store_data_path_in_postgresql(data_path):
    """
    Store a file path in the PostgreSQL table 'data_paths'.
    
    Parameters:
    - data_path (str): The file path of the data to be stored.
    """
    conn = connect_postgresql()
    if conn is not None:
        try:
            conn.autocommit = True  
            cur = conn.cursor()  # Create a cursor for executing queries
            
            # Create the table if it doesn't exist already
            cur.execute("""
                CREATE TABLE IF NOT EXISTS data_paths (
                    id SERIAL PRIMARY KEY,  -- Unique ID that auto-increments for each entry
                    path TEXT NOT NULL      -- Column to store the data path, cannot be NULL
                )
            """)
            
            # Insert the data path into the table
            cur.execute("INSERT INTO data_paths (path) VALUES (%s)", (data_path,))
            conn.commit()  # Commit the transaction
            st.write("Data path stored in PostgreSQL successfully.")
        except Exception as e:
            # Show an error message on the Streamlit app if insertion fails
            st.write(f"Error storing data path in PostgreSQL: {e}")
        finally:
            # Always close the cursor and the connection, whether successful or not
            if cur:
                cur.close()
            conn.close()


def retrieve_data_path_from_postgresql():
    """
    Retrieve the latest data path stored in the PostgreSQL 'data_paths' table.
    
    Returns:
    - data_path (str or None): The most recent data path if available; None otherwise.
    """
    conn = connect_postgresql()
    if conn is not None:
        try:
            cur = conn.cursor()  # Cursor to execute database queries
            
            # Query the table to get the latest data path (based on highest ID)
            cur.execute("SELECT path FROM data_paths ORDER BY id DESC LIMIT 1")
            result = cur.fetchone()  # Fetch the first result from the query
            
            # Check if a result is found
            if result:
                data_path = result[0]  # Extract the data path from the result
                st.write(f"Retrieved data path from PostgreSQL: {data_path}")
                return data_path
            else:
                # Inform the user if there is no data path in the table
                st.write("No data path found in PostgreSQL.")
                return None
        except Exception as e:
            # Display an error message if retrieval fails
            st.write(f"Error retrieving data path from PostgreSQL: {e}")
            return None
        finally:
            # Close the cursor and connection to free up resources
            if cur:
                cur.close()
            conn.close()

def store_model_path(generator, discriminator, gan):
    conn = connect_postgresql()
    if conn is not None:
        try:
            conn.autocommit = True  
            cur = conn.cursor()

            # Create the 'model_paths' table if it does not exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_paths (
                    id SERIAL PRIMARY KEY,  -- Unique ID that auto-increments for each entry
                    generator TEXT NOT NULL,  -- Column to store the generator path
                    discriminator TEXT NOT NULL,  -- Column to store the discriminator path
                    gan TEXT NOT NULL  -- Column to store the GAN model path
                )
            """)

            # Insert the paths for generator, discriminator, and gan
            cur.execute("INSERT INTO model_paths (generator, discriminator, gan) VALUES (%s, %s, %s)", 
                        (generator, discriminator, gan))
            
            st.write("Model paths stored in PostgreSQL successfully.")
        except Exception as e:
            st.write(f"Error storing model paths in PostgreSQL: {e}")
        finally:
            if cur:
                cur.close()
            conn.close()

def retrieve_model_path():
    """
    Retrieve model paths from the PostgreSQL table 'model_paths'.
    
    Returns:
    - A dictionary containing the paths of the generator, discriminator, and GAN models.
    """
    conn = connect_postgresql()
    model_paths = {}

    if conn is not None:
        try:
            cur = conn.cursor()
            # Query to retrieve the most recent entry from the model_paths table
            cur.execute("SELECT generator, discriminator, gan FROM model_paths ORDER BY id DESC LIMIT 1")
            result = cur.fetchone()  # Fetch the first row of results

            if result:
                model_paths = {
                    'generator': result[0],
                    'discriminator': result[1],
                    'gan': result[2]
                }
                st.write("Model paths retrieved successfully.")
            else:
                st.write("No model paths found in the database.")
        except Exception as e:
            st.write(f"Error retrieving model paths from PostgreSQL: {e}")
        finally:
            if cur:
                cur.close()
            conn.close()
    
    return model_paths['generator'], model_paths['discriminator'], model_paths['gan']
