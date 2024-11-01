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


# Import necessary libraries and modules
import psycopg2  # PostgreSQL adapter for Python
import streamlit as st  # Streamlit library for web applications
from sklearn.preprocessing import MinMaxScaler  # Scaler to normalize data values

# Define mappings to convert categorical values to numerical representations for model use
contract_mapping = {'One year': 0, 'Two year': 1, 'Month-to-month': 2}
internet_service_mapping = {'DSL': 0, 'Fiber optic': 1}
payment_mapping = {'Mailed check': 0, 'Bank transfer': 1, 'Credit card': 2, 'Electronic check': 3}
agree_mapping = {'Yes': 0, 'No': 1}

# Initialize a scaler instance for data normalization
scaler = MinMaxScaler()

# Function to label categorical data with numeric values for model compatibility
def labelling(data):
    """
    Convert categorical features to numerical values using predefined mappings.

    Parameters:
    - data (DataFrame): Input data with categorical features.

    Returns:
    - data (DataFrame): Data with categorical features converted to numerical values.
    """
    global contract_mapping, internet_service_mapping, payment_mapping, agree_mapping

    # Apply mappings to the respective columns
    data['Contract'] = data['Contract'].map(contract_mapping)
    data['InternetService'] = data['InternetService'].map(internet_service_mapping)
    data['PaymentMethod'] = data['PaymentMethod'].map(payment_mapping)
    data['OnlineSecurity'] = data['OnlineSecurity'].map(agree_mapping)
    data['TechSupport'] = data['TechSupport'].map(agree_mapping)
    data['StreamingTV'] = data['StreamingTV'].map(agree_mapping)
    data['StreamingMovies'] = data['StreamingMovies'].map(agree_mapping)

    # Fill any missing values in 'InternetService' with the mean value of the column
    data['InternetService'] = data['InternetService'].fillna(data['InternetService'].mean())

    return data

# Function to revert numerical labels to their original categorical values
def delabel(df):
    """
    Convert numeric labels back to their original categorical values.

    Parameters:
    - df (DataFrame): DataFrame with numerical labels.

    Returns:
    - df (DataFrame): Data with categorical features restored to their original string values.
    """
    # Create reverse mappings for converting numbers back to category labels
    contract_reverse_mapping = {v: k for k, v in contract_mapping.items()}
    internet_service_reverse_mapping = {v: k for k, v in internet_service_mapping.items()}
    payment_reverse_mapping = {v: k for k, v in payment_mapping.items()}
    agree_reverse_mapping = {v: k for k, v in agree_mapping.items()}
    
    # Apply reverse mappings to convert numeric columns back to categorical format
    df['Contract'] = df['Contract'].round().astype('int').map(contract_reverse_mapping)
    df['InternetService'] = df['InternetService'].round().astype('int').map(internet_service_reverse_mapping)
    df['PaymentMethod'] = df['PaymentMethod'].round().astype('int').map(payment_reverse_mapping)
    df['OnlineSecurity'] = df['OnlineSecurity'].round().astype('int').map(agree_reverse_mapping)
    df['TechSupport'] = df['TechSupport'].round().astype('int').map(agree_reverse_mapping)
    df['StreamingTV'] = df['StreamingTV'].round().astype('int').map(agree_reverse_mapping)
    df['StreamingMovies'] = df['StreamingMovies'].round().astype('int').map(agree_reverse_mapping)

    # Convert other necessary columns to integers for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].round().astype('bool')
    df['PaperlessBilling'] = df['PaperlessBilling'].round().astype('bool')
    df['Churn'] = df['Churn'].round().astype('bool')
    df['CustomerID'] = df['CustomerID'].round().astype('int')
    df['Tenure'] = df['Tenure'].round().astype('int')
    
    return df

# Function to preprocess data by applying labelling and scaling
def preprocess(data):
    """
    Label categorical data and scale numerical values for model compatibility.

    Parameters:
    - data (DataFrame): Input DataFrame to be processed.

    Returns:
    - scaled_data (array): Processed and scaled data ready for model input.
    """
    data = labelling(data)  # Apply labelling function
    scaled_data = scaler.fit_transform(data)  # Scale the data
    return scaled_data  # Return scaled data for model input

# Establish a connection to PostgreSQL database
def connect_postgresql():
    """
    Connect to PostgreSQL database.

    Returns:
    - conn (Connection object or None): Connection object if successful; None otherwise.
    """
    try:
        # Create a connection using database credentials
        conn = psycopg2.connect(
            host="localhost",
            database="churn",
            user="postgres",
            password="123456"
        )
        return conn
    except Exception as e:
        st.write("Database connection failed.")  # Display error in Streamlit
        return None

# Function to store data path in PostgreSQL database
def store_data_path_in_postgresql(data_path):
    """
    Store a file path in the 'data_paths' table in PostgreSQL.

    Parameters:
    - data_path (str): File path to be stored in the database.
    """
    conn = connect_postgresql()  # Connect to database
    if conn:
        try:
            conn.autocommit = True  # Enable auto-commit mode
            cur = conn.cursor()  # Cursor for executing SQL commands
            
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS data_paths (
                    id SERIAL PRIMARY KEY,
                    path TEXT NOT NULL
                )
            """)
            
            # Insert data path into table
            cur.execute("INSERT INTO data_paths (path) VALUES (%s)", (data_path,))
            conn.commit()
            st.write("Data path stored successfully.")
        except Exception as e:
            st.write(f"Error storing data path: {e}")
        finally:
            cur.close()  # Close cursor
            conn.close()  # Close connection

# Function to retrieve the latest data path from PostgreSQL
def retrieve_data_path_from_postgresql():
    """
    Retrieve the latest data path from the 'data_paths' table in PostgreSQL.

    Returns:
    - data_path (str or None): Most recent data path if found; None otherwise.
    """
    conn = connect_postgresql()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT path FROM data_paths ORDER BY id DESC LIMIT 1")
            result = cur.fetchone()  # Fetch latest path
            
            if result:
                data_path = result[0]
                st.write(f"Retrieved data path: {data_path}")
                return data_path
            else:
                st.write("No data path found.")
                return None
        except Exception as e:
            st.write(f"Error retrieving data path: {e}")
            return None
        finally:
            cur.close()
            conn.close()

# Store paths of the generator, discriminator, and GAN models in PostgreSQL
def store_model_path(generator, discriminator, gan):
    """
    Store model paths in the 'model_paths' table in PostgreSQL.

    Parameters:
    - generator (str): Path to the generator model.
    - discriminator (str): Path to the discriminator model.
    - gan (str): Path to the GAN model.
    """
    conn = connect_postgresql()
    if conn:
        try:
            conn.autocommit = True
            cur = conn.cursor()

            # Create table if it does not exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_paths (
                    id SERIAL PRIMARY KEY,
                    generator TEXT NOT NULL,
                    discriminator TEXT NOT NULL,
                    gan TEXT NOT NULL
                )
            """)

            # Insert model paths into table
            cur.execute("INSERT INTO model_paths (generator, discriminator, gan) VALUES (%s, %s, %s)", 
                        (generator, discriminator, gan))
            st.write("Model paths stored successfully.")
        except Exception as e:
            st.write(f"Error storing model paths: {e}")
        finally:
            cur.close()
            conn.close()

# Retrieve model paths from PostgreSQL
def retrieve_model_path():
    """
    Retrieve paths of the most recent generator, discriminator, and GAN models from PostgreSQL.

    Returns:
    - generator, discriminator, gan (str): Paths of the models if found.
    """
    conn = connect_postgresql()
    model_paths = {}

    if conn:
        try:
            cur = conn.cursor()
            # Fetch latest paths
            cur.execute("SELECT generator, discriminator, gan FROM model_paths ORDER BY id DESC LIMIT 1")
            result = cur.fetchone()

            if result:
                model_paths = {
                    'generator': result[0],
                    'discriminator': result[1],
                    'gan': result[2]
                }
                st.write("Model paths retrieved successfully.")
            else:
                st.write("No model paths found.")
        except Exception as e:
            st.write(f"Error retrieving model paths: {e}")
        finally:
            cur.close()
            conn.close()
    
    return model_paths['generator'], model_paths['discriminator'], model_paths['gan']

