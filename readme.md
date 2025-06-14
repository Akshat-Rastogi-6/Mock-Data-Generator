# DIY-Deep-Learning-NN-PyTorch

This is the Fake Data Generation for Churn Prediction branch.

---

## Fake Data Generation for Churn Prediction

### Business Case

A telecom company wants to simulate customer churn data to better understand patterns that may lead to customer attrition. Using synthetic data, the company can experiment with different churn scenarios, test predictive models, and create targeted strategies to reduce churn without compromising actual customer data privacy.

### Industry
Telecommunications

### Problem Statement

Creating realistic data for customer churn analysis is challenging, especially when privacy regulations restrict access to sensitive customer information. By generating synthetic churn data, the company can develop and test predictive models to understand how different factors influence churn, allowing for actionable insights without using actual customer records.

### Objective

The objective is to develop a process for generating realistic synthetic churn data that resembles real-world data in key characteristics, such as:

- **Demographics** (e.g., age, gender)
- **Service Subscriptions** (e.g., internet, phone, streaming services)
- **Billing and Contract Information** (e.g., monthly charges, contract type)

This synthetic data will allow for training and validating predictive models to identify patterns that lead to churn, enabling the telecom company to strategize on customer retention efforts effectively.

---

### Database Setup

#### PostgreSQL Setup

1. **Install PostgreSQL**
   - Download and install PostgreSQL from [official website](https://www.postgresql.org/download/)
   - Verify installation: `psql --version`

2. **Create Database and User**
   ```sql
   CREATE DATABASE fake_data_db;
   CREATE USER fake_data_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE fake_data_db TO fake_data_user;
   ```

3. **Configure Environment Variables**
   ```
   POSTGRES_DB=fake_data_db
   POSTGRES_USER=fake_data_user
   POSTGRES_PASSWORD=your_password
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

4. **Initialize Tables**
   ```sql
   CREATE TABLE model_paths (
       id SERIAL PRIMARY KEY,
       path VARCHAR(255) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE TABLE data_paths (
       id SERIAL PRIMARY KEY,
       path VARCHAR(255) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

   **Note**: Do these in ingest_transform.py file.

#### CouchDB Setup

1. **Install CouchDB**
   - Download and install CouchDB from [official website](https://couchdb.apache.org/#download)
   - Verify installation by accessing: `http://localhost:5984/_utils`

2. **Create Database**
   - Open Fauxton UI (CouchDB Dashboard)
   - Create two databases:
     - `model_paths`
     - `data_paths`

3. **Configure Environment Variables**
   ```
   COUCHDB_USER=admin
   COUCHDB_PASSWORD=your_password
   COUCHDB_HOST=localhost
   COUCHDB_PORT=5984
   COUCHDB_MODEL_DB=model_paths
   COUCHDB_DATA_DB=data_paths
   ```

  **Note**: Do these in ingest_transform_couchdb.py file.


### Directory Structure

---
- **code/**: Contains the source code files for the project.
  - `__pycache__/`: Directory for compiled Python files.
  - `app.py`: Main application script.
  - `generate.py`: Script for generating data or models.
  - `ingest_transform_couchdb.py`: Script for data ingestion and transformation with CouchDB.
  - `ingest_transform.py`: Script for data ingestion and transformation.
  - `load.py`: Script for loading data or models.
  - `train.py`: Script for training the model.

- **data/**: Contains data files and model files.
  - **master/**: Directory containing the main dataset.
    - `telecom_customer_data.csv`: CSV file with customer data for analysis.
  - **pretrained_models/**: Contains pretrained model files.
    - `discriminator_model.h5`: Pretrained discriminator model.
    - `gan_model.h5`: Pretrained GAN model.
    - `generator_model.h5`: Pretrained generator model.
  - **saved_models/**: Contains saved model files after training.
    - `discriminator_model.h5`: Saved discriminator model.
    - `gan_model.h5`: Saved GAN model.
    - `generator_model.h5`: Saved generator model.

- `.gitattributes`: Git attributes file.
- `.gitignore`: Git ignore file for excluding files from version control.
- `readme.md`: This README file.
- `sample1.ipynb`, `sample2.ipynb`, `sample3.ipynb`: Jupyter Notebook samples for exploration and analysis.

---



---

### Data Definition

The synthetic dataset contains features that mirror real-world customer data, including:

- **Customer demographics** (e.g., age, gender, region)
- **Service subscription details** (e.g., internet, phone, streaming services)
- **Financial data** (e.g., monthly charges, contract type, total charges)
- **Tenure** (e.g., length of subscription)
- **Churn status** (target variable indicating whether a customer has churned)

---

### Program Flow

1. **Data Generation**: Generate synthetic churn data using a generative model (e.g., GANs, Variational Autoencoders) that mirrors real-world characteristics. [data_generation.py]
2. **Data Ingestion**: Load synthetic data into a database (MongoDB or SQLite) for further processing and analysis. [ingest_transform_mongodb.py, ingest_transform.py]
3. **Data Transformation**: Preprocess the synthetic data, including encoding categorical features, handling missing values, and normalizing numerical values. The data is then split into training, testing, and validation sets. [ingest_transform.py]
4. **Model Training**: Train deep learning models (e.g., using TensorFlow or PyTorch) to understand patterns in the synthetic data, with cross-validation and hyperparameter tuning. [train.py]
5. **Manual Prediction**: Enable manual input of synthetic customer data for churn prediction. [generate.py]
6. **Web Application**: A Streamlit app that allows interaction with the data generation pipeline and model, providing churn predictions based on synthetic data input. [app.py]

*Note*: The data generated also includes CustomerID. Therefore duplicate customerId would possible. Please Drop the customerId before using for data analysis.

---

### Steps to Run

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Launch Application**: Run `code/app.py` to start the Streamlit web application and access the entire pipeline via GUI. 

This setup will allow users to explore the impact of different factors on churn and fine-tune customer retention strategies using generated data.
