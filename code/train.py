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
from load import data_load
from ingest_transform import preprocess
from ingest_transform import store_model_path as smpsql
from ingest_transform_couchdb import store_model_path as smpcdb
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd

def build_generator(latent_dim, n_features):
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(n_features, activation='sigmoid'))
    return model

def build_discriminator(n_features):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_dim=n_features, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def train_gan(generator, discriminator, gan, real_data, epochs=10000, batch_size=32, latent_dim=10):
    for epoch in range(epochs):
        # 1. Select a random batch of real data
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_samples = real_data[idx]

        # 2. Generate a batch of synthetic data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_samples = generator.predict(noise)

        # 3. Create labels for the discriminator
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # 4. Train the discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 5. Train the generator (via the combined model)
        # Update noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # Set the target labels to `real_labels` (ones) to fool the discriminator
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} | D Loss: {d_loss[0]} | D Accuracy: {100*d_loss[1]:.2f} | G Loss: {g_loss}")

        return f'{100*d_loss[1]:.2f}'

def train_(df, epochs, database_choice):
    df, _ = data_load(df)
    latent_dim = 10  # Dimension of the latent space
    n_features = df.shape[1]  # Number of features in the dataset

    generator = build_generator(latent_dim, n_features)
    discriminator = build_discriminator(n_features)

    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    generated_sample = generator(gan_input)
    gan_output = discriminator(generated_sample)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    score = train_gan(generator, discriminator, gan, df, epochs)

    model_dir = 'data/saved_models'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the generator model
    generator.save(os.path.join(model_dir, 'generator_model.h5'))

    # Save the discriminator model
    discriminator.save(os.path.join(model_dir, 'discriminator_model.h5'))

    # Save the GAN model (if needed for continued training)
    gan.save(os.path.join(model_dir, 'gan_model.h5'))

    smpsql(os.path.join(model_dir, 'generator_model.h5'), os.path.join(model_dir, 'discriminator_model.h5'), os.path.join(model_dir, 'gan_model.h5')) if database_choice == "PostgreSQL" else smpcdb(os.path.join(model_dir, 'generator_model.h5'), os.path.join(model_dir, 'discriminator_model.h5'), os.path.join(model_dir, 'gan_model.h5'))

    return "Models trained and saved successfully!", score