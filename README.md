## Car Price Prediction Web System 

This project is a machine learning–based web system that predicts the selling price of a used car based on user-provided details such as year, kilometers driven, fuel type, transmission, and ownership history. 
The application runs locally on a laptop and provides real-time predictions through a web interface.


## Project Overview

The goal of this project is to demonstrate an end-to-end machine learning
workflow, starting from dataset selection and preprocessing to model training
and real-time inference using a web-based UI.

The focus is on:
- Applying Python, Pandas, and NumPy for data handling
- Training a regression model using real-world data
- Converting a console-based ML workflow into a usable web application


## Features

- User-friendly web interface for entering car details
- Real-time prediction of selling price
- Machine learning model trained on used car data
- Runs completely on a local machine
- Can be exposed via a temporary HTTPS link for demos


## Dataset  :  The project uses a publicly available used-car dataset sourced from Kaggle.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit


## Input Parameters

The application takes the following inputs from the user:

- Manufacturing year
- Kilometers driven
- Fuel type (Petrol / Diesel / CNG)
- Transmission type (Manual / Automatic)
- Number of previous owners

Based on these inputs, the model predicts the estimated selling price.



## Working

- The dataset is loaded and cleaned using Pandas
- Categorical features are encoded into numeric values
- A regression model is trained using scikit-learn
- The trained model is saved and reused for real-time predictions
- This version uses a baseline linear regression model and a limited feature set
