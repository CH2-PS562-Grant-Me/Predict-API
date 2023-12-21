# Predict-API
Repository for Model Machine Learning API
# API for Recommendation System : Grant Me App
This is a repository for hosting Machine Learning model APIs that will be deployed and can be used in applications.
## Overview
By using this API, it will bring up any recommended scholarships based on the inputted data. The scholarships are taken from our scholarship database which will be displayed in our application.

## Table of Contents

- [Setup](#setup)
  - [Requirements](#requirements)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Database](#database)
- [Model](#model)
- [Endpoints](#endpoints)
- [License](#license)

## Setup

### Requirements
To install all the libraries needed in this model, use the syntax below:
```bash
pip install -r requirements.txt
```
You also need to provide *MySQL Server* or your database server
### Environment Variables
```bash
# Clone the repository
git clone https://github.com/CH2-PS562-Grant-Me/Predict-API.git

# Change directory
cd PredictAPI # change with your folder name

# Install dependencies
pip install -r requirements.txt
```
Create a `.env` file with the following variables:

```plaintext
DB_HOST=your_database_host
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=your_database_name
```
## Usage

There are a few things that need to be done to use this API:
1. Run the Flask API
   ```bash
   python app.py
    ```
   The API will be accessible at `http://127.0.0.1:5000`.
2. Make a POST request to `http://127.0.0.1:5000/predict` with input data in JSON format:
   ```plaintext
   {
    "IPK": 3.8,
    "Sertifikasi": 5,
    "SertifikasiProfesional": 3,
    "prestasiNasional": 2,
    "lombaNasional": 1,
    "prestasiInternasional": 1,
    "lombaInternasional": 1,
    "internMagang": 14,
    "Kepanitiaan": 13
    }
   ```
The API will return a *prediction*, *accuracy percentage*, and *list of related scholarship information*.

## Database
The API connects to a MySQL database to retrieve scholarship information. Make sure to set up the database connection variables in the `.env` file.

## Model
The machine learning model `(model_V2.h5)` is loaded during API initialization and used for predictions.

## Endpoints
- **Endpoint**: /predict
- **Method**: POST
- **Input**: JSON data with features for prediction
- **Output**: *prediction*, *accuracy percentage*, and *list of related scholarship information*.

## License
Copyright © 2023 Pakintaki Group . All rights reserved.



