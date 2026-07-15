# Heart Disease Risk Prediction

An end-to-end machine learning project that predicts a patient's risk of heart disease from clinical measurements, served through a Flask web app.

## Overview

The pipeline ingests raw patient data, transforms it, trains and compares multiple classifiers, and serves predictions via a simple web form.

- **Data ingestion** (`src/components/data_ingestion.py`) — loads `data/raw/heart.csv` and splits it into train/test sets.
- **Data transformation** (`src/components/data_transformation.py`) — imputes missing values, scales numerical features, and one-hot encodes categorical features via a `ColumnTransformer`.
- **Model training** (`src/components/model_trainer.py`) — trains and tunes Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, XGBoost, CatBoost, and AdaBoost classifiers, then selects the best model by ROC-AUC score.
- **Prediction pipeline** (`src/pipeline/predict_pipeline.py`) — loads the saved model and preprocessor to score new patient data.
- **Web app** (`application.py`) — Flask app with a form (`templates/home.html`) for entering patient data and viewing the predicted risk.

## Input Features

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | M / F |
| ChestPainType | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar (0/1) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y / N |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, Down |

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/components/data_ingestion.py
```

This runs ingestion → transformation → training, and saves `data/preprocessor.pkl`, `data/model.pkl`, and `data/model_results.csv` (a comparison of all candidate models).

### Run the web app

```bash
python application.py
```

Then open `http://localhost:5000` and use the form to get a prediction.

## Project Structure

```
├── application.py              # Flask app entry point
├── data/                       # Raw/processed data and trained artifacts
├── src/
│   ├── components/              # Ingestion, transformation, training
│   ├── pipeline/                 # Train and predict pipelines
│   ├── exception.py              # Custom exception handling
│   ├── logger.py                 # Logging setup
│   └── utils.py                  # Shared helpers (save/load objects, evaluation)
├── templates/                  # HTML templates for the web UI
└── requirements.txt
```
