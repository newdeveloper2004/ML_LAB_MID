# ML Models Project

A small end-to-end ML project built on the Iris dataset (`input/data.csv`):

- **Models:** a linear regression model (predicts `petal_length`, reports R²) and a random forest classifier (predicts `species`, reports accuracy), both saved with joblib in `models/`.
- **Backend:** FastAPI app with two POST endpoints, `/predict/regression` and `/predict/classification`.
- **Frontend:** one HTML page with a form for each model.

## Project structure

    input/data.csv          dataset
    src/train.py            trains and saves both models
    src/predict.py          loads models and runs a sample prediction
    src/model_selection.py  compares candidate models
    src/tune_model.py       grid search for the classifier
    src/utils.py            data loading and train/test splits
    src/app.py              FastAPI backend
    src/index.html          frontend page
    models/                 model1.pkl (regression), model2.pkl (classifier)
    notebooks/              data exploration

## How to run

All commands are run from the project root.

1. Install dependencies:

       pip install -r requirements.txt

2. (Optional) Retrain the models. This prints R² and accuracy and overwrites `models/*.pkl`:

       python src/train.py

3. Start the backend:

       python src/app.py

4. Open http://127.0.0.1:8000 in your browser and use the two forms.

## API

**POST `/predict/regression`**

    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_width": 0.2}
    -> {"result": 1.492}

**POST `/predict/classification`**

    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    -> {"result": "setosa"}

Interactive API docs are available at http://127.0.0.1:8000/docs.

## Results

On the 20% test split: regression R² ≈ 0.96, classification accuracy = 1.0.
