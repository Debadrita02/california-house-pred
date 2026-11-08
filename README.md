# California Housing Price Prediction

## Project Overview

This project uses Machine Learning techniques to predict median house values in California based on housing and geographical features.

It also uses a Split Conformal Prediction approach to generate prediction intervals, providing an estimated range for the predicted house value rather than only a single prediction.

A Streamlit web application allows users to enter housing details and view the predicted value and its prediction interval.

## Dataset

- Dataset: California Housing Dataset
- Source: Scikit-learn
- Total records: 20,640
- Number of input features: 8
- Target variable: `MedHouseVal`

### Input Features

1. MedInc – Median income in the area
2. HouseAge – Median house age
3. AveRooms – Average number of rooms
4. AveBedrms – Average number of bedrooms
5. Population – Population of the area
6. AveOccup – Average number of household members per household
7. Latitude – Geographic latitude
8. Longitude – Geographic longitude

The target variable, `MedHouseVal`, represents median house value in units of $100,000. The target values are capped at $500,000 in the original dataset.

## Data Preprocessing

- Explored the dataset and its statistical properties.
- Examined missing values and feature distributions.
- Separated input features and the target variable.
- Divided the data into training, calibration, and testing sets.
- Used the calibration dataset separately for calculating prediction intervals.

### Dataset Split

| Dataset | Records |
|---|---:|
| Training | 13,209 |
| Calibration | 3,303 |
| Testing | 4,128 |
| Total | 20,640 |

## Machine Learning Models

Three regression algorithms were implemented and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The Random Forest Regressor was selected as the final model used in the web application.

## Model Evaluation

The regression models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help evaluate prediction errors and how well the models explain variations in the target variable.

## Split Conformal Prediction

Split Conformal Prediction was implemented to estimate prediction intervals.

- Calibration samples: 3,303
- Target coverage level: 90%
- Observed test coverage: 90.12%
- Average prediction interval width: 1.626 (in units of $100,000)

The prediction interval provides a range around the estimated house value.

**Note:** The observed test coverage is an evaluation result, not a guarantee that every individual prediction will be correct. The dataset also represents historical census information from 1990, not current housing prices.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
california-housing-prediction/
│
├── notebooks/
│   └── california housing prediction.ipynb
│
├── app.py
├── california_housing_model.pkl
├── calibration_threshold.pkl
└── README.md
```

## How to Run the Project

### 1. Install Required Libraries

```bash
pip install streamlit scikit-learn numpy pandas matplotlib seaborn joblib
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

### 3. Open the Application

Open the local URL displayed in the terminal to access the application.

## Application Features

- Enter housing and geographical information.
- Predict median house value.
- Display an estimated prediction interval.
- View the model's prediction range.
- Understand the limitations of the prediction.

## Limitations

- The dataset is based on 1990 census information.
- The target values are capped at $500,000.
- Predictions are estimates and should not be treated as current market valuations.
- Prediction intervals depend on the calibration data and underlying statistical assumptions.

## Author

Machine Learning Internship Project