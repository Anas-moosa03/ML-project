# ML Project
# Mobile Price Classification & Vehicle Accidents Prediction

## Overview
This project involves two separate analyses:
1. **Mobile Price Classification**: Predicting mobile phone price categories using machine learning algorithms.
2. **Vehicle Accidents Time Series Prediction**: Forecasting vehicle accidents based on historical data.

## Datasets

### 1. Mobile Price Classification
- **Source**: [Kaggle - Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- **Features**: Attributes of mobile phones (e.g., battery power, clock speed, etc.).
- **Target Variable**: `price_range` - Categorical variable representing price categories.

### 2. Vehicle Accidents Time Series
- **Source**: [Kaggle - Vehicle Accidents](https://www.kaggle.com/datasets/ddosad/vehicle-accidents)
- **Features**: Time series data of vehicle accidents.
- **Target**: Predict future accidents based on historical data.

## Project Structure

### Mobile Price Classification

#### Data Pre-processing
- **Splitting Data**: Using `train_test_split` for training and testing sets.
- **Feature Engineering**: 
  - Correlation analysis with heatmap visualization.
  - Feature importance selection using Random Forest.

#### Models Applied
- **Random Forest Classifier**: 
  - Feature importance visualization.
  - Hyperparameter tuning with GridSearchCV.

- **Multi-layer Perceptron (MLP)**:
  - Extensive hyperparameter tuning via GridSearchCV.
  - Performance evaluation with accuracy, classification reports, and visualizations.

- **Support Vector Machine (SVM)**:
  - Basic model followed by parameter optimization using Optuna.

#### Analysis & Visualization
- Feature importance and correlation plots.
- Model performance metrics and visualizations.

### Vehicle Accidents Time Series Prediction

#### Data Pre-processing
- **Normalization**: Using `MinMaxScaler` for scaling accident data.
- **Data Splitting**: Into training and testing sets for time series data.

#### Models Applied
- **Echo State Network (ESN)**:
  - Custom implementation for time series prediction.

- **LSTM (Long Short-Term Memory)**:
  - Neural network model for time series forecasting with Keras.

- **Bidirectional LSTM**:
  - Enhanced LSTM for capturing patterns from both past and future contexts.

#### Analysis & Visualization
- Error analysis with plots.
- Residuals and actual vs. predicted value comparisons.

### Usage
To run the code, ensure you have the following libraries installed:
- `pandas`, `numpy`, `matplotlib`, `seaborn` for data manipulation and visualization
- `scikit-learn` for traditional ML algorithms
- `optuna` for optimization
- `keras` for neural network models

```bash
pip install pandas numpy matplotlib seaborn scikit-learn optuna keras
