# Salary Prediction using Polynomial Regression

A supervised machine learning project for predicting employee salaries using Polynomial Regression. The project covers data preprocessing, exploratory data analysis, feature transformation, model training, and regression model evaluation.

---

## Overview

The objective of this project is to predict salary based on employee-related attributes such as:

* Age
* Gender
* Education Level
* Job Title
* Years of Experience

The target variable is `Salary`.

The project follows a structured machine learning workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Feature Scaling
   ↓
Polynomial Feature Transformation
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
```

---

## Dataset

The dataset initially contains 375 records and 6 features, including the target variable. The dataset contains both numerical and categorical attributes.

| Feature             | Description               |
| ------------------- | ------------------------- |
| Age                 | Age of the employee       |
| Gender              | Gender of the employee    |
| Education Level     | Educational qualification |
| Job Title           | Current job position      |
| Years of Experience | Professional experience   |
| Salary              | Target variable           |

After data cleaning, the working dataset contains 324 records.

---

## Data Preprocessing

The following data quality checks and preprocessing operations were performed:

### Missing Values

Missing values were identified using:

```python
df.isnull().sum()
```

The missing records were removed before further analysis.

### Duplicate Records

Duplicate records were identified and analyzed.

The dataset initially contained 49 duplicate records. These were removed using:

```python
df = df.drop_duplicates(keep="first")
```

The final dataset contains no duplicate records.

---

## Exploratory Data Analysis

Exploratory analysis was performed to understand the dataset and target variable before model development.

The project uses:

* Matplotlib
* Seaborn
* Pandas
* NumPy

Salary distribution was analyzed using a histogram with KDE:

```python
sns.histplot(
    data=df,
    x="Salary",
    bins=20,
    kde=True
)
```

This provides an understanding of the distribution and variation of salary values.

---

## Machine Learning Approach

### Polynomial Regression

The project uses Polynomial Regression to model nonlinear relationships between the input variables and salary.

Polynomial feature transformation is performed using Scikit-learn:

```python
from sklearn.preprocessing import PolynomialFeatures
```

The transformed features are then used with:

```python
from sklearn.linear_model import LinearRegression
```

This approach extends a basic linear regression model by introducing higher-order feature combinations.

---

## Feature Scaling

Feature scaling is incorporated using Scikit-learn's `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler
```

Scaling helps place numerical features on a comparable scale before applying the polynomial transformation and regression model.

---

## Model Development

The model development workflow uses:

```text
Train-Test Split
       ↓
Feature Scaling
       ↓
Polynomial Feature Generation
       ↓
Linear Regression
       ↓
Salary Prediction
```

The project uses `train_test_split` to divide the available data into training and testing sets.

---

## Model Evaluation

Since this is a regression problem, the model is evaluated using multiple regression metrics.

### Mean Absolute Error

Measures the average absolute difference between actual and predicted salary.

```text
MAE = Average |Actual - Predicted|
```

### Mean Squared Error

Measures the average squared prediction error and gives greater weight to larger errors.

```text
MSE = Average (Actual - Predicted)²
```

### Root Mean Squared Error

The square root of MSE, expressed in the same units as salary.

```text
RMSE = √MSE
```

### R² Score

Measures how much of the variation in the target variable is explained by the model.

The notebook uses:

```python
mean_absolute_error
mean_squared_error
r2_score
```

for model evaluation.

---

## Technology Stack

| Category               | Technologies                             |
| ---------------------- | ---------------------------------------- |
| Language               | Python                                   |
| Data Analysis          | Pandas, NumPy                            |
| Visualization          | Matplotlib, Seaborn                      |
| Machine Learning       | Scikit-learn                             |
| Regression             | Polynomial Regression, Linear Regression |
| Feature Transformation | PolynomialFeatures                       |
| Feature Scaling        | StandardScaler                           |
| Model Evaluation       | MAE, MSE, RMSE, R²                       |
| Environment            | Jupyter Notebook / Kaggle                |

---

## Project Structure

```text
Salary-Prediction-Polynomial-Regression/
│
├── Salary Prediction (Polynomial Regression).ipynb
├── Salary Data.csv
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AtifMazhar-01/Salary_Prediction.git
cd Salary-Prediction-Polynomial-Regression
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Salary Prediction (Polynomial Regression).ipynb
```

---

## Key Concepts Demonstrated

* Supervised Learning
* Regression
* Polynomial Regression
* Polynomial Feature Transformation
* Feature Scaling
* Train-Test Split
* Data Cleaning
* Missing Value Handling
* Duplicate Detection
* Exploratory Data Analysis
* Data Visualization
* Model Evaluation
* MAE
* MSE
* RMSE
* R² Score

---

## Results

The model is evaluated using:

| Metric | Purpose                                    |
| ------ | ------------------------------------------ |
| MAE    | Measures average absolute prediction error |
| MSE    | Penalizes larger prediction errors         |
| RMSE   | Measures prediction error in target units  |
| R²     | Measures variance explained by the model   |

The notebook contains the final calculated evaluation metrics.

---

## Learning Outcomes

This project provided practical experience with the complete workflow of a regression problem:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
EDA
     ↓
Feature Engineering
     ↓
Feature Scaling
     ↓
Polynomial Transformation
     ↓
Model Training
     ↓
Prediction
     ↓
Evaluation
```

The project demonstrates how Polynomial Regression can be used to capture nonlinear relationships while retaining the simplicity of a linear regression estimator.

---

## Author

Atif Mazhar

Computer Engineering | Machine Learning | Data Science | MLOps

Pune, India

---

If you found this project useful, consider starring the repository.
