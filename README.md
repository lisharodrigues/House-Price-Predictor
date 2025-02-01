# Smart House Price Prediction using Machine Learning  

## Introduction  
Finding the right house at the right price can be a daunting task, involving extensive research, negotiations, and the risk of fraud. This project presents a **Machine Learning-based House Price Prediction Model** trained on real-world housing data to estimate property prices accurately.  

## Dataset Overview  
The dataset used for training consists of **13 key features** affecting house prices:  

| Feature       | Description  |
|--------------|-------------|
| **Id**       | Record identifier |
| **MSSubClass** | Type of dwelling involved in the sale |
| **MSZoning** | General zoning classification of the sale |
| **LotArea** | Lot size in square feet |
| **LotConfig** | Configuration of the lot |
| **BldgType** | Type of dwelling |
| **OverallCond** | Rates the overall condition of the house |
| **YearBuilt** | Original construction year |
| **YearRemodAdd** | Remodel date (same as construction date if no remodeling) |
| **Exterior1st** | Exterior covering on house |
| **BsmtFinSF2** | Type 2 finished square feet |
| **TotalBsmtSF** | Total square feet of basement area |
| **SalePrice** | Target variable (to be predicted) |

## Technologies & Libraries Used  
- **Pandas** – Data handling  
- **Matplotlib** – Data visualization  
- **Seaborn** – Heatmap and feature correlation analysis  
- **Scikit-learn** – Machine Learning models  

---

## Data Preprocessing & Exploration  

### 1. Data Cleaning  
- Removed irrelevant columns (e.g., `Id`).  
- Handled missing values by filling `SalePrice` with the mean.  
- Dropped records with excessive missing data.  

### 2. Exploratory Data Analysis (EDA)  
- Used **heatmaps** to analyze feature correlations.  
- Plotted **bar charts** to understand categorical feature distributions.  

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
```

---

## Feature Engineering & Encoding  

Since some features are categorical, they are converted into numerical values using **OneHotEncoding**:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = pd.DataFrame(encoder.fit_transform(dataset[object_cols]))
dataset = pd.concat([dataset.drop(object_cols, axis=1), encoded_features], axis=1)
```

---

## Model Selection & Training  

### Train-Test Split  
```python
from sklearn.model_selection import train_test_split

X = dataset.drop(['SalePrice'], axis=1)
y = dataset['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Implemented Models & Performance  

| Model | Error Metric (MAPE) |
|--------------|----------------|
| **Support Vector Machine (SVM)** | **0.187** |
| **Random Forest Regressor** | 0.192 |
| **Linear Regression** | 0.187 |
| **CatBoost Regressor** | **R² Score: 0.89** |

#### Support Vector Machine (SVM) Implementation
```python
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error

model_svr = SVR()
model_svr.fit(X_train, y_train)
y_pred = model_svr.predict(X_valid)

print(mean_absolute_percentage_error(y_valid, y_pred))
```

#### Random Forest Regressor Implementation
```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=10)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_valid)

print(mean_absolute_percentage_error(y_valid, y_pred))
```

#### Linear Regression Implementation
```python
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_valid)

print(mean_absolute_percentage_error(y_valid, y_pred))
```

#### CatBoost Regressor Implementation
```python
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

cb_model = CatBoostRegressor()
cb_model.fit(X_train, y_train)
preds = cb_model.predict(X_valid)

r2 = r2_score(y_valid, preds)
print("R² Score:", r2)
```

---

## Conclusion  

Among the tested models, **SVM and CatBoost Regressor performed the best**, with **low mean absolute percentage error** and high accuracy.  
Future improvements could include **ensemble learning** (Bagging & Boosting) for better predictions.  




