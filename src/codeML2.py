#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:00:08 2025


"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:39:53 2025


"""


# === 1. IMPORT LIBRARIES ===


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


# File access path
#file_path = "~/Desktop/Renewable_energy_ML"

# Use '..' to go up one level from 'src', then into 'data'
df = pd.read_csv('../Data/RE.csv')

# Cleaning column names
df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('  ', ' ')

# Drop geographical and columns no intersting
columns_to_drop = ['Entity','Year', 'Latitude', 'Longitude', 'Land Area(Km2)', 'Density\\n(P/Km2)']
#columns_to_drop = ['Entity','Year', 'Latitude', 'Longitude',]
df_clean = df.drop(columns=columns_to_drop)

# Convert all to numeric to work in data (kinda better)
df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

# Drop rows where the target is missing
df_clean = df_clean.dropna(subset=["Renewable energy share in the total final energy consumption (%)"])

#  correlation matrix for numeric variables
plt.figure(figsize=(14, 12))
corr_matrix = df_clean.corr()

sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

#Based on correlation matrix we drop correlated features
columns_to_drop = ['Access to clean fuels for cooking','Electricity from fossil fuels (TWh)']
df_clean = df_clean.drop(columns=columns_to_drop)

# show correlations with the target variable which is your Y (renawble energy)
target_corr = corr_matrix["Renewable energy share in the total final energy consumption (%)"].sort_values(ascending=False)
print("\nCorrelation of features with the target:")
print(target_corr)


# Define target
target = "Renewable energy share in the total final energy consumption (%)"
X = df_clean.drop(columns=[target])
y = df_clean[target]

#KNN for missing values, K=3 to represent local tendencies

scaler = StandardScaler()
X_scaled_temp = scaler.fit_transform(X)  # temporary for imputation

imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(X_scaled_temp)

X_scaled = StandardScaler().fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


X = pd.DataFrame(X_imputed, columns=X.columns)

# Mixed feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression


X_temp = pd.DataFrame(X)
lr_fs = LinearRegression()

#Apply mixed stepwise
sfs = SFS(lr_fs,
          k_features='best',
          forward=True,
          floating=True,
          scoring='r2',
          cv=5)

sfs = sfs.fit(X_temp, y)
# selected features
selected_features = list(sfs.k_feature_names_)
print("Selected features after mixed stepwise selection:", selected_features)

# Reduce X to selected features
X = X_temp[selected_features]
pd.set_option('display.max_columns', None)
print("X: ")
print(X.head())

# Impute missing values using KNN (better than mean)
from sklearn.impute import KNNImputer


# Final train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize models
lr = LinearRegression()
lasso = LassoCV(cv=5, random_state=42)
rf = RandomForestRegressor(random_state=42)

# Fit models
lr.fit(X_train, y_train)
lasso.fit(X_train, y_train)
rf.fit(X_train, y_train)

#t-test, summary of the model
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train.reset_index(drop=True))
y_train_reset = y_train.reset_index(drop=True)
ols_model = sm.OLS(y_train_reset, X_train_const).fit()
print(ols_model.summary())


# Predict
y_pred_lr = lr.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluate
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Lasso Regression R²:", r2_score(y_test, y_pred_lasso))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Lasso RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))


#ridge regression

from sklearn.linear_model import RidgeCV

# Ridge Regression with cross-validation for alpha
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5)
ridge.fit(X_train, y_train)

# Predict
y_pred_ridge = ridge.predict(X_test)

# print
print("Ridge Regression R²:", r2_score(y_test, y_pred_ridge))
print("Ridge Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))



## K-Fold Cross validation (for linear, Rodge and Lasso)


# for linear regression

from sklearn.model_selection import cross_val_score

# Cross-validation for Linear Regression
cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Linear Regression CV MSE:", -np.mean(cv_scores_lr))

#for ridge and lasso

# Ridge
cv_scores_ridge = cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Ridge Regression CV MSE:", -np.mean(cv_scores_ridge))

# Lasso
cv_scores_lasso = cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Lasso Regression CV MSE:", -np.mean(cv_scores_lasso))


import matplotlib.pyplot as plt

models = ['Linear', 'Ridge', 'Lasso', 'Random Forest']
rmse_scores = [
    np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    np.sqrt(mean_squared_error(y_test, y_pred_rf))
]


plt.bar(models, rmse_scores)
plt.title('Comparison of RMSE by Model')
plt.ylabel('RMSE')
plt.show()

## just to print the result in a kinda of tab 

import pandas as pd

# Create a dataframe for test set performance
test_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest'],
    'R²': [0.7823, 0.7823, 0.7820, 0.9666],
    'RMSE': [13.54, 13.54, 13.55, 5.30]
})

# Display
print(test_results)



# Display
print(test_results)


## try the robustness of the model with boostrap 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Settings
n_iterations = 1000  # number of bootstrap samples
n_size = int(len(X_train) * 0.8)  # size of each resampled dataset (80% of train set)

# Initialize storage for coefficients
coefs = np.zeros((n_iterations, X_train.shape[1]))

# Bootstrap Loop
for i in range(n_iterations):
    # Resample X_train and y_train
    X_resampled, y_resampled = resample(X_train, y_train, n_samples=n_size, random_state=i)
    
    # Fit Linear Regression on resampled data
    model = LinearRegression()
    model.fit(X_resampled, y_resampled)
    
    # Store coefficients
    coefs[i, :] = model.coef_

# Calculate statistics
coef_means = np.mean(coefs, axis=0)
coef_std = np.std(coefs, axis=0)

# Create a tab for results
bootstrap_results = pd.DataFrame({
    'Feature': X_train.columns,
    'Mean Coefficient': coef_means,
    'Std Deviation': coef_std
})

# Display results
print(bootstrap_results)



## from the robustness to print the resultat for gdp per capita 


import matplotlib.pyplot as plt
'''
feature_name = "gdp_per_capita"
feature_idx = list(X.columns).index(feature_name)

# Plot histogram of the bootstrapped coefficients for this feature
plt.figure(figsize=(8,6))
plt.hist(coefs[:, feature_idx], bins=30, edgecolor='black')
plt.title(f'Bootstrap Distribution of Coefficient: {feature_name}')
plt.xlabel('Coefficient Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
'''






















