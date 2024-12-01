
# # Module 1

# ### TODO:
# 1. Load the Ames housing dataset.
# 2. Perform Exploratory Data Analysis and use descriptive statistics to describe the data.
# 3. Prepare the dataset for modeling by imputing missing values with the variable's mean value or any other value that you prefer.
# 4. Use the "cor()" function to produce a correlation matrix of the numeric values.
# 5. Produce a plot of the correlation matrix, and explain how to interpret it. (hint - check the corrplot or ggcorrplot plot libraries)
# 6. Make a scatter plot for the X continuous variable with the highest correlation with SalePrice. Do the same for the X variable that has the lowest correlation with SalePrice. Finally, make a scatter plot between X and SalePrice with the correlation closest to 0.5. Interpret the scatter plots and describe how the patterns differ.
# 7. Using at least 3 continuous variables, fit a regression model in R.
# 8. Report the model in equation form and interpret each coefficient of the model in the context of this problem.
# 9. Use the "plot()" function to plot your regression model. Interpret the four graphs that are produced.
# 10. Check your model for multicollinearity and report your findings. What steps would you take to correct multicollinearity if it exists?
# 11. Check your model for outliers and report your findings. Should these observations be removed from the model?
# 12. Attempt to correct any issues that you have discovered in your model. Did your changes improve the model, why or why not?
# 13. Use the all subsets regression method to identify the "best" model. State the preferred model in equation form.
# 14. Compare the preferred model from step 13 with your model from step 12. How do they differ? Which model do you prefer and why?


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score


#  Load the data and clean it up

raw_data = pd.read_csv("AmesHousing.csv")


print(raw_data.head())


print(raw_data.describe())


print(raw_data.info())


# Define ordinal mappings based on the documentation for features like Overall Qual, Exter Qual, etc.
ordinal_mappings = {
    'Overall Qual': {'Very Excellent': 10, 'Excellent': 9, 'Very Good': 8, 'Good': 7, 
                     'Above Average': 6, 'Average': 5, 'Below Average': 4, 
                     'Fair': 3, 'Poor': 2, 'Very Poor': 1},
    'Exter Qual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Exter Cond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Bsmt Qual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'Bsmt Cond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Garage Qual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'Garage Cond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'Pool QC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0},
    'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
}

# Apply ordinal mappings to columns if they exist in the dataset
for col, mapping in ordinal_mappings.items():
    if col in raw_data.columns:
        raw_data[col] = raw_data[col].map(mapping)


# Define the feature sets again, checking if they exist in the dataframe
numeric_features = [col for col in ['Gr Liv Area', 'Year Built', 'Total Bsmt SF', 
                                    'Garage Area', 'Full Bath', 'Lot Area', 'SalePrice'] if col in raw_data.columns]

categorical_features = [
    'Overall Qual','MS SubClass', 'MS Zoning', 'Street', 'Lot Shape', 'Land Contour', 
    'Utilities', 'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1', 
    'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Exterior 1st', 
    'Foundation', 'Heating', 'Central Air', 'Garage Type', 'Sale Type', 
    'Sale Condition'
]

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
raw_data[numeric_features] = imputer.fit_transform(raw_data[numeric_features])

# Create dummy variables for categorical features, only if they exist in df
categorical_features = [col for col in categorical_features if col in raw_data.columns]
df_encoded = pd.get_dummies(raw_data, columns=categorical_features, drop_first=True)

# Drop columns with high NAs ~>100 from 2900 records and the PID column if present
if 'PID' in df_encoded.columns:
    df_encoded.drop(columns=["PID"], inplace=True)
null_counts = df_encoded.isnull().sum()
null_cols = null_counts[null_counts > 100]
df_encoded.drop(columns=null_cols.index, inplace=True)


df_encoded.dropna(inplace=True)


correlation_matrix = raw_data[numeric_features].corr()
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Identify variables with highest, lowest, and moderate correlation
corr_with_saleprice = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)
print(corr_with_saleprice)
highest_corr = corr_with_saleprice.index[1]  # Excluding SalePrice itself
lowest_corr = corr_with_saleprice.index[-1]
moderate_corr = corr_with_saleprice.iloc[(corr_with_saleprice - 0.5).abs().argsort()[0]]

# Create scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.scatterplot(data=df_encoded, x=highest_corr, y='SalePrice', ax=axes[0])
sns.scatterplot(data=df_encoded, x=lowest_corr, y='SalePrice', ax=axes[1])
sns.scatterplot(data=df_encoded, x=moderate_corr, y='SalePrice', ax=axes[2])
plt.tight_layout()
plt.show()


X = df_encoded[['Gr Liv Area', 'Total Bsmt SF', 'Garage Area']]
y = df_encoded['SalePrice']

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R-squared:", model.score(X, y))


y_pred = model.predict(X)
residuals = y - y_pred

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(y_pred, residuals)
axes[0, 0].set_xlabel('Predicted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

axes[0, 1].hist(residuals, bins=30)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_title('Histogram of Residuals')

sm.qqplot(residuals, line='s', ax=axes[1, 0])  # Plot on existing axis
axes[1, 0].set_title('Normal Q-Q Plot')
axes[1, 0].set_xlabel('Theoretical Quantiles')
axes[1, 0].set_ylabel('Standardized Residuals')

axes[1, 1].scatter(y_pred, np.abs(residuals))
axes[1, 1].set_xlabel('Predicted values')
axes[1, 1].set_ylabel('|Residuals|')
axes[1, 1].set_title('Scale-Location')

plt.tight_layout()
plt.show()


vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


Q1 = df_encoded['SalePrice'].quantile(0.25)
Q3 = df_encoded['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_encoded[(df_encoded['SalePrice'] < lower_bound) | (df_encoded['SalePrice'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")


y_log = np.log(y)
model_improved = Ridge(alpha=1.0)
model_improved.fit(X, y_log)

print("Improved model R-squared:", model_improved.score(X, y_log))


df_cleaned = df_encoded[(df_encoded['SalePrice'] >= lower_bound) & (df_encoded['SalePrice'] <= upper_bound)]
X = df_cleaned[['Gr Liv Area', 'Total Bsmt SF', 'Garage Area']]
y = df_cleaned['SalePrice']

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R-squared:", model.score(X, y))


rfe = RFE(estimator=LinearRegression(), n_features_to_select=8)
rfe.fit(df_encoded[numeric_features].drop('SalePrice', axis=1), y_log)

selected_features = df_encoded[numeric_features].drop('SalePrice', axis=1).columns[rfe.support_]
print("Selected features:", selected_features)

X_best = df_encoded[selected_features]
model_best = LinearRegression()
model_best.fit(X_best, y_log)

y_pred = model_best.predict(X_best)

r2 = r2_score(y_log, y_pred)


n = X.shape[0]  
k = X.shape[1]  

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f'Best model R-squared: {r2}')
print(f'Adjusted R²: {adjusted_r2}')


