import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Ignore the 'squared' deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning)

# Your existing code here

# Load the dataset
data = pd.read_csv("Dataset - Factors Influencing Technology Adoption in Consumer Households.csv")

# Set indexing
data = data.reset_index(drop=True)

# Handle missing values: Consider imputation techniques instead of dropping rows
data = data.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['EducationLevel'] = label_encoder.fit_transform(data['EducationLevel'])

# Define features and target variable
X = data.drop(columns=['HouseholdIncome'])

# Feature engineering
# Education-Income Ratio
X['Education_Income_Ratio'] = data['HouseholdIncome'] / (data['EducationLevel'] + 1)  # Adding 1 to avoid division by zero

# Education Level Categories
# Considering 5 categories: High school or less, Some college, Bachelor's degree, Master's degree, Doctorate
education_categories = ['High School or Less', 'Some College', "Bachelor's Degree", "Master's Degree", 'Doctorate']
X['Education_Level_Category'] = pd.cut(data['EducationLevel'], bins=[-1, 0, 1, 2, 3, 4], labels=education_categories, right=False)

# Education-Income Gradient
education_income_gradient = LinearRegression()
education_income_gradient.fit(data[['EducationLevel']], data['HouseholdIncome'])
X['Education_Income_Gradient'] = education_income_gradient.coef_[0] * data['EducationLevel']

# Split the dataset into training and testing sets
y = data['HouseholdIncome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode Education_Level_Category
X_train_encoded = pd.get_dummies(X_train, columns=['Education_Level_Category'])
X_test_encoded = pd.get_dummies(X_test, columns=['Education_Level_Category'])

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor()
}

# Define parameter grids for GridSearchCV
param_grids = {
    'Linear Regression': {},
    'Ridge Regression': {'alpha': [0.1, 1, 10]},
    'Decision Tree': {'max_depth': [3, 5, 7, 9]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]},
    'SVR': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]},
    'XGBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]}
}

# Perform k-fold cross-validation and hyperparameter tuning
best_models = {}
best_rmse = float('inf')
best_r2 = -float('inf')
best_model_name_rmse = None
best_model_name_r2 = None

for name, model in models.items():
    print(f"Training and tuning {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")

    # Evaluate the model using RMSE and R-squared
    y_train_pred = grid_search.predict(X_train_scaled)
    y_test_pred = grid_search.predict(X_test_scaled)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"RMSE for {name} (train): {rmse_train}")
    print(f"RMSE for {name} (test): {rmse_test}")
    print(f"R-squared for {name} (train): {r2_train}")
    print(f"R-squared for {name} (test): {r2_test}")

    # Update best models based on RMSE and R-squared
    if rmse_test < best_rmse:
        best_rmse = rmse_test
        best_model_name_rmse = name
    if r2_test > best_r2:
        best_r2 = r2_test
        best_model_name_r2 = name

    print()

# Print the best model based on RMSE and R-squared
print(f"The best model based on RMSE is: {best_model_name_rmse}, with RMSE: {best_rmse}")
print(f"The best model based on R-squared is: {best_model_name_r2}, with R-squared: {best_r2}")

# Plot feature importances from ensemble models ('Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVR', 'XGBoost')
best_model = best_models[best_model_name_r2]
if best_model_name_r2 in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVR', 'XGBoost']:
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\nTop 5 most important features:")
    print(importance_df.head())

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    print(importance_df)
    plt.barh(importance_df['Feature'][:5], importance_df['Importance'][:5], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 5 Most Important Features')
    plt.gca().invert_yaxis()
    plt.show()

    # Plot predicted vs. actual values for the best model
    y_pred = best_model.predict(X_test_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
    plt.xlabel('Actual HouseholdIncome')
    plt.ylabel('Predicted HouseholdIncome')
    plt.title(f"{best_model_name_r2}: Predicted vs. Actual HouseholdIncome")
    plt.show()

# Print the best model
print(f"The best model overall is: {best_model_name_r2}")

# Plot predicted vs. actual values for the best model
y_pred = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.xlabel('Actual HouseholdIncome')
plt.ylabel('Predicted HouseholdIncome')
plt.title(f"{best_model_name_r2}: Predicted vs. Actual HouseholdIncome")
plt.show()
