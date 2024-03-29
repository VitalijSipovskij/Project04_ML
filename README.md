# Machine Learning Project

This Python script analyzes a dataset on technology adoption in households, focusing on predicting household income 
using regression models. It preprocesses the data, selects and tunes seven regression models, evaluates their 
performance using RMSE and R-squared, visualizes feature importance for ensemble models, and plots predicted versus 
actual household income values. The script aims to identify the best-performing regression model and provide insights 
into factors influencing household income.

## INSTALLATION

### Requirements

- Python 3.12.0
- Requirements listed in [requirements.txt](requirements.txt)

## Usage
1. Clone the repository download the repository to your local machine to access the scripts.
2. Install dependencies in terminal using `pip install -r requirements.txt`. 
3. Run `Project04_main.py` script by running it in a Python environment. 
4. Ensure that the necessary libraries are installed. 
5. Open `Project04_ML.ipynb` script file and execute it using integrated development environment (IDE)
like Jupyter Notebook or Colaboratory. And upload Dataset into that platform so that it could initiate properly. !Notice
uploaded dataset into platform will be available only for 24 hours after that you will need to upload it again into that
platform.

## ABOUT THE CODE IN GENERAL

The provided code is a comprehensive machine learning pipeline designed to address a regression problem using various 
regression algorithms such as Linear Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, 
Support Vector Regression (SVR), and XGBoost. The dataset "Dataset - Factors Influencing Technology Adoption in Consumer
Households.csv" is loaded and preprocessed, including handling missing values, encoding categorical variables, and 
feature engineering. Features are standardized, and the dataset is split into training and testing sets. GridSearchCV 
is used for hyperparameter tuning and model selection for each algorithm, with evaluation metrics such as root mean 
squared error (RMSE) and R-squared. The best-performing model, based on RMSE and R-squared, is selected and visualized 
for feature importance and predicted vs. actual household income. Overall, the code demonstrates a systematic approach 
to building and evaluating regression models for predicting household income based on various factors influencing 
technology adoption.

## ABOUT CODE PARAMETERS

The parameters found as the best were determined through a systematic process of hyperparameter tuning and model 
evaluation. The choice of parameters was influenced by various factors related to the dataset and the nature of the 
regression problem. For instance, in Ridge Regression, the choice of alpha parameter reflects the trade-off between 
model complexity and regularization. Higher values of alpha may have been chosen to penalize large coefficients, thereby
preventing overfitting, especially if the dataset had high dimensionality or multicollinearity among features. 
Similarly, in ensemble models like Gradient Boosting and Random Forest, the number of estimators (trees) and the 
learning rate were tuned to strike a balance between model complexity and generalization.

Moreover, the choice of parameters may also have been influenced by the distribution and shape of the data, as well as 
the presence of clear decision boundaries. For example, in Decision Tree models, the maximum depth parameter controls 
the complexity of the tree structure, which may have been adjusted based on the complexity of the underlying 
relationships in the data. Additionally, the selection of kernel and regularization parameter (C) in SVR might have been
influenced by the linearity or non-linearity of the data distribution, with different kernels capturing different 
decision boundary shapes.

Overall, the parameters chosen as the best reflect a careful consideration of the trade-offs between model complexity, 
regularization, and generalization, tailored to the specific characteristics and requirements of the dataset and 
regression problem at hand.

---
To change the parameters and test their effect on model performance, modify the values within the param_grids 
dictionary. Each key in the dictionary corresponds to a regression model, and the associated value is another 
dictionary containing the parameters and their possible values for that model. 

`# Define parameter grids for GridSearchCV
param_grids = {
    'Linear Regression': {},
    'Ridge Regression': {'alpha': [0.1, 1, 10]},
    'Decision Tree': {'max_depth': [3, 5, 7, 9]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]},
    'SVR': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]},
    'XGBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]}
}
`

For example, to adjust the parameters for Ridge Regression, change the values within the dictionary associated with the 
key 'Ridge Regression'. The 'alpha' parameter can be set to different values such as [0.01, 0.1, 1, 10] to see how it 
affects the model's performance.

Similarly, parameters for other models like Decision Tree, Random Forest, Gradient Boosting, SVR, and XGBoost can be 
modified by changing the values within their respective dictionaries. For instance, 'max_depth' for Decision Tree can 
be adjusted to different values like [3, 5, 7, 9, None] to explore its impact on the model's behavior.

After modifying the parameters, rerun the code to train the models with the new parameter settings and evaluate their 
performance to determine the optimal configuration for each model.