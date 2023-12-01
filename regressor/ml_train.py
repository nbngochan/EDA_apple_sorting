import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from utils.metrics import print_evaluation_metric_regressor, model_performance_classifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


def train_evaluate_ml(csv_path):
    ml_models = {'decision_tree': None, 'gradient_boosting': None, 'knn': None,
            'logistic_regression': None, 'naive_bayes': None, 'random_forest': None, 'svm': None}
    
    df = pd.read_csv(csv_path)
    
    # Split data into train, test, validation
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    
    X = df['area'].values.reshape(-1, 1)
    y = df['weight']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio,
                                                        random_state=44, stratify=df['label'])

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio),
                                                    random_state=44, stratify=df['label'][y_test.index])
    
    # Train model
    for model in ML_OPTIONS:
        if model == 'decision_tree':
            regressor = DecisionTreeRegressor()
        elif model == 'gradient_boosting':
            regressor = GradientBoostingRegressor()
        elif model == 'knn':
            regressor = KNeighborsRegressor(weights='distance')
        elif model == 'logistic_regression':
            regressor = LogisticRegression()
        elif model == 'naive_bayes':
            regressor = GaussianNB()
        elif model == 'random_forest':
            regressor = RandomForestRegressor()
        elif model == 'svm':
            regressor = SVR()
        else:
            raise ValueError(f'Invalid model type: {model}')
        
        # Train 
        regressor.fit(X_train, y_train)
        ml_models[model] = regressor
        y_true = y_val
        y_pred = regressor.predict(X_val)
        
        # Evaluate the model
        mse = mean_squared_error(y_true, y_pred) 
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)

        # Print the evaluation metrics
        print(f'Model: {model}')
        print_evaluation_metric_regressor(mse, r2, mae, evs)
        print('--------------------------------------')
        
    return ml_models


def eval_model_regressor(target, pred):
    mse = mean_squared_error(target, pred) 
    r2 = r2_score(target, pred)
    mae = mean_absolute_error(target, pred)
    evs = explained_variance_score(target, pred)

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "MSE": mse,
            "MAE": mae,
            "R-squared": r2,
            "Ex. var score": evs,
        },
        index=[0],
    )

    return df_perf


def ml_test_infer(csv_path, model):
    print(f'Model: {model} inference performance on test dataset')
    
    test_df = pd.read_csv(csv_path)
    
    target = test_df['weight']
    pred = model.predict(test_df['area'].values.reshape(-1, 1))
    
    # Regression Metric
    df_perf = eval_model_regressor(target, pred)
    
    print(df_perf)
    
    # Classification Metric: 0: <=250g; 1: 250-350g; 2: >350g
    y_true = test_df['label'].to_list()
    modulus = np.vectorize(lambda x: 0 if x <= 2500 else 1 if x <= 3500 else 2)
    pred = model.predict(test_df['area'].values.reshape(-1, 1))
    y_pred = modulus(pred).tolist()
    
    model_performance_classifier(y_true, y_pred)
    
    print('--------------------------------------')
    

if __name__ == '__main__':
    ML_OPTIONS = ['decision_tree', 'gradient_boosting', 'knn',
              'logistic_regression', 'naive_bayes', 'random_forest', 'svm']
    
    ml_models = train_evaluate_ml(csv_path='../assets/v2/dataset_v2.csv')

    for model in ml_models:
        ml_test_infer(csv_path='../assets/v2/test.csv', model=ml_models[model])

