import os
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), save_path=None):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")

#     if save_path:
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         plt.savefig(os.path.join(save_path, f"{title}.png"))

def plot_learning_curve(regressor, title, X, y, cv, save_path):
    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'hspace': 0.4})
    
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": cv,
        "scoring": 'r2',  # if 'neg_mean_squared_error', change the sign of train_scores_mean and test_scores_mean
        "n_jobs": 4,
        "shuffle": True
    }
    
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(regressor, return_times=True, **common_params)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    
    # Learning curve plot
    ax[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax[0].set_title(title)
    ax[0].set_xlabel("Training examples")
    ax[0].set_ylabel("R2 Score")
    ax[0].grid(True)
    ax[0].legend(loc="best")
    
    # Scalability analysis plot
    ax[1].plot(fit_times.mean(axis=1), test_scores_mean, "o-")
    ax[1].fill_between(
        fit_times.mean(axis=1),
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.3,
    )
    ax[1].set_ylabel("R2 Score")
    ax[1].set_xlabel("Fit time (s)")
    ax[1].set_title(f"Performance of the {regressor.__class__.__name__} regressor")

    plt.savefig(save_path)

    
    
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
        
        title = f"Learning Curves for ({model.title()})"
        plot_learning_curve(regressor, title, X, y, cv=ShuffleSplit(n_splits=50, test_size=0.2, random_state=0), save_path=f'../assets/plot/learning_curve_new_{model}.png')

        
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

