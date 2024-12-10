import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from pandas.plotting import parallel_coordinates
from scipy.stats import randint, uniform
from helper_functions import *
from sklearn.preprocessing import StandardScaler


def main():
    # Load the data:
    train = pd.read_csv('data/train_imputed.csv')
    test = pd.read_csv('data/test_imputed.csv')

    # create_training_and_test_sets
    X_train, y_train, X_test, y_test, train, test = create_training_and_test_set(train, test, iteration=3)

    # Standardize the features

    scaler = StandardScaler()

    # Standardize the features and convert back to DataFrames
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
)

    # Create a logistic regression model:
    lr = LogisticRegression(random_state=42)

    # Define the hyperparameter ranges:
    # TODO: Redefine the hyperparameter ranges for the logistic regression
    # Adjusted hyperparameter ranges
    hyperparameter_ranges = {
    'C': uniform(loc=0.001, scale=10),  # Regularization strength
    'solver': ['liblinear', 'saga'],    # Solvers supporting l1/l2 penalties
    'class_weight': [None, 'balanced'], # Handle class imbalance
    'max_iter': [100, 200, 500, 1000, 2000]   # Ensure convergence
}

    # Perform the random search:
    search, best_model, best_hyperparameters, validation_roc_data = search_cv(lr, 
                                                                              X_train,
                                                                              y_train,
                                                                              hyperparameter_ranges,
                                                                              n_iter=150,
                                                                              num_folds=3,
                                                                              best_hyperparameters_path='output/logistic_regression/logistic_regression_best_hyperparameters.json')

    # Save the search results:
    search_results = save_cv_results(search,
                                     best_hyperparameters,
                                     'output/logistic_regression/logistic_regression_search_results.csv')
    
    # # Plot the search results:
    plot_parallel_coordinates_for_logistic_regression(search_results, 
                                      'output/logistic_regression/logistic_regression_search_results.png')
    
    # Train the model now
    best_model = train_model(best_model, 
                             X_train, 
                             y_train,
                             'output/logistic_regression/logistic_regression_model.pkl')
    
   
    # Evaluate the model:
    results = evaluate_model(best_model,
                             X_train,
                             y_train,
                             train,
                             X_test,
                             y_test,
                             test,
                             False,
                             'output/logistic_regression/logistic_regression_train_results.csv',
                             'output/logistic_regression/logistic_regression_test_results.csv',
                             'output/logistic_regression/logistic_regression_train_predictions.csv',
                             'output/logistic_regression/logistic_regression_test_predictions.csv')
    
    # Plot the ROC curve:
    auc_score = plot_valid_test_roc_curve(best_model,
                                          validation_roc_data,
                                          final_evaluation = False,
                                          only_test_curve = False,
                                          X_test = X_test,
                                          y_test = y_test,
                                          path = 'output/logistic_regression/logistic_regression_roc_curve.png')
    
    print(auc_score)
    
                                                         
if __name__ == '__main__':
    main()