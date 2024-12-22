import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from pandas.plotting import parallel_coordinates
from scipy.stats import randint, uniform
from helper_functions import *

# This script is used to perform a robustness check for the XGBoost model by using the not imputed data

def main():
    # Load the data:
    train = pd.read_csv('data/train_original.csv')
    test = pd.read_csv('data/test_original.csv')

    # Create_training_and_test_sets
    X_train, y_train, X_test, y_test, train, test = create_training_and_test_set(train, test, iteration=3)

    # Create an XGBoost classifier:
    xgb = XGBClassifier(random_state=42)

    # Define the hyperparameter ranges:
    # TODO: Redefine the hyperparameter ranges for the XGBoost

    with open('output/xgb/xgb_best_hyperparameters.json', 'r') as f:
        params = json.load(f)

    hyperparameter_ranges = {key: [value] for key, value in params.items()}


    # Perform the random search:
    search, best_model, best_hyperparameters, validation_roc_data = search_cv(xgb, 
                                                                              X_train,
                                                                              y_train,
                                                                              hyperparameter_ranges,
                                                                              n_iter=1,
                                                                              num_folds=3,
                                                                              best_hyperparameters_path='output/xgb_robustness_check/xgb_best_hyperparameters.json')

    # Save the search results:
    search_results = save_cv_results(search,
                                     best_hyperparameters,
                                     'output/xgb_robustness_check/xgb_search_results.csv')
    
    # # Plot the search results:
    # plot_parallel_coordinates_for_xgb(search_results, 
    #                                  'output/xgb_robustness_check/xgb_search_results.png')
    
    
    # Train the model now
    best_model = train_model(best_model, 
                             X_train, 
                             y_train,
                             'output/xgb_robustness_check/xgb_model.pkl')
    
    # Plot the feature importances:
    top_features = feature_importance(best_model,
                                            X_train,
                                            num_features=15,
                                            plot_path='output/xgb_robustness_check/xgb_feature_importances.png',
                                            csv_path='output/xgb_robustness_check/xgb_feature_importances.csv')
    
    # Evaluate the model:
    results = evaluate_model(best_model,
                             X_train,
                             y_train,
                             train,
                             X_test,
                             y_test,
                             test,
                             True,
                             'output/xgb_robustness_check/xgb_train_results.csv',
                             'output/xgb_robustness_check/xgb_test_results.csv',
                             'output/xgb_robustness_check/xgb_train_predictions.csv',
                             'output/xgb_robustness_check/xgb_test_predictions.csv')
    
    # Plot the ROC curve:
    auc_score = plot_valid_test_roc_curve(best_model,
                                          validation_roc_data,
                                          final_evaluation=True,
                                          only_test_curve=False,
                                          X_test = X_test,
                                          y_test = y_test,
                                          path = 'output/xgb_robustness_check/xgb_roc_curve.png')
    
    print(auc_score)

                                                         
if __name__ == '__main__':
    main()
