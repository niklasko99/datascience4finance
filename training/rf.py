import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from pandas.plotting import parallel_coordinates
from scipy.stats import randint, uniform
from helper_functions import *

def main():
    # Load the data:
    train = pd.read_csv('data/train_imputed.csv')
    test = pd.read_csv('data/test_imputed.csv')

    # create_training_and_test_sets
    X_train, y_train, X_test, y_test, train, test = create_training_and_test_set(train, test, iteration=3)

    # Create a random forest classifier:
    rf = RandomForestClassifier(random_state=42)

    # Define the hyperparameter ranges:
    # TODO: Redefine the hyperparameter ranges for the random forest

#     # Adjusted hyperparameter ranges
#     hyperparameter_ranges = {
#         'n_estimators': randint(550, 1200),  # Larger range for robustness, common in financial modeling #first 500
#         'max_features': ["sqrt"],  # Focus on square root of features for generalization
#         'max_depth': randint(5, 6),  # Slightly increased depth range to capture more complexity
#         'min_samples_split': randint(20, 40),  # Wider range for splits to balance overfitting/generalization
#         'min_samples_leaf': randint(17, 25),  # Expanded range for leaf sizes to handle class imbalance
#         'bootstrap': [True],  # Always true in Random Forest to enable bootstrapping
#         'min_impurity_decrease': uniform(0.0003, 0.0025), # first 0.005
#         'criterion': ['entropy']  # Expanded criterion options for impurity calculation # only entropy first
# }

    # Adjusted hyperparameter ranges
    hyperparameter_ranges = {
        'n_estimators': randint(550, 1400),  # Larger range for robustness, common in financial modeling #first 500
        'max_features': ["sqrt"],  # Focus on square root of features for generalization
        'max_depth': randint(5, 9),  # Slightly increased depth range to capture more complexity
        'min_samples_split': randint(10, 50),  # Wider range for splits to balance overfitting/generalization
        'min_samples_leaf': randint(10, 35),  # Expanded range for leaf sizes to handle class imbalance
        'bootstrap': [True],  # Always true in Random Forest to enable bootstrapping
        'min_impurity_decrease': uniform(0.000, 0.0025), # first 0.005
        'criterion': ['entropy']  # Expanded criterion options for impurity calculation # only entropy first
}

    # Perform the random search:
    search, best_model, best_hyperparameters, validation_roc_data = search_cv(rf, 
                                                                              X_train,
                                                                              y_train,
                                                                              hyperparameter_ranges,
                                                                              n_iter=150,
                                                                              num_folds=3,
                                                                              best_hyperparameters_path='output/rf/rf_best_hyperparameters.json')

    # Save the search results:
    search_results = save_cv_results(search,
                                     best_hyperparameters,
                                     'output/rf/rf_search_results.csv')
    
    # Plot the search results:
    plot_parallel_coordinates_for_rf(search_results, 
                                     'output/rf/rf_search_results.png')
    
    # Train the model now
    best_model = train_model(best_model, 
                             X_train, 
                             y_train,
                             'output/rf/rf_model.pkl')
    
    # Plot the feature importances:
    top_features = feature_importance(best_model,
                                            X_train,
                                            num_features=15,
                                            plot_path='output/rf/rf_feature_importances.png',
                                            csv_path='output/rf/rf_feature_importances.csv')
    
    # Evaluate the model:
    results = evaluate_model(best_model,
                             X_train,
                             y_train,
                             train,
                             X_test,
                             y_test,
                             test,
                             False,
                             'output/rf/rf_train_results.csv',
                             'output/rf/rf_test_results.csv',
                             'output/rf/rf_train_predictions.csv',
                             'output/rf/rf_test_predictions.csv')
    
    # Plot the ROC curve:
    auc_score = plot_valid_test_roc_curve(best_model,
                                          validation_roc_data,
                                          final_evaluation = False,
                                          only_test_curve = False,
                                          X_test = X_test,
                                          y_test = y_test,
                                          path = 'output/rf/rf_roc_curve.png')
    
    print(auc_score)
    
                                                         
if __name__ == '__main__':
    main()