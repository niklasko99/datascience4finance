import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from pandas.plotting import parallel_coordinates
from scipy.stats import randint, uniform
from helper_functions import *

def main():
    # Load the data:
    train = pd.read_csv('data/train_imputed.csv')
    test = pd.read_csv('data/test_imputed.csv')

    # Create_training_and_test_sets
    X_train, y_train, X_test, y_test, train, test = create_training_and_test_set(train, test, iteration=3)

    # Create an XGBoost classifier:
    xgb = XGBClassifier(random_state=42)

    # Define the hyperparameter ranges:
    # TODO: Redefine the hyperparameter ranges for the XGBoost

    # hyperparameter_ranges = {
    #     'n_estimators': randint(650, 700),  # Refined range for estimators
    #     'max_depth': randint(3, 5),  # Refined range for depth
    #     'learning_rate': uniform(0.04, 0.06),  # Narrowed learning rate range for finer tuning
    #     'subsample': uniform(0.6, 0.1),  # Focused range for subsample (fraction of samples)
    #     'colsample_bytree': uniform(0.65, 0.1),  # Refined range for feature sampling per tree
    #     'min_child_weight': randint(35, 40),  # Focused range for min_child_weight (leaf node size)
    #     'gamma': uniform(0.7, 1.5),  # Narrowed gamma range for better regularization control
    #     'booster': ['gbtree'],  # Fixed booster type (as before)
    #     'objective': ['binary:logistic'],  # Fixed objective (binary classification)
    #     'lambda': uniform(55, 5),  # Keep L2 regularization range as is
    #     'alpha': uniform(18, 8),  # Keep L1 regularization range as is
    # }

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
                                                                              best_hyperparameters_path='output/xgb/xgb_best_hyperparameters.json')

    # Save the search results:
    search_results = save_cv_results(search,
                                     best_hyperparameters,
                                     'output/xgb/xgb_search_results.csv')
    
    # # Plot the search results:
    # plot_parallel_coordinates_for_xgb(search_results, 
    #                                  'output/xgb/xgb_search_results.png')
    
    
    # Train the model now
    best_model = train_model(best_model, 
                             X_train, 
                             y_train,
                             'output/xgb/xgb_model.pkl')
    
    # Plot the feature importances:
    top_features = feature_importance(best_model,
                                            X_train,
                                            num_features=15,
                                            plot_path='output/xgb/xgb_feature_importances.png',
                                            csv_path='output/xgb/xgb_feature_importances.csv')
    
    # Evaluate the model:
    results = evaluate_model(best_model,
                             X_train,
                             y_train,
                             train,
                             X_test,
                             y_test,
                             test,
                             True,
                             'output/xgb/xgb_train_results.csv',
                             'output/xgb/xgb_test_results.csv',
                             'output/xgb/xgb_train_predictions.csv',
                             'output/xgb/xgb_test_predictions.csv')
    
    # Plot the ROC curve:
    auc_score = plot_valid_test_roc_curve(best_model,
                                          validation_roc_data,
                                          final_evaluation=True,
                                          only_test_curve=False,
                                          X_test = X_test,
                                          y_test = y_test,
                                          path = 'output/xgb/xgb_roc_curve.png')
    
    print(auc_score)

                                                         
if __name__ == '__main__':
    main()
