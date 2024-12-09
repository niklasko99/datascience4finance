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
    X_train, y_train, X_test, y_test, train, test = create_training_and_test_set(train, test, iteration=1)

    # Create an XGBoost classifier:
    xgb = XGBClassifier(random_state=42)

    # Define the hyperparameter ranges:
    # TODO: Redefine the hyperparameter ranges for the XGBoost

    # Adjusted hyperparameter ranges for XGBoost corresponding to Random Forest parameters
    hyperparameter_ranges = {
        'n_estimators': randint(550, 1100),  # Same range for number of estimators
        'max_depth': randint(3, 5),  # Same range for depth of trees
        'learning_rate': uniform(0.005, 0.045),  # Not directly equivalent, but it is necessary for gradient boosting algorithm
        'subsample': uniform(0.5, 0.4),  # Analogous to min_samples_split (fraction of samples used per tree)
        'colsample_bytree': uniform(0.5, 0.35),  # Analogous to max_features (fraction of features used per tree)
        'min_child_weight': randint(30, 50),  # Analogous to min_samples_leaf (controls the minimum sum of instance weight in a child)
        'gamma': uniform(0.5, 5.0),  # Analogous to min_impurity_decrease (controls regularization)
        'booster': ['gbtree'],  # 'gbtree' used for tree-based models, similar to Random Forest
        'objective': ['binary:logistic'],  # Common objective for binary classification (similar to criterion for impurity calculation)
        'lambda': uniform(45, 45),  # L2 regularization, increased range for stronger regularization
        'alpha': uniform(18, 25),  # L1 regularization, increased range for stronger regularization
    }

    # Perform the random search:
    search, best_model, best_hyperparameters, validation_roc_data = search_cv(xgb, 
                                                                              X_train,
                                                                              y_train,
                                                                              hyperparameter_ranges,
                                                                              n_iter=100,
                                                                              num_folds=3,
                                                                              best_hyperparameters_path='output/xgb/xgb_best_hyperparameters.json')

    # Save the search results:
    search_results = save_cv_results(search,
                                     best_hyperparameters,
                                     'output/xgb/xgb_search_results.csv')
    
    # Plot the search results:
    plot_parallel_coordinates_for_xgb(search_results, 
                                     'output/xgb/xgb_search_results.png')
    
    
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
                             False,
                             'output/xgb/xgb_train_results.csv',
                             'output/xgb/xgb_test_results.csv',
                             'output/xgb/xgb_train_predictions.csv',
                             'output/xgb/xgb_test_predictions.csv')
    
    # Plot the ROC curve:
    auc_score = plot_valid_test_roc_curve(best_model,
                                          validation_roc_data,
                                          final_evaluation=False,
                                          only_test_curve=False,
                                          X_test = X_test,
                                          y_test = y_test,
                                          path = 'output/xgb/xgb_roc_curve.png')
    
    print(auc_score)

                                                         
if __name__ == '__main__':
    main()
