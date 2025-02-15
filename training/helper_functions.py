import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import pickle
import joblib
import gzip
import warnings
import json

warnings.filterwarnings("ignore")


def create_training_and_test_set(train, test, iteration=None):
    """
    Prepares training and test datasets by sorting and selecting relevant features.
    
    Parameters:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
        iteration (int): Optional feature iteration to load specific feature subsets.
    
    Returns:
        X_train, y_train, X_test, y_test: Processed feature and target datasets.
    """
    # Convert 'period' to datetime for consistent sorting
    train['period'] = pd.to_datetime(train['period'])
    test['period'] = pd.to_datetime(test['period'])

    # Sort datasets by 'ticker' and 'period' to maintain time-series order
    train = train.sort_values('period').reset_index(drop=True)
    test = test.sort_values('period').reset_index(drop=True)

    # select float64 columns
    float_columns = train.select_dtypes(include=['float64']).columns
    train[float_columns] = train[float_columns].astype('float32')
    test[float_columns] = test[float_columns].astype('float32')

    # Extract features (X) and labels (y)
    X_train = train.drop(['ticker', 'cik', 'sic', 'period', 'y', 'adsh'], axis=1)
    y_train = train['y']
    X_test = test.drop(['ticker', 'cik', 'sic', 'period', 'y', 'adsh'], axis=1)
    y_test = test['y']

    # Convert boolean columns to floats if present
    X_train[X_train.select_dtypes(include=['bool']).columns] = X_train.select_dtypes(include=['bool']).astype(int)
    X_test[X_test.select_dtypes(include=['bool']).columns] = X_test.select_dtypes(include=['bool']).astype(int)

    # # Perform undersampling to balance the training set
    # rus = RandomUnderSampler(random_state=42)
    # X_train, y_train = rus.fit_resample(X_train, y_train)

    # Load specific feature subsets if iteration is provided
    if iteration:
        feature_file = f'feature_selection/{iteration}_iteration/top_features_mean.json'
        try:
            with open(feature_file) as f:
                features = json.load(f)
            X_train = X_train[features]
            X_test = X_test[features]
            # print number of columns in X_train and X_test
            
            print(f"Number of columns in X_train: {X_train.shape[1]}")
            print(f"Number of columns in X_test: {X_test.shape[1]}")

        except FileNotFoundError:
            print(f"Feature file {feature_file} not found. Using all features.")

    return X_train, y_train, X_test, y_test, train, test


def search_cv(model, 
              X_train, 
              y_train, 
              hyperparameters, 
              n_iter=100, 
              num_folds=3,
              best_hyperparameters_path='best_hyperparameters.json'):
    """
    Performs time-series cross-validation and randomized hyperparameter tuning.

    Parameters:
        model: Machine learning model compatible with `RandomizedSearchCV` (e.g., from Scikit-Learn).
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target variable corresponding to X_train.
        hyperparameters (dict): Dictionary specifying the hyperparameter search space.
        n_iter (int): Number of iterations for random hyperparameter sampling.
        num_folds (int): Number of time-series splits (folds) for cross-validation.
        best_hyperparameters_path (str): File path to save the best hyperparameters in JSON format.

    Returns:
        tuple:
            - search (RandomizedSearchCV): The fitted RandomizedSearchCV object.
            - best_model: Best model instance with the optimal hyperparameters found during search.
            - best_hyperparameters (dict): Dictionary of the best hyperparameter values.
            - validation_roc_data (list of tuples): ROC curve data for validation folds. 
              Each tuple contains (fpr, tpr, roc_auc) for a fold.
    """
    # Initialize time-series cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=num_folds)

    # Set up RandomizedSearchCV with the given model and hyperparameter search space
    search = RandomizedSearchCV(
        model,  # The base machine learning model
        hyperparameters,  # Hyperparameter search space
        n_iter=n_iter,  # Number of hyperparameter combinations to evaluate
        scoring='roc_auc',  # Use ROC AUC as the evaluation metric
        n_jobs=-1,  # Use all available CPU cores for parallel processing
        cv=tscv,  # Use time-series cross-validation
        random_state=42,  # Ensure reproducibility
        verbose=3  # Print progress during search
    )
    
    # Perform hyperparameter search using RandomizedSearchCV
    search.fit(X_train, y_train)

    # Extract the best model and its hyperparameters
    best_model = search.best_estimator_  # Best model instance
    best_hyperparameters = search.best_params_  # Optimal hyperparameters

    # Save the best hyperparameters to a JSON file for reproducibility
    print("Best model and hyperparameters found. Saving to disk...")
    with open(best_hyperparameters_path, 'w') as f:
        json.dump(best_hyperparameters, f)

    # Collect ROC curve data for validation folds
    validation_roc_data = []  # List to store (fpr, tpr, auc) for each fold
    for train_idx, val_idx in tscv.split(X_train):
        # Split training data into current training and validation sets
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train the model on the current fold's training data
        best_model.fit(X_train_fold, y_train_fold)
        
        # Predict probabilities for the validation set
        y_val_pred_prob = best_model.predict_proba(X_val_fold)[:, 1]
        
        # Compute the ROC curve and AUC for the validation set
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_prob)  # False positive rate and true positive rate
        roc_auc = roc_auc_score(y_val_fold, y_val_pred_prob)  # Area under the ROC curve
        validation_roc_data.append((fpr, tpr, roc_auc))  # Store fold data

    return search, best_model, best_hyperparameters, validation_roc_data


def save_cv_results(search, best_hyperparameters, path):
    """
    Saves cross-validation results as a sorted CSV file.

    Parameters:
        search (RandomizedSearchCV object): Completed hyperparameter search.
        best_hyperparameters (dict): Best hyperparameters from search.
        path (str): Path to save the CSV file.

    Returns:
        results (pd.DataFrame): DataFrame of sorted results.
    """
    results = pd.DataFrame(search.cv_results_)
    
    # Extract relevant columns
    interested_columns = [f'param_{param}' for param in best_hyperparameters.keys()] + \
                         ['mean_test_score', 'std_test_score', 'rank_test_score']
    
    # Sort and save results
    results = results[interested_columns].sort_values('mean_test_score', ascending=False)
    results.to_csv(path, index=False)
    print(f"Results saved to {path}.")
    return results







def train_model(best_model, X_train, y_train, path):
    """
    Trains the best model on the full training set and saves it.

    Parameters:
        best_model: Machine learning model to be trained.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training labels.
        path (str): Path to save the trained model.

    Returns:
        best_model: Trained model after fitting.
    """
    # Train the model on the full training set
    best_model.fit(X_train, y_train)

    # Save the model to the specified path
    joblib.dump(best_model, path)
    print(f"Trained model saved to {path}")
    
    return best_model


def evaluate_model(best_model, 
                   X_train, y_train, train, 
                   X_test, y_test, test, 
                   final_evaluation=False, 
                   path_for_train_metrics=None, 
                   path_for_test_metrics=None, 
                   path_for_train_predictions=None, 
                   path_for_test_predictions=None):
    """
    Evaluates the best model on both the training and test sets, saving metrics and predictions.

    Parameters:
        best_model: Trained model to evaluate.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training labels.
        train (pd.DataFrame): Training dataset with identifiers.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test labels.
        test (pd.DataFrame): Test dataset with identifiers.
        final_evaluation (bool, optional): If True, the model is evaluated for final results.
        path_for_train_metrics (str, optional): File path to save training evaluation metrics.
        path_for_test_metrics (str, optional): File path to save test evaluation metrics.
        path_for_train_predictions (str, optional): File path to save training model predictions.
        path_for_test_predictions (str, optional): File path to save test model predictions.

    Returns:
        dict: Dictionary containing evaluation metrics and results for both datasets.
    """
    # Evaluate training set
    y_train_pred = best_model.predict(X_train)
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]

    # Evaluate test set
    y_test_pred = best_model.predict(X_test)
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

    # Performance metrics for training set
    train_metrics = {
        "Accuracy": best_model.score(X_train, y_train),
        "F1": f1_score(y_train, y_train_pred),
        "Recall": recall_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred),
        "AUC": roc_auc_score(y_train, y_train_pred_prob)
    }

    if path_for_train_metrics:
        with open(path_for_train_metrics, 'w') as f:
            for metric, value in train_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        print(f"Training performance metrics saved to {path_for_train_metrics}")

    # Save metrics if paths are provided
    if final_evaluation:
            # Performance metrics for test set
        test_metrics = {
            "Accuracy": best_model.score(X_test, y_test),
            "F1": f1_score(y_test, y_test_pred),
            "Recall": recall_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred),
            "AUC": roc_auc_score(y_test, y_test_pred_prob)
        }
        if path_for_test_metrics:
            with open(path_for_test_metrics, 'w') as f:
                for metric, value in test_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            print(f"Test performance metrics saved to {path_for_test_metrics}")

    # Create DataFrames for predictions
    train_results = pd.DataFrame({
        'y_true': y_train,
        'y_pred': y_train_pred,
        'y_pred_prob': y_train_pred_prob
    })
    train_identifiers = train[['ticker', 'period', 'sic']]
    train_results = pd.concat([train_identifiers, train_results], axis=1, join="inner")
    train_results.reset_index(drop=True, inplace=True)

    test_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_pred_prob': y_test_pred_prob
    })
    test_identifiers = test[['ticker', 'period', 'sic']]
    test_results = pd.concat([test_identifiers, test_results], axis=1)

    # Save predictions if paths are provided
    if path_for_train_predictions:
        train_results.to_csv(path_for_train_predictions, index=False)
        print(f"Training predictions saved to {path_for_train_predictions}")

    if path_for_test_predictions:
        test_results.to_csv(path_for_test_predictions, index=False)
        print(f"Test predictions saved to {path_for_test_predictions}")

    # Return metrics and results
    if final_evaluation:
        results = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_results": train_results,
            "test_results": test_results
        }
        return results
    else:
        results = {
            "train_metrics": train_metrics,
            "train_results": train_results,
            "test_results": test_results
        }
        return results


def plot_valid_test_roc_curve(model,
                              validation_roc_data,
                              final_evaluation=False,
                              only_test_curve=False, 
                              X_test=None, 
                              y_test=None, 
                              path=None):
    """
    Plots ROC curves for validation (and optionally test) sets.

    Parameters:
        model: Trained classifier with `predict_proba`.
        validation_roc_data: List of tuples containing FPR, TPR, and AUC for validation folds.
        final_evaluation (bool): If True, allows plotting the test set ROC curve.
        only_test_curve (bool): If True and final_evaluation is enabled, plots only the test ROC curve.
        X_test (pd.DataFrame, optional): Test feature set.
        y_test (pd.Series, optional): Test labels.
        path (str, optional): Path to save the plot if provided.

    Returns:
        dict: Contains validation AUC and (if applicable) test AUC.
    """
    # Initialize AUC results
    test_auc = None

    # Plot only the test curve if `only_test_curve` is True and `final_evaluation` is enabled
    if only_test_curve and final_evaluation:
        if X_test is not None and y_test is not None:
            # Compute test ROC
            y_test_pred = model.predict_proba(X_test)[:, 1]
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_pred)

            # Plot the test ROC curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr_test, tpr_test, linestyle="-", label=f"Test ROC (AUC = {test_auc:.3f})", color="green")
            plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5)")

            # Add labels and legend
            plt.title("Test ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)

            # Save plot if path is provided
            if path:
                plt.savefig(path, bbox_inches="tight")
                print(f"Test ROC curve plot saved to {path}")

            plt.show()
            plt.close()

        else:
            print("Error: X_test and y_test must be provided to plot the test ROC curve.")
        return {"validation_auc": "N/A", "test_auc": test_auc if test_auc is not None else "N/A"}

    # Plot validation and optionally test ROC curves if `only_test_curve` is False
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    # Compute mean ROC for validation folds
    for fpr, tpr, _ in validation_roc_data:
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr /= len(validation_roc_data)
    mean_auc = np.mean([roc_auc for _, _, roc_auc in validation_roc_data])

    # Initialize plot
    plt.figure(figsize=(10, 8))

    # Plot validation ROC curve
    plt.plot(mean_fpr, mean_tpr, linestyle="--", label=f"Validation Mean ROC (AUC = {mean_auc:.3f})", color="blue")    
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5)")

    # Optionally plot test ROC curve if `final_evaluation` is True
    if final_evaluation and X_test is not None and y_test is not None:
        y_test_pred = model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        plt.plot(fpr_test, tpr_test, linestyle="-", label=f"Test ROC (AUC = {test_auc:.3f})", color="green")

    # Add titles and labels
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save plot if path is provided
    if path:
        plt.savefig(path, bbox_inches="tight")
        print(f"ROC curve plot saved to {path}")

    plt.show()
    plt.close()

    # Return AUC results
    return {"validation_auc": mean_auc, "test_auc": test_auc if test_auc is not None else "N/A"}


def feature_importance(best_model, 
                       X_train, 
                       num_features=15, 
                       plot_path=None, 
                       csv_path=None):
    """
    Plots top feature importances and saves all feature importances as provided by the model.

    Parameters:
        best_model: Trained model with a `feature_importances_` attribute.
        X_train (pd.DataFrame): Training features to extract feature names.
        num_features (int): Number of top features to display in the plot.
        plot_path (str, optional): Path to save the plot.
        csv_path (str, optional): Path to save the CSV file with feature importances.

    Returns:
        feature_importance_df (pd.DataFrame): DataFrame containing feature importances.
    """
    try:
        # Extract feature importances
        feature_importance = best_model.feature_importances_

        # Create DataFrame for CSV export
        feature_importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)

        # Save CSV if specified
        if csv_path:
            feature_importance_df.to_csv(csv_path, index=False)
            print(f"Full feature importances saved to {csv_path}")

        # Extract top features for plotting
        top_features = feature_importance_df.head(num_features)

        # Plot top features
        plt.figure(figsize=(10, 8))
        plt.barh(top_features["Feature"], top_features["Importance"], color='skyblue')
        plt.xlabel("Feature Importance")
        plt.title(f"Top {num_features} Feature Importances")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.gca().invert_yaxis()  # Invert y-axis for descending order

        # Save plot if specified
        if plot_path:
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Top {num_features} feature importance plot saved to {plot_path}")

        plt.show()
        plt.close()

        return feature_importance_df

    except AttributeError:
        print("Error: Model does not have a 'feature_importances_' attribute.")
        return None


# -----------------------
def plot_parallel_coordinates_for_rf(results, path=None):
    """
    Plots parallel coordinates for Random Forest hyperparameter tuning results.

    Parameters:
        results (pd.DataFrame): DataFrame containing cross-validation results.
        path (str, optional): File path to save the plot. If None, the plot will not be saved.

    Returns:
        None
    """
    # Initialize MinMaxScaler for feature scaling
    scaler = MinMaxScaler()
    print(results.columns)
    # Rename relevant columns for readability
    results = results.rename(columns={
        'param_n_estimators': 'n_estimators',
        'param_max_depth': 'max_depth',
        'param_min_samples_split': 'min_samples_split',
        'param_min_samples_leaf': 'min_samples_leaf',
        'param_min_impurity_decrease': 'min_impurity_decrease',
        'param_criterion': 'criterion',
    })

    # if criterion is "gini", set it to 0, else 1
    results['criterion'] = results['criterion'].apply(lambda x: 0 if x == 'gini' else 1)

    # drop criterion
    results = results.drop(columns=['criterion'], errors='ignore')
    # Normalize selected hyperparameters
    for param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_impurity_decrease']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))


    # Drop columns that are not relevant for plotting
    results = results.drop(columns=['param_max_features', 'param_bootstrap', 
                                    'std_test_score', 'rank_test_score'], errors='ignore')

    # Plot the parallel coordinates
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.title("Random Forest Hyperparameter Tuning - Parallel Coordinates")
    plt.xlabel("Hyperparameters")
    plt.ylabel("Normalized Values")
    plt.grid(True)
    plt.legend().remove()

    # Save plot if path is specified
    if path:
        plt.savefig(path)
        print(f"Plot saved to {path}")
    plt.show()
    plt.close()


def plot_parallel_coordinates_for_xgb(results, path=None):
    """
    Plots parallel coordinates for XGBoost hyperparameter tuning results.

    Parameters:
        results (pd.DataFrame): DataFrame containing cross-validation results.
        path (str, optional): File path to save the plot. If None, the plot will not be saved.

    Returns:
        None
    """
    # Initialize MinMaxScaler for feature scaling
    scaler = MinMaxScaler()
    print(results.columns)

    # Rename relevant columns for readability
    results = results.rename(columns={
        'param_n_estimators': 'n_estimators',
        'param_learning_rate': 'learning_rate',
        'param_max_depth': 'max_depth',
        'param_min_child_weight': 'min_child_weight',
        'param_subsample': 'subsample',
        'param_colsample_bytree': 'colsample_bytree',
        'param_reg_lambda': 'reg_lambda',
        'param_reg_alpha': 'reg_alpha',
        'param_scale_pos_weight': 'scale_pos_weight',
        'param_gamma': 'gamma',
        'param_booster': 'booster',
        'param_objective': 'objective',
        'param_lambda': 'lambda',
        'param_alpha': 'alpha'
    })

    # If booster is "gbtree", set it to 1, else 0
    results['booster'] = results['booster'].apply(lambda x: 1 if x == 'gbtree' else 0)
    # drop booster
    results = results.drop(columns=['booster'], errors='ignore')
    # Normalize selected hyperparameters
    for param in ['max_depth', 'min_child_weight', 'n_estimators', 'alpha', 'lambda', 'gamma', 'learning_rate', 'subsample', 'colsample_bytree']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))

    # Drop columns that are not relevant for plotting
    results = results.drop(columns=['std_test_score', 'rank_test_score', 'objective'], errors='ignore')

    # Plot the parallel coordinates
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.title("XGBoost Hyperparameter Tuning - Parallel Coordinates")
    plt.xlabel("Hyperparameters")
    plt.ylabel("Normalized Values")
    plt.grid(True)
    plt.legend().remove()

    # Save plot if path is specified
    if path:
        plt.savefig(path)
        print(f"Plot saved to {path}")
    plt.show()
    plt.close()



def plot_parallel_coordinates_for_logistic_regression(results, path=None):
    """
    Plots parallel coordinates for Logistic Regression hyperparameter tuning results.

    Parameters:
        results (pd.DataFrame): DataFrame containing cross-validation results.
        path (str, optional): File path to save the plot. If None, the plot will not be saved.

    Returns:
        None
    """
    # Initialize MinMaxScaler for feature scaling
    scaler = MinMaxScaler()
    print("Available Columns:", results.columns)

    # Rename relevant columns for readability
    results = results.rename(columns={
        'param_C': 'C',
        'param_penalty': 'penalty',
        'param_solver': 'solver',
        'param_class_weight': 'class_weight',
        'param_max_iter': 'max_iter'
    })

    # Encode categorical parameters as numerical values
    #results['penalty'] = results['penalty'].map({'l1': 1, 'l2': 0}).fillna(-1)
    results['solver'] = results['solver'].map({'liblinear': 1, 'saga': 0}).fillna(-1)
    results['class_weight'] = results['class_weight'].map({'balanced': 1, None: 0}).fillna(-1)

    # Normalize numerical hyperparameters
    for param in ['C', 'max_iter']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))

    # Drop irrelevant columns
    results = results.drop(columns=['std_test_score', 'rank_test_score'], errors='ignore')

    # Plot the parallel coordinates
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.title("Logistic Regression Hyperparameter Tuning - Parallel Coordinates")
    plt.xlabel("Hyperparameters")
    plt.ylabel("Normalized Values")
    plt.grid(True)
    plt.legend().remove()

    # Save the plot if a path is specified
    if path:
        plt.savefig(path)
        print(f"Plot saved to {path}")
    plt.show()
    plt.close()