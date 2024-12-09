# Feature Selection
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("1ST ITERATION")

# Load feature importance scores
xgboost_importance = pd.read_csv('feature_selection/1_iteration/xgb/xgb_feature_importances.csv')
random_forest_importance = pd.read_csv('feature_selection/1_iteration/rf/rf_feature_importances.csv')

# Ensure feature lists are aligned
assert set(xgboost_importance['Feature']) == set(random_forest_importance['Feature']), "Feature lists are not identical!"

# Merge importances on 'Feature'
feature_importances = pd.merge(
    xgboost_importance, random_forest_importance, on='Feature', suffixes=('_xgb', '_rf')
)

# Compute mean importance scores
feature_importances['Mean_Importance'] = feature_importances[['Importance_xgb', 'Importance_rf']].mean(axis=1)

# Sort by mean importance
feature_importances = feature_importances.sort_values(by='Mean_Importance', ascending=False)

# Plot Feature Importances (Mean)
plt.figure(figsize=(14, 7))
plt.bar(feature_importances['Feature'], feature_importances['Mean_Importance'], color='skyblue')
plt.axvline(x=0.6 * len(feature_importances), color='red', linestyle='--', label='Top 60% Threshold')
plt.xticks([])
plt.title('Mean Feature Importance Scores (XGBoost + Random Forest)')
plt.xlabel('Features')
plt.ylabel('Mean Importance Scores')
plt.legend()
plt.tight_layout()
plt.savefig('feature_selection/1_iteration/mean_feature_importance_1_iteration.png')
plt.close()

# Select Top 60% of features
top_features = feature_importances.head(int(0.6 * len(feature_importances)))['Feature'].tolist()

# Save Selected Features as JSON
with open('feature_selection/1_iteration/top_features_mean.json', 'w') as f:
    json.dump(top_features, f)

# Summary
print(f"Number of total features: {len(xgboost_importance)}")
print(f"Number of features selected (Top 60%): {len(top_features)}")

# Identify Features with Zero Importance
zero_importance_xgb = xgboost_importance[xgboost_importance['Importance'] == 0]
zero_importance_rf = random_forest_importance[random_forest_importance['Importance'] == 0]

# Number of Features with Zero Importance in xgb
print(f"Number of features with importance 0 in xgboost: {len(zero_importance_xgb)}")

# Number of Features with Zero Importance in rf
print(f"Number of features with importance 0 in random forest: {len(zero_importance_rf)}")
# Find Zero Importance in Both Models
features_zero_in_both = pd.merge(zero_importance_xgb, zero_importance_rf, on='Feature')
print(f"Number of features with importance 0 in both models: {len(features_zero_in_both)}")

# get the sum of weights in xgboost_importance
sum_weights_xgb = xgboost_importance['Importance'].sum()
print(f"Sum of weights in xgboost: {sum_weights_xgb}")

# get the sum of weights in random_forest_importance
sum_weights_rf = random_forest_importance['Importance'].sum()
print(f"Sum of weights in random forest: {sum_weights_rf}")

# get the sum of weights in feature_importances
sum_weights = feature_importances['Mean_Importance'].sum()
print(f"Sum of weights in feature importances: {sum_weights}")
# get the sum of weights in feature_importances of the top 60%
sum_weights_top = feature_importances.head(int(0.60 * len(feature_importances)))['Mean_Importance'].sum()
print(f"Sum of weights in feature importances of the top 60%: {sum_weights_top}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("\n2ND ITERATION")

# # 2ND ITERATION
# Load feature importance scores
xgboost_importance = pd.read_csv('feature_selection/2_iteration/xgb/xgb_feature_importances.csv')
random_forest_importance = pd.read_csv('feature_selection/2_iteration/rf/rf_feature_importances.csv')

# Ensure feature lists are aligned
assert set(xgboost_importance['Feature']) == set(random_forest_importance['Feature']), "Feature lists are not identical!"

# Merge importances on 'Feature'
feature_importances = pd.merge(
    xgboost_importance, random_forest_importance, on='Feature', suffixes=('_xgb', '_rf')
)

# Compute mean importance scores
feature_importances['Mean_Importance'] = feature_importances[['Importance_xgb', 'Importance_rf']].mean(axis=1)

# Sort by mean importance
feature_importances = feature_importances.sort_values(by='Mean_Importance', ascending=False)

# Plot Feature Importances (Mean)
plt.figure(figsize=(14, 7))
plt.bar(feature_importances['Feature'], feature_importances['Mean_Importance'], color='skyblue')
plt.axvline(x=0.75 * len(feature_importances), color='red', linestyle='--', label='Top 75% Threshold')
plt.xticks([])
plt.title('Mean Feature Importance Scores (XGBoost + Random Forest)')
plt.xlabel('Features')
plt.ylabel('Mean Importance Scores')
plt.legend()
plt.tight_layout()
plt.savefig('feature_selection/2_iteration/mean_feature_importance_2_iteration.png')
plt.close()

# Select Top 75% of features
top_features = feature_importances.head(int(0.75 * len(feature_importances)))['Feature'].tolist()

# Save Selected Features as JSON
with open('feature_selection/2_iteration/top_features_mean.json', 'w') as f:
    json.dump(top_features, f)

# Summary
print(f"Number of total features: {len(xgboost_importance)}")
print(f"Number of features selected (Top 75%): {len(top_features)}")

# Identify Features with Zero Importance
zero_importance_xgb = xgboost_importance[xgboost_importance['Importance'] == 0]
zero_importance_rf = random_forest_importance[random_forest_importance['Importance'] == 0]

# Number of Features with Zero Importance in xgb
print(f"Number of features with importance 0 in xgboost: {len(zero_importance_xgb)}")

# Number of Features with Zero Importance in rf
print(f"Number of features with importance 0 in random forest: {len(zero_importance_rf)}")
# Find Zero Importance in Both Models
features_zero_in_both = pd.merge(zero_importance_xgb, zero_importance_rf, on='Feature')
print(f"Number of features with importance 0 in both models: {len(features_zero_in_both)}")

# get the sum of weights in xgboost_importance
sum_weights_xgb = xgboost_importance['Importance'].sum()
print(f"Sum of weights in xgboost: {sum_weights_xgb}")

# get the sum of weights in random_forest_importance
sum_weights_rf = random_forest_importance['Importance'].sum()
print(f"Sum of weights in random forest: {sum_weights_rf}")

# get the sum of weights in feature_importances
sum_weights = feature_importances['Mean_Importance'].sum()
print(f"Sum of weights in feature importances: {sum_weights}")
# get the sum of weights in feature_importances of the top 60%
sum_weights_top = feature_importances.head(int(0.75 * len(feature_importances)))['Mean_Importance'].sum()
print(f"Sum of weights in feature importances of the top 75%: {sum_weights_top}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("\n3RD ITERATION")

# # 3RD ITERATION
# Load feature importance scores
xgboost_importance = pd.read_csv('feature_selection/3_iteration/xgb/xgb_feature_importances.csv')
random_forest_importance = pd.read_csv('feature_selection/3_iteration/rf/rf_feature_importances.csv')

# Ensure feature lists are aligned
assert set(xgboost_importance['Feature']) == set(random_forest_importance['Feature']), "Feature lists are not identical!"

# Merge importances on 'Feature'
feature_importances = pd.merge(
    xgboost_importance, random_forest_importance, on='Feature', suffixes=('_xgb', '_rf')
)

# Compute mean importance scores
feature_importances['Mean_Importance'] = feature_importances[['Importance_xgb', 'Importance_rf']].mean(axis=1)

# Sort by mean importance
feature_importances = feature_importances.sort_values(by='Mean_Importance', ascending=False)

# Plot Feature Importances (Mean)
plt.figure(figsize=(14, 7))
plt.bar(feature_importances['Feature'], feature_importances['Mean_Importance'], color='skyblue')
plt.axvline(x=0.75 * len(feature_importances), color='red', linestyle='--', label='Top 75% Threshold')
plt.xticks([])
plt.title('Mean Feature Importance Scores (XGBoost + Random Forest)')
plt.xlabel('Features')
plt.ylabel('Mean Importance Scores')
plt.legend()
plt.tight_layout()
plt.savefig('feature_selection/3_iteration/mean_feature_importance_3_iteration.png')
plt.close()

# Select Top 75% of features
top_features = feature_importances.head(int(0.75 * len(feature_importances)))['Feature'].tolist()

# Save Selected Features as JSON
with open('feature_selection/3_iteration/top_features_mean.json', 'w') as f:
    json.dump(top_features, f)

# Summary
print(f"Number of total features: {len(xgboost_importance)}")
print(f"Number of features selected (Top 75%): {len(top_features)}")

# Identify Features with Zero Importance
zero_importance_xgb = xgboost_importance[xgboost_importance['Importance'] == 0]
zero_importance_rf = random_forest_importance[random_forest_importance['Importance'] == 0]

# Number of Features with Zero Importance in xgb
print(f"Number of features with importance 0 in xgboost: {len(zero_importance_xgb)}")

# Number of Features with Zero Importance in rf
print(f"Number of features with importance 0 in random forest: {len(zero_importance_rf)}")
# Find Zero Importance in Both Models
features_zero_in_both = pd.merge(zero_importance_xgb, zero_importance_rf, on='Feature')
print(f"Number of features with importance 0 in both models: {len(features_zero_in_both)}")

# get the sum of weights in xgboost_importance
sum_weights_xgb = xgboost_importance['Importance'].sum()
print(f"Sum of weights in xgboost: {sum_weights_xgb}")

# get the sum of weights in random_forest_importance
sum_weights_rf = random_forest_importance['Importance'].sum()
print(f"Sum of weights in random forest: {sum_weights_rf}")

# get the sum of weights in feature_importances
sum_weights = feature_importances['Mean_Importance'].sum()
print(f"Sum of weights in feature importances: {sum_weights}")
# get the sum of weights in feature_importances of the top 60%
sum_weights_top = feature_importances.head(int(0.75 * len(feature_importances)))['Mean_Importance'].sum()
print(f"Sum of weights in feature importances of the top 75%: {sum_weights_top}")


def plot_feature_importance(importances_df, top_features, model_name, save_path, top_n=100):
    """
    Plots feature importance with additional information on percentage of kept features.
    
    Parameters:
        importances_df (pd.DataFrame): DataFrame with feature importance data.
        top_features (list): List of kept features.
        model_name (str): Name of the model (e.g., 'XGBoost').
        save_path (str): File path to save the plot.
        top_n (int): Number of top features to plot.
    """
    # Sort features by importance and filter top N
    importances_df = importances_df.sort_values(by='Importance', ascending=False).head(top_n)

    # Calculate percentage of kept features
    total_top_features = len(importances_df)
    kept_features = sum(1 for x in importances_df['Feature'] if x in top_features)
    kept_percentage = (kept_features / total_top_features) * 100

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(14, 7))

    # Define colors based on feature selection
    colors = ['#2ca02c' if x in top_features else '#1f77b4' for x in importances_df['Feature']]

    # Plot feature importances
    ax.bar(importances_df['Feature'], importances_df['Importance'], color=colors)

    # Customize axis labels and title
    ax.set_xticks([])
    ax.set_title(f'{model_name} Top {top_n} Feature Importance After First Iteration', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Importance Scores', fontsize=14)

    # Add gridlines for Y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage information on the right side
    percentage_info = f"{kept_percentage:.1f}%\nof Top {top_n}\nFeatures\nWere Kept"
    ax.text(1.02, 0.5, percentage_info, transform=ax.transAxes, fontsize=14, 
            fontweight='bold', color='black', ha='center', va='center', bbox=dict(boxstyle="round", facecolor="#f0f0f0"))

    # Add custom legend
    legend_labels = ['Kept Features', 'Removed Features']
    legend_colors = ['#2ca02c', '#1f77b4']
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    ax.legend(handles, legend_labels, title='Feature Status', fontsize=12, title_fontsize=13, loc='upper right')

    # Final layout adjustments
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Load Data for XGBoost and Random Forest
xgboost_importance = pd.read_csv('feature_selection/1_iteration/xgb/xgb_feature_importances.csv')
rf_importance = pd.read_csv('feature_selection/1_iteration/rf/rf_feature_importances.csv')
# Plot Feature Importance for Both Models
plot_feature_importance(xgboost_importance, top_features, 'XGBoost', 'feature_selection/xgb_importance_kept_features.png', 150)
plot_feature_importance(rf_importance, top_features, 'Random Forest', 'feature_selection/rf_importance_kept_features.png', 150)