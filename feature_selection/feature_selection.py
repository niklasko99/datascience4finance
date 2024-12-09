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
plt.axvline(x=0.6 * len(feature_importances), color='red', linestyle='--', label='Top 75% Threshold')
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

# print("\n3RD ITERATION")

# # 3RD ITERATION
# # load feature importance scores (xgboost)
# xgboost_importance3 = pd.read_csv('feature_selection/3rd_iteration/xgboost_importance_3rd_iteration.csv')
# # other columns: feature, importance

# # Plot graph (bar charts) to visualize feature importance scores for xgboost
# plt.figure(figsize=(14, 7))
# plt.bar(xgboost_importance3['feature'], xgboost_importance3['importance'])
# plt.axvline(x=0.6 * len(xgboost_importance3), color='red', linestyle='--')
# plt.xticks([])
# plt.title('Feature Importance Scores (XGBoost)')
# plt.xlabel('Features')
# plt.ylabel('Importance Scores')
# plt.tight_layout()
# plt.savefig('feature_selection/3rd_iteration/xgboost_importance_3rd_iteration.png')

# # load feature importance scores (random forest)
# random_forest_importance3 = pd.read_csv('feature_selection/3rd_iteration/rf_importance_3rd_iteration.csv')
# # other columns: feature, importance

# # Plot graph (bar charts) to visualize feature importance scores for random forest
# plt.figure(figsize=(14, 7))
# plt.bar(random_forest_importance3['feature'], random_forest_importance3['importance'])
# plt.axvline(x=0.6 * len(random_forest_importance3), color='red', linestyle='--')
# plt.xticks([])
# plt.title('Feature Importance Scores (Random Forest)')
# plt.xlabel('Features')
# plt.ylabel('Importance Scores')
# plt.tight_layout()
# plt.savefig('feature_selection/3rd_iteration/rf_importance_3rd_iteration.png')
# plt.close()

# # Find features that have an importance of 0
# xgboost_importance3_0 = xgboost_importance3[xgboost_importance3['importance'] == 0]
# print("Number of features with importance 0 in xgboost: ", len(xgboost_importance3_0))
# random_forest_importance3_0 = random_forest_importance3[random_forest_importance3['importance'] == 0]
# print("Number of features with importance 0 in random forest: ", len(random_forest_importance3_0))

# # Find features that have an importance of 0 for both xgboost and random forest
# features3_0 = pd.merge(xgboost_importance3_0, random_forest_importance3_0, on='feature', how='inner')
# print("Number of features with importance 0 for both xgboost and random forest: ", len(features3_0))

# # Find features whose difference in ranks (index) between xgboost and random forest is greater than 15
# xgboost_importance3['index'] = xgboost_importance3.index
# random_forest_importance3['index'] = random_forest_importance3.index
# features_diff3 = pd.merge(xgboost_importance3, random_forest_importance3, on='feature', how='inner')
# features_diff3['diff'] = abs(features_diff3['index_x'] - features_diff3['index_y'])
# features_diff3 = features_diff3[features_diff3['diff'] > 15]
# print("Number of features whose difference in ranks between xgboost and random forest is greater than 20 (10% of features): ", len(features_diff3))

# # How many features are the same in the top 60% of features for both xgboost and random forest?
# xgboost_importance3 = xgboost_importance3.sort_values(by='importance', ascending=False)
# random_forest_importance3 = random_forest_importance3.sort_values(by='importance', ascending=False)
# top_60_xgboost3 = xgboost_importance3.head(int(0.6 * len(xgboost_importance3)))
# top_60_rf3 = random_forest_importance3.head(int(0.6 * len(random_forest_importance3)))
# features_same_top3_60 = pd.merge(top_60_xgboost3, top_60_rf3, on='feature', how='inner')
# print("Number of features that are the same in the top 60% (ca. 100) of features for both xgboost and random forest: ", len(features_same_top3_60))

# # Save features that are the same in the top 60% of features for both xgboost and random forest as json file
# features_same3 = features_same_top3_60['feature'].tolist()
# with open('feature_selection/3rd_iteration/features_same_top_60.json', 'w') as f:
#     json.dump(features_same3, f)

# # VISUALIZATION FOR REPORT

# # XGBoost
# # Sort and select the top 100 important features
# xgboost_importance_top_100 = xgboost_importance.sort_values(by='importance', ascending=False).head(100)

# # Create a figure and main subplot
# fig, ax = plt.subplots(figsize=(14, 7))
# ax.bar(xgboost_importance['feature'], xgboost_importance['importance'])
# ax.bar(xgboost_importance['feature'][0], xgboost_importance['importance'][0], width=1.5, color='tab:blue')
# ax.set_xticks([])
# ax.set_title('Feature Importance Scores (XGBoost)')
# ax.set_xlabel('All Features')
# ax.set_ylabel('Importance Scores')

# # Add an inset subplot
# left, bottom, width, height = [0.565, 0.4, 0.4, 0.5]
# ax_inset = fig.add_axes([left, bottom, width, height])
# colors = ['green' if x in features_same3 else 'blue' for x in xgboost_importance_top_100['feature']]
# ax_inset.bar(xgboost_importance_top_100['feature'], xgboost_importance_top_100['importance'], color=colors)
# ax_inset.set_xticks([])
# ax_inset.set_title('Top 100 Features of First Iteration')
# ax_inset.set_xlabel('Features', fontsize=12)
# ax_inset.set_ylabel('Importance Scores', fontsize=12)

# # Create a custom legend
# legend_labels = ['Kept Features', 'Removed Features']
# legend_colors = ['green', 'blue']
# handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
# ax_inset.legend(handles, legend_labels, title='Feature Status')

# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig('plots_for_report/xgboost_importance_kept_features.png')
# plt.close()

# # Sort and select the top 100 important features
# random_forest_importance_top_100 = random_forest_importance.sort_values(by='importance', ascending=False).head(100)

# # Create a figure and main subplot
# fig, ax = plt.subplots(figsize=(14, 7))
# ax.bar(random_forest_importance['feature'], random_forest_importance['importance'])
# ax.set_xticks([])
# ax.set_title('Feature Importance Scores (Random Forest)')
# ax.set_xlabel('All Features')
# ax.set_ylabel('Importance Scores')

# # Add an inset subplot
# left, bottom, width, height = [0.565, 0.4, 0.4, 0.5]
# ax_inset = fig.add_axes([left, bottom, width, height])
# colors = ['green' if x in features_same3 else 'blue' for x in random_forest_importance_top_100['feature']]
# ax_inset.bar(random_forest_importance_top_100['feature'], random_forest_importance_top_100['importance'], color=colors)
# ax_inset.set_xticks([])
# ax_inset.set_title('Top 100 Features of First Iteration')
# ax_inset.set_xlabel('Features', fontsize=12)
# ax_inset.set_ylabel('Importance Scores', fontsize=12)

# # Create a custom legend
# legend_labels = ['Kept Features', 'Removed Features']
# legend_colors = ['green', 'blue']
# handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
# ax_inset.legend(handles, legend_labels, title='Feature Status')

# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig('plots_for_report/rf_importance_kept_features.png')
# plt.close()
