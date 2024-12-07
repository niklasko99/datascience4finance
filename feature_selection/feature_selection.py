# Feature Selection
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# This function performs feature importance analysis using XGBoost and Random Forest models across three iterations.
# It loads feature importance scores, visualizes them, identifies features with zero importance, compares feature
# rankings, and saves common important features to JSON files.
print("1ST ITERATION")

# 1ST ITERATION

# load feature importance scores (xgboost)
xgboost_importance = pd.read_csv('feature_selection/1st_iteration/xgboost_importance_1st_iteration.csv')
# other columns: feature, importance

# Plot graph (bar charts) to visualize feature importance scores for xgboost
plt.figure(figsize=(14, 7))
plt.bar(xgboost_importance['feature'], xgboost_importance['importance'])
plt.bar(xgboost_importance['feature'][0], xgboost_importance['importance'][0], width=1.5, color='tab:blue')
plt.axvline(x=0.3 * len(xgboost_importance), color='red', linestyle='--')
plt.xticks([])
plt.title('Feature Importance Scores (XGBoost)')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.tight_layout()
plt.savefig('feature_selection/1st_iteration/xgboost_importance_1st_iteration.png')
plt.close()

# load feature importance scores (random forest)
random_forest_importance = pd.read_csv('feature_selection/1st_iteration/rf_importance_1st_iteration.csv')
# other columns: feature, importance

# Plot graph (bar charts) to visualize feature importance scores for random forest
plt.figure(figsize=(14, 7))
plt.bar(random_forest_importance['feature'], random_forest_importance['importance'])
plt.axvline(x=0.3 * len(random_forest_importance), color='red', linestyle='--')
plt.xticks([])
plt.title('Feature Importance Scores (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.tight_layout()
plt.savefig('feature_selection/1st_iteration/rf_importance_1st_iteration.png')
plt.close()

# Find features that have an importance of 0
xgboost_importance_0 = xgboost_importance[xgboost_importance['importance'] == 0]
print("Number of features with importance 0 in xgboost: ", len(xgboost_importance_0))
random_forest_importance_0 = random_forest_importance[random_forest_importance['importance'] == 0]
print("Number of features with importance 0 in random forest: ", len(random_forest_importance_0))

# Find features that have an importance of 0 for both xgboost and random forest
features_0 = pd.merge(xgboost_importance_0, random_forest_importance_0, on='feature', how='inner')
print("Number of features with importance 0 for both xgboost and random forest: ", len(features_0))

# Find features whose difference in ranks (index) between xgboost and random forest is greater than 150
xgboost_importance['index'] = xgboost_importance.index
random_forest_importance['index'] = random_forest_importance.index
features_diff = pd.merge(xgboost_importance, random_forest_importance, on='feature', how='inner')
features_diff['diff'] = abs(features_diff['index_x'] - features_diff['index_y'])
features_diff = features_diff[features_diff['diff'] > 150]
print("Number of features whose difference in ranks between xgboost and random forest is greater than 150 (10% of features): ", len(features_diff))

# How many features are the same in the top 30% of features for both xgboost and random forest?
xgboost_importance = xgboost_importance.sort_values(by='importance', ascending=False)
random_forest_importance = random_forest_importance.sort_values(by='importance', ascending=False)
top_30_xgboost = xgboost_importance.head(int(0.3 * len(xgboost_importance)))
top_30_rf = random_forest_importance.head(int(0.3 * len(random_forest_importance)))
features_same_top_30 = pd.merge(top_30_xgboost, top_30_rf, on='feature', how='inner')
print("Number of features that are the same in the top 30% (ca. 450) of features for both xgboost and random forest: ", len(features_same_top_30))

# Save features that are the same in the top 30% of features for both xgboost and random forest as json file
features_same = features_same_top_30['feature'].tolist()
with open('feature_selection/1st_iteration/features_same_top_30.json', 'w') as f:
    json.dump(features_same, f)

print("\n2ND ITERATION")

# 2ND ITERATION

# load feature importance scores (xgboost)
xgboost_importance2 = pd.read_csv('feature_selection/2nd_iteration/xgboost_importance_2nd_iteration.csv')
# other columns: feature, importance

# Plot graph (bar charts) to visualize feature importance scores for xgboost
plt.figure(figsize=(14, 7))
plt.bar(xgboost_importance2['feature'], xgboost_importance2['importance'])
plt.axvline(x=0.6 * len(xgboost_importance2), color='red', linestyle='--')
plt.xticks([])
plt.title('Feature Importance Scores (XGBoost)')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.tight_layout()
plt.savefig('feature_selection/2nd_iteration/xgboost_importance_2nd_iteration.png')

# load feature importance scores (random forest)
random_forest_importance2 = pd.read_csv('feature_selection/2nd_iteration/rf_importance_2nd_iteration.csv')
# other columns: feature, importance

# Plot graph (bar charts) to visualize feature importance scores for random forest
plt.figure(figsize=(14, 7))
plt.bar(random_forest_importance2['feature'], random_forest_importance2['importance'])
plt.axvline(x=0.6 * len(random_forest_importance2), color='red', linestyle='--')
plt.xticks([])
plt.title('Feature Importance Scores (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.tight_layout()
plt.savefig('feature_selection/2nd_iteration/rf_importance_2nd_iteration.png')
plt.close()

# Find features that have an importance of 0
xgboost_importance2_0 = xgboost_importance2[xgboost_importance2['importance'] == 0]
print("Number of features with importance 0 in xgboost: ", len(xgboost_importance2_0))
random_forest_importance2_0 = random_forest_importance2[random_forest_importance2['importance'] == 0]
print("Number of features with importance 0 in random forest: ", len(random_forest_importance2_0))

# Find features that have an importance of 0 for both xgboost and random forest
features2_0 = pd.merge(xgboost_importance2_0, random_forest_importance2_0, on='feature', how='inner')
print("Number of features with importance 0 for both xgboost and random forest: ", len(features2_0))

# Find features whose difference in ranks (index) between xgboost and random forest is greater than 40
xgboost_importance2['index'] = xgboost_importance2.index
random_forest_importance2['index'] = random_forest_importance2.index
features_diff2 = pd.merge(xgboost_importance2, random_forest_importance2, on='feature', how='inner')
features_diff2['diff'] = abs(features_diff2['index_x'] - features_diff2['index_y'])
features_diff2 = features_diff2[features_diff2['diff'] > 40]
print("Number of features whose difference in ranks between xgboost and random forest is greater than 50 (10% of features): ", len(features_diff2))

# How many features are the same in the top 60% of features for both xgboost and random forest?
xgboost_importance2 = xgboost_importance2.sort_values(by='importance', ascending=False)
random_forest_importance2 = random_forest_importance2.sort_values(by='importance', ascending=False)
top_60_xgboost2 = xgboost_importance2.head(int(0.6 * len(xgboost_importance2)))
top_60_rf2 = random_forest_importance2.head(int(0.6 * len(random_forest_importance2)))
features_same_top2_60 = pd.merge(top_60_xgboost2, top_60_rf2, on='feature', how='inner')
print("Number of features that are the same in the top 60% (ca. 240) of features for both xgboost and random forest: ", len(features_same_top2_60))

# Save features that are the same in the top 60% of features for both xgboost and random forest as json file
features_same2 = features_same_top2_60['feature'].tolist()
with open('feature_selection/2nd_iteration/features_same_top_60.json', 'w') as f:
    json.dump(features_same2, f)

print("\n3RD ITERATION")

# 3RD ITERATION
# load feature importance scores (xgboost)
xgboost_importance3 = pd.read_csv('feature_selection/3rd_iteration/xgboost_importance_3rd_iteration.csv')
# other columns: feature, importance

# Plot graph (bar charts) to visualize feature importance scores for xgboost
plt.figure(figsize=(14, 7))
plt.bar(xgboost_importance3['feature'], xgboost_importance3['importance'])
plt.axvline(x=0.6 * len(xgboost_importance3), color='red', linestyle='--')
plt.xticks([])
plt.title('Feature Importance Scores (XGBoost)')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.tight_layout()
plt.savefig('feature_selection/3rd_iteration/xgboost_importance_3rd_iteration.png')

# load feature importance scores (random forest)
random_forest_importance3 = pd.read_csv('feature_selection/3rd_iteration/rf_importance_3rd_iteration.csv')
# other columns: feature, importance

# Plot graph (bar charts) to visualize feature importance scores for random forest
plt.figure(figsize=(14, 7))
plt.bar(random_forest_importance3['feature'], random_forest_importance3['importance'])
plt.axvline(x=0.6 * len(random_forest_importance3), color='red', linestyle='--')
plt.xticks([])
plt.title('Feature Importance Scores (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.tight_layout()
plt.savefig('feature_selection/3rd_iteration/rf_importance_3rd_iteration.png')
plt.close()

# Find features that have an importance of 0
xgboost_importance3_0 = xgboost_importance3[xgboost_importance3['importance'] == 0]
print("Number of features with importance 0 in xgboost: ", len(xgboost_importance3_0))
random_forest_importance3_0 = random_forest_importance3[random_forest_importance3['importance'] == 0]
print("Number of features with importance 0 in random forest: ", len(random_forest_importance3_0))

# Find features that have an importance of 0 for both xgboost and random forest
features3_0 = pd.merge(xgboost_importance3_0, random_forest_importance3_0, on='feature', how='inner')
print("Number of features with importance 0 for both xgboost and random forest: ", len(features3_0))

# Find features whose difference in ranks (index) between xgboost and random forest is greater than 15
xgboost_importance3['index'] = xgboost_importance3.index
random_forest_importance3['index'] = random_forest_importance3.index
features_diff3 = pd.merge(xgboost_importance3, random_forest_importance3, on='feature', how='inner')
features_diff3['diff'] = abs(features_diff3['index_x'] - features_diff3['index_y'])
features_diff3 = features_diff3[features_diff3['diff'] > 15]
print("Number of features whose difference in ranks between xgboost and random forest is greater than 20 (10% of features): ", len(features_diff3))

# How many features are the same in the top 60% of features for both xgboost and random forest?
xgboost_importance3 = xgboost_importance3.sort_values(by='importance', ascending=False)
random_forest_importance3 = random_forest_importance3.sort_values(by='importance', ascending=False)
top_60_xgboost3 = xgboost_importance3.head(int(0.6 * len(xgboost_importance3)))
top_60_rf3 = random_forest_importance3.head(int(0.6 * len(random_forest_importance3)))
features_same_top3_60 = pd.merge(top_60_xgboost3, top_60_rf3, on='feature', how='inner')
print("Number of features that are the same in the top 60% (ca. 100) of features for both xgboost and random forest: ", len(features_same_top3_60))

# Save features that are the same in the top 60% of features for both xgboost and random forest as json file
features_same3 = features_same_top3_60['feature'].tolist()
with open('feature_selection/3rd_iteration/features_same_top_60.json', 'w') as f:
    json.dump(features_same3, f)

# VISUALIZATION FOR REPORT

# XGBoost
# Sort and select the top 100 important features
xgboost_importance_top_100 = xgboost_importance.sort_values(by='importance', ascending=False).head(100)

# Create a figure and main subplot
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(xgboost_importance['feature'], xgboost_importance['importance'])
ax.bar(xgboost_importance['feature'][0], xgboost_importance['importance'][0], width=1.5, color='tab:blue')
ax.set_xticks([])
ax.set_title('Feature Importance Scores (XGBoost)')
ax.set_xlabel('All Features')
ax.set_ylabel('Importance Scores')

# Add an inset subplot
left, bottom, width, height = [0.565, 0.4, 0.4, 0.5]
ax_inset = fig.add_axes([left, bottom, width, height])
colors = ['green' if x in features_same3 else 'blue' for x in xgboost_importance_top_100['feature']]
ax_inset.bar(xgboost_importance_top_100['feature'], xgboost_importance_top_100['importance'], color=colors)
ax_inset.set_xticks([])
ax_inset.set_title('Top 100 Features of First Iteration')
ax_inset.set_xlabel('Features', fontsize=12)
ax_inset.set_ylabel('Importance Scores', fontsize=12)

# Create a custom legend
legend_labels = ['Kept Features', 'Removed Features']
legend_colors = ['green', 'blue']
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
ax_inset.legend(handles, legend_labels, title='Feature Status')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('plots_for_report/xgboost_importance_kept_features.png')
plt.close()

# Sort and select the top 100 important features
random_forest_importance_top_100 = random_forest_importance.sort_values(by='importance', ascending=False).head(100)

# Create a figure and main subplot
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(random_forest_importance['feature'], random_forest_importance['importance'])
ax.set_xticks([])
ax.set_title('Feature Importance Scores (Random Forest)')
ax.set_xlabel('All Features')
ax.set_ylabel('Importance Scores')

# Add an inset subplot
left, bottom, width, height = [0.565, 0.4, 0.4, 0.5]
ax_inset = fig.add_axes([left, bottom, width, height])
colors = ['green' if x in features_same3 else 'blue' for x in random_forest_importance_top_100['feature']]
ax_inset.bar(random_forest_importance_top_100['feature'], random_forest_importance_top_100['importance'], color=colors)
ax_inset.set_xticks([])
ax_inset.set_title('Top 100 Features of First Iteration')
ax_inset.set_xlabel('Features', fontsize=12)
ax_inset.set_ylabel('Importance Scores', fontsize=12)

# Create a custom legend
legend_labels = ['Kept Features', 'Removed Features']
legend_colors = ['green', 'blue']
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
ax_inset.legend(handles, legend_labels, title='Feature Status')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('plots_for_report/rf_importance_kept_features.png')
plt.close()
