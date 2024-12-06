import pandas as pd
import fastparquet
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import numpy as np
import warnings
import random
warnings.filterwarnings("ignore")

# https://data-explorer.oecd.org/vis?lc=en&fs[0]=Topic%2C1%7CEconomy%23ECO%23%7CLeading%20indicators%23ECO_LEA%23&pg=0&fc=Topic&bp=true&snb=1&vw=ov&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_STES%40DF_CLI&df[ag]=OECD.SDD.STES&df[vs]=4.1&pd=2013-01%2C2024-11&dq=G20%2BUSA.M.BCICP...AA...H&to[TIME_PERIOD]=false
def clean_external_data():
    bc_indicators = pd.read_csv("data/business_confidence_indicators.csv")
    bc_indicators = bc_indicators[['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]
    bc_indicators["OBS_VALUE"] = bc_indicators["OBS_VALUE"].astype('float32')
    bc_indicators['period'] = pd.to_datetime(bc_indicators['TIME_PERIOD'])
    bc_indicators = bc_indicators.drop(columns=['TIME_PERIOD'])
    grouped_data = bc_indicators.groupby('REF_AREA')

    # Create a list to store data tables
    business_confidence_tables = []

    for ref_area in grouped_data.groups.keys():
        data_table = grouped_data.get_group(ref_area).copy()
        data_table.rename(columns={'OBS_VALUE': f'{ref_area}_business_confidence'}, inplace=True)
        data_table.reset_index(drop=True, inplace=True)
        data_table.drop(columns=['REF_AREA'], inplace=True)
        business_confidence_tables.append(data_table)

    # First sort tables by period
    for table in business_confidence_tables:
        table.sort_values(by='period', inplace=True)
        table.reset_index(drop=True, inplace=True)

    # Merge all data tables in business_confidence_tables into one by merging on the 'period' column
    bc_indicators_merged = pd.merge(business_confidence_tables[0], business_confidence_tables[1], on='period', how='inner')

    return bc_indicators_merged

def get_companies_with_full_observations(data):

    # Step 1: Extract year and prepare company data
    company_data = data[["period", "cik"]].copy()  # Copy to avoid modifying the original dataframe
    company_data["year"] = company_data["period"].dt.year  # Extract year from 'period'
    company_data.drop(columns=["period"], inplace=True)  # Drop 'period' column
    company_data.drop_duplicates(inplace=True)  # Remove duplicates

    # Step 2: Identify companies with entries for all mandatory years (2014–2023)
    required_years = set(range(2014, 2024))  # Mandatory years (2014–2023)

    # Group by CIK and aggregate the years present for each company
    company_years = company_data.groupby("cik")["year"].apply(set)

    # Filter companies that have entries for all required years
    valid_companies = company_years[company_years.apply(lambda x: required_years.issubset(x))].index

    # Step 3: Filter the original data for only valid companies
    filtered_data = company_data[company_data["cik"].isin(valid_companies)]  # Retain only valid companies

    # Step 4: Validate that the number of companies is the same across all required years
    result = filtered_data[filtered_data["year"].between(2014, 2023)]  # Only focus on required years
    yearly_counts = result.groupby("year")["cik"].nunique()  # Count unique companies by year

    print(yearly_counts)  # Print the count of companies per year (2014–2023)

    # Filter the reshaped data for the valid companies
    reduced_data = data[data["cik"].isin(valid_companies)]

    return reduced_data


def add_indicator_variables(df, missing_threshold=0.3):
    # Add indicator variables for columns where at least 30% of values are missing
    for column in df.columns:
        missing_percentage = df[(df["period"].dt.year < 2021)][column].isna().mean()
        if missing_percentage >= missing_threshold:
            # Create an indicator variable
            indicator_column_name = f"{column}_missing"
            df[indicator_column_name] = df[column].isna().astype(int)
    return df


# def remove_outliers_iqr(df):
#     # Get numeric columns
#     numeric_columns = df.select_dtypes(include=['float']).columns

#     df["year"] = df["period"].dt.year

#     training = df[df["year"] < 2021]

#     for column in numeric_columns:
#         Q1 = training[column].quantile(0.25)  # First quartile
#         Q3 = training[column].quantile(0.75)  # Third quartile
#         IQR = Q3 - Q1  # Interquartile range
        
#         # Define bounds
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
        
#         # Replace outliers with NaN
#         df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan

#     # Drop the year column
#     df.drop(columns=["year"], inplace=True) 
#     return df


def create_lagged_and_percentage_change_features(dataframe, imputed = True):
    """
    Creates lagged features (shifted by 1) and percentage changes for all float columns in the dataframe.

    Parameters:
        dataframe (pd.DataFrame): Input dataframe with at least one float64 column.
    
    Returns:
        pd.DataFrame: A dataframe with original, lagged, and percentage change features.
    """
 # Sort by ticker and period to ensure correct temporal order
    dataframe = dataframe.sort_values(by=["ticker", "period"]).reset_index(drop=True)

    # Identify float columns
    float_columns = dataframe.select_dtypes(include=['float']).columns
    
    # Create lagged features (shifted by 1 within each ticker group)
    lagged_features = dataframe.groupby("ticker")[float_columns].shift(1).add_suffix('_lag1')
    
    # Combine the original dataframe with new features
    combined_dataframe = pd.concat([dataframe, lagged_features], axis=1)

    # Calculate percentage change
    for column in float_columns:
        lagged_col = f"{column}_lag1"
        change_col = f"{column}_change"
        
        # Use np.where to handle the condition efficiently
        combined_dataframe[change_col] = np.where(
            combined_dataframe[lagged_col] == 0,  # Condition: lagged value is zero
            combined_dataframe[column],           # If True: use current value
            (combined_dataframe[column] - combined_dataframe[lagged_col]) / combined_dataframe[lagged_col]  # Else: calculate percentage change
        )

    if imputed:
        # Fill NaN values for lagged and change columns
        mask = combined_dataframe.columns.str.endswith('_lag1') | combined_dataframe.columns.str.endswith('_change')
        combined_dataframe.loc[:, mask] = combined_dataframe.loc[:, mask].fillna(0)
    
    return combined_dataframe


def plot_random_variable_distributions(dataframe, rows=2, cols=10, figsize=(20, 8)):
    """
    Plots the distributions of 20 randomly selected floating-point variables in a grid layout.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        rows (int): Number of rows in the plot grid.
        cols (int): Number of columns in the plot grid.
        figsize (tuple): Size of the entire figure.
    """
    # Select only floating-point columns
    float_columns = dataframe.select_dtypes(include=['float']).columns

    # Randomly select up to 20 variables
    selected_variables = random.sample(list(float_columns), min(20, len(float_columns)))

    # Plot distributions
    num_vars = len(selected_variables)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, constrained_layout=True)

    for i, ax in enumerate(axes.flatten()):
        if i < num_vars:
            variable = selected_variables[i]
            ax.hist(dataframe[variable], bins=20, alpha=0.7, color='blue', edgecolor='black')
            # ax.set_title(variable, fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
        else:
            ax.axis('off')

    plt.suptitle('Distributions of Randomly Selected Variables', fontsize=16)
    plt.show()


def drop_highly_correlated_features(dataframe, threshold=0.9, method='spearman'):
    """
    Identifies and collects features with high correlation (above a given threshold)
    based on the Spearman correlation matrix in an iterative manner.
    Prioritizes dropping features with `lag_1` or `_change` suffixes.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        threshold (float): Correlation threshold above which features are considered highly correlated.
        method (str): Correlation method ('pearson', 'spearman', etc.).

    Returns:
        list: A list of features to drop due to high correlation.
    """
    # To avoid information leakage, we will only look at the data before 2021
    dataframe = dataframe[(dataframe["period"].dt.year < 2021)]

    # Select only floating-point variables
    float_columns = dataframe.select_dtypes(include=['float']).columns
    df_floats = dataframe[float_columns]

    # Compute the correlation matrix
    corr_matrix = df_floats.corr(method=method).abs()

    # Collect features to drop iteratively
    to_drop = set()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if corr_matrix.iloc[i, j] > threshold:
                col1 = corr_matrix.index[i]
                col2 = corr_matrix.columns[j]
                
                # Prioritize dropping based on suffix
                if col1.endswith('_lag1') or col1.endswith('_change'):
                    to_drop.add(col1)
                elif col2.endswith('_lag1') or col2.endswith('_change'):
                    to_drop.add(col2)
                else:
                    # Default behavior if no suffix preference
                    if col1 not in to_drop and col2 not in to_drop:
                        to_drop.add(col2)

    # Handle `_missing` variables
    missing_to_drop = set()
    for feature in to_drop:
        if feature + "_missing" in dataframe.columns:
            missing_to_drop.add(feature + "_missing")
    to_drop.update(missing_to_drop)

    print(f"Number of features to drop: {len(to_drop)}")

    return list(to_drop)


def one_hot_encode_year(df):
    df["year"] = df["period"].dt.year
    # now we want to add an indicator_variable for labeling the observations with 1 where the year is less than 2021
    df["pre_covid"] = (df["year"] < 2020).astype(int)
    df = pd.get_dummies(df, columns=["year"], drop_first=True)
    return df


def main():
    # Load the dataframe
    data = pd.read_csv("data/annual_filings.csv")
    data.drop(columns=["Unnamed: 0"], inplace=True)

    data.drop(columns=["is_amended", "amendment_adsh", "form", "version", "accepted", "start", "end", "computed"], inplace=True)

    # make period to datetime format
    data["period"] = pd.to_datetime(data["period"])

    # Filter data for the years 2013–2024
    data = data[(data["period"].dt.year >= 2013) & (data["period"].dt.year <= 2024)]

    data = get_companies_with_full_observations(data)

    # Reshape the data into the desired format
    reshaped_data = data.pivot(index=['adsh', 'cik', 'sic', 'period', 'ticker'], 
                            columns='tag', 
                            values='reported_figure')

    # Remove the column index name ('tag')
    reshaped_data.columns.name = None

    # Reset the index to flatten the structure
    reshaped_data.reset_index(inplace=True)

    # Percentage of missing values
    print(f"Initial percentage of missing values: {reshaped_data.isna().sum().sum() / (reshaped_data.shape[0] * reshaped_data.shape[1])}")

    # NetIncomeLossAvailableToCommonStockholdersDiluted (Diluted NI Available to Common Stockerholder) and WeightedAverageNumberOfDilutedSharesOutstanding 
    # (Diluted Average Shares Outstanding) are the two columns that are important for calculating the EPS. Moreover, for standardizing the variable Assets is important
    # Drop the values that are missing in these two columns
    reshaped_data = reshaped_data.dropna(subset=["NetIncomeLossAvailableToCommonStockholdersDiluted", "WeightedAverageNumberOfDilutedSharesOutstanding", "Assets"])

    # also drop the rows where the Assets is 0 or WeightedAverageNumberOfDilutedSharesOutstanding is 0
    reshaped_data = reshaped_data[(reshaped_data["Assets"] != 0) & (reshaped_data["WeightedAverageNumberOfDilutedSharesOutstanding"] != 0)]

    reshaped_data = get_companies_with_full_observations(reshaped_data)

    # get the percentage of missing values in the reshaped data and plot the distribution of percentages as a barplot
    missing_values = reshaped_data.isna().sum() / reshaped_data.shape[0]
    missing_values = missing_values.sort_values(ascending=False)

    # Retrieve the names of columns where the percentage of missing values is greater than 0.6
    columns_to_drop = missing_values[missing_values > 0.6].index

    reshaped_data = reshaped_data.drop(columns=columns_to_drop)

    print(f"Percentage of missing values after dropping high missing variables: {reshaped_data.isna().sum().sum() / (reshaped_data.shape[0] * reshaped_data.shape[1])}")

    # group by the ticker and find out what the percentage of missing values is. If the percentage is higher than 0.5, drop the ticker/company
    missing_values = reshaped_data.groupby("ticker").apply(lambda x: x.isna().sum().sum() / (x.shape[0] * x.shape[1]))
    valid_tickers = missing_values[missing_values <= 0.5].index

    # Filter the reshaped data for the valid tickers
    reshaped_data = reshaped_data[reshaped_data["ticker"].isin(valid_tickers)]

    print(f"Percentage of missing values after dropping high missing companies: {reshaped_data.isna().sum().sum() / (reshaped_data.shape[0] * reshaped_data.shape[1])}")

    # Number of companies in the filtered dataset
    print(f"Number of observed companies: {reshaped_data.groupby('ticker')['ticker'].nunique().sum()}")

    # Calculate the EPS for each company
    reshaped_data["EPS"] = reshaped_data["NetIncomeLossAvailableToCommonStockholdersDiluted"] / reshaped_data["WeightedAverageNumberOfDilutedSharesOutstanding"]

    # sanity check: are the values correct for apple: YES! (source: https://finance.yahoo.com/quote/AAPL/financials/)
    print(reshaped_data.loc[reshaped_data["ticker"] == "aapl", ["EPS", "period"]])

    # Ensure data is sorted by ticker and period
    reshaped_data = reshaped_data.sort_values(by=["ticker", "period"]).reset_index(drop=True)

    # Shift EPS to get next year's EPS
    reshaped_data["EPS+1"] = reshaped_data.groupby("ticker")["EPS"].shift(-1)

    # Drop the rows with missing EPS+1 values
    reshaped_data.dropna(subset=["EPS+1"], inplace=True)

    # Calculate EPS change
    reshaped_data["EPS_change"] = reshaped_data["EPS+1"] - reshaped_data["EPS"]

    # Rolling window average (minimum 1 year for early cases)
    reshaped_data["Avg_EPS_change"] = (
        reshaped_data.groupby("ticker")["EPS_change"]
        .rolling(window=4, min_periods=1)  # Allow partial windows initially
        .mean()
        .reset_index(level=0, drop=True)  # Reset index to align with original dataframe
    )

    # Compute detrended EPS
    reshaped_data["Detrended_EPS"] = reshaped_data["EPS_change"] - reshaped_data["Avg_EPS_change"]

    # Binary dependent variable
    reshaped_data["y"] = (reshaped_data["Detrended_EPS"] >= 0).astype(int)

    # Drop columns that are no longer needed
    reshaped_data.drop(columns=["EPS+1", "EPS_change", "Avg_EPS_change", "Detrended_EPS"], inplace=True)

    # Reset index
    reshaped_data.reset_index(inplace=True, drop=True)

    reshaped_data = add_indicator_variables(reshaped_data, missing_threshold=0.3)

    reshaped_data2 = reshaped_data.copy()
    # Get the columns which have float64 as datatype and divide them by the Assets column, but Assets column should not be divided by itself
    columns = reshaped_data.select_dtypes(include=['float64']).columns
    columns = columns.drop("Assets")
    columns = columns.drop("EPS")
    reshaped_data[columns] = reshaped_data[columns].div(reshaped_data["Assets"], axis=0)
    reshaped_data2[columns] = reshaped_data2[columns].div(reshaped_data2["Assets"], axis=0)

    # Drop outliers
    #reshaped_data = remove_outliers_iqr(reshaped_data)

    # Now we will compute the missing values. First we will impute the missing values with the median of the column group by the ticker. 
    # The median is based on the values smaller than 2021 to avoid information leakage.
    reshaped_data = reshaped_data.groupby("ticker").apply(lambda x: x.fillna(x[(x["period"].dt.year < 2021)].median())).reset_index(level=0)

    print(f"percentage of missing values after company specific imputation: {reshaped_data.isna().sum().sum() / (reshaped_data.shape[0] * reshaped_data.shape[1])}")

    # Now i am imputing the remaining missing values with the median of the respective industry in the respective year
    reshaped_data = reshaped_data.groupby(["sic", "period"]).apply(
        lambda group: group.assign(
            **{col: group[col].fillna(group[col].median()) for col in group.select_dtypes(include=['float']).columns}
        )
    ).reset_index(drop=True)

    print(f"percentage of missing values after industry and year specific imputation: {reshaped_data.isna().sum().sum() / (reshaped_data.shape[0] * reshaped_data.shape[1])}")

    # Now i am imputing the remaining missing values with 0
    reshaped_data.fillna(0, inplace=True)

    print(f"percentage of missing values after 0 filling: {reshaped_data.isna().sum().sum() / (reshaped_data.shape[0] * reshaped_data.shape[1])}")


    # now merge the external data with the reshaped data. Both have the period column. Thereby the respective values in the external data should 
    # be added to the reshaped data where the year and the month aligns with the period column in reshaped data
    # Extract year and month for merging
    reshaped_data['year'] = reshaped_data['period'].dt.year
    reshaped_data['month'] = reshaped_data['period'].dt.month

    external = clean_external_data()
    external['year'] = external['period'].dt.year
    external['month'] = external['period'].dt.month

    # Merge on year and month
    reshaped_data = pd.merge(
        reshaped_data,
        external[['year', 'month', 'G20_business_confidence', 'USA_business_confidence']],
        on=['year', 'month'],
        how='left'
    )

    reshaped_data2['year'] = reshaped_data2['period'].dt.year
    reshaped_data2['month'] = reshaped_data2['period'].dt.month

    # Merge on year and month
    reshaped_data2 = pd.merge(
        reshaped_data2,
        external[['year', 'month', 'G20_business_confidence', 'USA_business_confidence']],
        on=['year', 'month'],
        how='left'
    )


   # Drop auxiliary columns
    reshaped_data = reshaped_data.drop(columns=['year', 'month'])
    reshaped_data2 = reshaped_data2.drop(columns=['year', 'month'])

    # Create lagged and percentage change features
    reshaped_data = create_lagged_and_percentage_change_features(reshaped_data)
    reshaped_data2 = create_lagged_and_percentage_change_features(reshaped_data2, imputed=False)

    # Example usage for determining the distribution of random variables:
    plot_random_variable_distributions(reshaped_data)

    # Determining high correlated features:
    high_corr_features = drop_highly_correlated_features(reshaped_data, threshold=0.9, method='spearman')

    reshaped_data.drop(columns=high_corr_features, inplace=True)
    reshaped_data2.drop(columns=high_corr_features, inplace=True)

    reshaped_data = one_hot_encode_year(reshaped_data)
    reshaped_data2 = one_hot_encode_year(reshaped_data2)

    # Split the data into train and test
    train_imputed = reshaped_data[(reshaped_data["period"].dt.year < 2021)]
    test_imputed = reshaped_data[(reshaped_data["period"].dt.year >= 2021)]

    # Split the data into train and test, this is needed for the robustness check to check if our imputation strategy is correct
    train_original = reshaped_data2[(reshaped_data2["period"].dt.year < 2021)]
    test_original = reshaped_data2[(reshaped_data2["period"].dt.year >= 2021)]


    train_imputed.to_csv("data/train_imputed.csv", index=False)
    test_imputed.to_csv("data/test_imputed.csv", index=False)

    train_original.to_csv("data/train_original.csv", index=False)
    test_original.to_csv("data/test_original.csv", index=False)

if __name__ == "__main__":
    main()