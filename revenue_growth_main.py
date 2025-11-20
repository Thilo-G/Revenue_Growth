#Press strg I to chat with copilot
# ctrl + enter: gitub copilot suggestion
# ctrl + alt + I: open chat view
#debugging F5
#Autoformation Shift+Alt+F
# """ """ for multiline comments
#==============================================================================
# 1. Importing the data files & Data Exploration
#==============================================================================
# Importing the libraries
import pandas as pd
import openpyxl
import numpy as np

# Quarterly Revenue
file1_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\RRR_FinancialImplication\\Data\\2025-0319a-TK-quarterlyrevenue-collection_Python.xlsx"

try:
    # Read the revenue file
    df_revenue = pd.read_excel(file1_path, header=None);  # Load with no header
    print("Revenue-Daten erfolgreich eingelesen.")
    print("Shape:", df_revenue.shape)

except Exception as e:
    #Exception is the base class for all exceptions (FilenotFoundError, ValueError, etc.)
    #As e saves the "exception"
    print(f"Fehler beim Einlesen der Dateien: {e}")

# Normalize strings to ensure case-insensitive and whitespace-consistent comparison
def normalize_strings(strings):
    return set(s.upper().strip() for s in strings)
normalized_firm_names_revenue = normalize_strings(set(df_revenue.iloc[0].dropna().astype(str).str.strip()));

# Print the number of unique firms in each dataset
print(f"Number of Unique Firms in Revenue File: {len(normalized_firm_names_revenue)}")

#==============================================================================
# 2. Data Wrangling
#==============================================================================

###################
# 2.1 Revenue Data#
###################

# Extract the first two rows for processing headers
header_rows_rev = df_revenue.iloc[:2];
data_rows_rev = df_revenue.iloc[2:];  # Remaining rows are the actual data

# Combine the first two rows to create a single-level column header
combined_headers = header_rows_rev.apply(lambda x: x.str.strip() if x.dtype == "object" else x);
column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0);
data_rows_rev.columns = column_headers;  # Assign proper headers to the data rows

# Rename the first column to "Date"
data_rows_rev.rename(columns={data_rows_rev.columns[0]: 'Date'}, inplace=True);

# Reset the index for the data rows
df_revenue = data_rows_rev.reset_index(drop=True);

### replace empty cells with NaN
df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True);



#Add totel and RRR and shares
# Remove duplicate columns
df_revenue = df_revenue.loc[:, ~df_revenue.columns.duplicated()];
# Identify columns related to "New Customer" and "Returning Customer"
new_customer_cols = [col for col in df_revenue.columns if '#New_Customers' in col];
returning_customer_cols = [col for col in df_revenue.columns if '#Returning_Customers' in col];

"""
code is slow not optimal-> try to improve
"""

# Create new columns for Total Revenue, share, RRR, revenue growth, and revenue_new_growth
for new_col, ret_col in zip(new_customer_cols, returning_customer_cols):
    # Generate new column names
    total_revenue_col = new_col.replace('#New_Customers', '#Total_Revenue')
    share_ret_revenue_col = ret_col.replace('#Returning_Customers', '#Share_Ret_Revenue')
    rrr_col = ret_col.replace('#Returning_Customers', '#RRR')
    revenue_growth_col = ret_col.replace('#Returning_Customers', '#Revenue_Growth')
    acq_rate_col = new_col.replace('#New_Customers', '#Acq_Rate')
    gm_col = new_col.replace('#New_Customers', '#Growth_Mix')
    growth_indicator_col = new_col.replace('#New_Customers', '#Growth_Indicator')

    # Add Total Revenue
    df_revenue[total_revenue_col] = df_revenue[new_col] + df_revenue[ret_col]

    # Calculate Share of Returning Revenue
    #### have some 0 df_revenue[share_ret_revenue_col] = df_revenue[ret_col] / df_revenue[total_revenue_col].replace(0, np.nan)
    df_revenue[share_ret_revenue_col] = df_revenue[ret_col] / df_revenue[total_revenue_col].replace(0, np.nan)

    # Calculate RRR
    df_revenue[rrr_col] = df_revenue[ret_col] / df_revenue[total_revenue_col].shift(1).replace(0, np.nan)

    # Calculate Revenue Growth
    df_revenue[revenue_growth_col] = (
        (df_revenue[total_revenue_col] - df_revenue[total_revenue_col].shift(1)) / df_revenue[total_revenue_col].shift(1)
    )

    # Calculate Revenue New Growth
    df_revenue[acq_rate_col] = (df_revenue[new_col] / df_revenue[new_col].shift(1))

    # Calculate Growth Mix
    df_revenue[gm_col] = (df_revenue[ret_col] -  df_revenue[ret_col].shift(1)) / (df_revenue[total_revenue_col] -  df_revenue[total_revenue_col].shift(1))

    # calculate Growth Indicator
    df_revenue[growth_indicator_col] = df_revenue[revenue_growth_col].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    # Reorder columns to place new columns next to Returning Customers
    cols = list(df_revenue.columns) #Get current column order
    ret_col_index = cols.index(ret_col) # Find index of the Returning Customer column
    # Remove and reinsert new columns next to the Returning Customer column
    cols.remove(total_revenue_col)
    cols.remove(share_ret_revenue_col)
    cols.remove(rrr_col)
    cols.remove(revenue_growth_col)
    cols.remove(acq_rate_col)
    cols.remove(gm_col)
    cols.remove(growth_indicator_col)

    cols.insert(ret_col_index + 1, total_revenue_col)
    cols.insert(ret_col_index + 2, share_ret_revenue_col)
    cols.insert(ret_col_index + 3, rrr_col)
    cols.insert(ret_col_index + 4, revenue_growth_col)
    cols.insert(ret_col_index + 5, acq_rate_col)
    cols.insert(ret_col_index + 6, gm_col)
    cols.insert(ret_col_index + 7, growth_indicator_col)
    df_revenue = df_revenue[cols]


#==============================================================================
# 3. First Analysis of seperated data
#==============================================================================

###################
# 3.1 Revenue Data#
###################

import seaborn as sns
import matplotlib.pyplot as plt

"""
make it more efficient, create function
"""


# Define the columns of interest
columns_of_interest = ["#RRR", "#Revenue_Growth", "#Acq_Rate", "#Share_Ret_Revenue", "#Growth_Mix"]

# Ensure the Date column is in datetime format for proper sorting
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'])

# Iterate over each metric to generate boxplots and histograms
for metric in columns_of_interest:
    # Extract relevant columns for the current metric
    metric_cols = [col for col in df_revenue.columns if metric in col]

    # Melt the dataset for easier plotting with seaborn
    df_metric_melted = df_revenue.melt(
        id_vars=['Date'], 
        value_vars=metric_cols, 
        var_name='Firm', 
        value_name=metric
    )

    # Remove the top 4 highest values correctly for display purposes (not statistics)
    df_filtered = df_metric_melted.copy()
    df_filtered = df_filtered[df_filtered[metric] < df_filtered[metric].nlargest(4).min()]  # Exclude top 4 values
    # Remove the lowest 4 values for display purposes (not statistics)
    df_filtered = df_filtered[df_filtered[metric] > df_filtered[metric].nsmallest(4).max()]  # Exclude lowest 4 values

    # Compute additional statistics
    q25, q50, q75 = df_metric_melted[metric].quantile([0.25, 0.50, 0.75])

    # Ensure the x-axis (dates) remain sorted
    df_filtered = df_filtered.sort_values(by="Date")

    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Date', y=metric, data=df_filtered, color='skyblue')

    # Customize the chart
    plt.xlabel('Date')
    plt.ylabel(metric.replace("#", ""))  # Remove '#' for cleaner labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Print summary statistics including quantiles
    print(f"\nSummary statistics for {metric.replace('#', '')}:")
    print(df_metric_melted[metric].describe().loc[['mean', 'std', 'min', 'max']])
    print(f"25th Percentile (Q1): {q25:.2f}")
    print(f"50th Percentile (Q2): {q50:.2f}")
    print(f"75th Percentile (Q3): {q75:.2f}")

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df_filtered[metric], bins=30, kde=True, color='skyblue')

    # Customize the plot
    plt.xlabel(metric.replace("#", ""))
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

"""
Summary statistics for RRR (Excluding Top 4):
mean    0.799884
std     0.312226
min     0.001062
max     3.889442
Name: #RRR, dtype: float64
25th Percentile (Q1): 0.61
75th Percentile (Q3): 0.95

Summary statistics for Revenue_Growth (Excluding Top 4):
mean    0.080728
std     0.410411
min    -0.998914
max     5.636012
Name: #Revenue_Growth, dtype: float64
25th Percentile (Q1): -0.08
75th Percentile (Q3): 0.17

Summary statistics for Revenue_New_Growth:
mean     0.287643
std      0.411074
min      0.000024
max     16.130858
Name: #Acq_Rate, dtype: float64
25th Percentile (Q1): 0.10
75th Percentile (Q3): 0.37
"""
