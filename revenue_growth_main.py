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
file1_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\Revenue_Growth\\RG_Data\\Firms-Quarterly-RevenueGrowth-2025-11-21-Python.xlsx"
file2_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\Revenue_Growth\\RG_Data\\Firms-Industry-2025-11-21-Python.xlsx"
try:
    # Read the revenue file
    df_revenue = pd.read_excel(file1_path, header=None);  # Load with no header
    print("Revenue-Daten erfolgreich eingelesen.")
    print("Shape:", df_revenue.shape)

    # Read the industry file
    df_industry = pd.read_excel(file2_path, header=None);  # Load with no header
    print("Industry-Daten erfolgreich eingelesen.")
    print("Shape:", df_industry.shape)

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


# Use the first row as header for industry dataframe
df_industry.columns = df_industry.iloc[0]
df_industry = df_industry.iloc[1:].copy()

# Firms before filter
n_firms_industry_before = df_industry["ID"].nunique()
print("Number of Firms in Industry (before filter):", n_firms_industry_before)

# Keep only firms where gics_sub_industry_name is NOT NA
df_industry = df_industry[df_industry["gics_sub_industry_name"].notna()].copy()

n_industries_before = df_industry['gics_industry_name'].nunique(dropna=True)
print("Number of Industries:", n_industries_before)

# After removing rows with missing sub-industry, exclude sub-industries with fewer than 3 firms
if 'gics_sub_industry_name' not in df_industry.columns:
    raise KeyError("Expected column 'gics_sub_industry_name' in df_industry")

# raw counts per sub-industry (before threshold)
sub_counts_raw = df_industry['gics_sub_industry_name'].value_counts(dropna=True)

# keep only sub-industries with at least 3 firms
min_firms_per_sub = 3
keep_subs = sub_counts_raw[sub_counts_raw >= min_firms_per_sub].index
df_industry = df_industry[df_industry['gics_sub_industry_name'].isin(keep_subs)].copy()

# Firms after applying sub-industry size threshold
n_firms_industry_after = df_industry["ID"].nunique()
print("Number of Firms in Industry (after filter & size threshold):", n_firms_industry_after)

# Normalized set of valid IDs for the Revenue dataset (only firms in kept sub-industries)
valid_ids_norm = set(df_industry["ID"].astype(str).str.upper().str.strip())

# Select columns: always keep the date column (index 0) + all firms with valid sub-industry
cols_to_keep = [0]  # Keep the DATES column
for col_idx, val in df_revenue.iloc[0].items():
    if col_idx == 0 or pd.isna(val):
        continue
    firm_id_norm = str(val).upper().strip()
    if firm_id_norm in valid_ids_norm:
        cols_to_keep.append(col_idx)

df_revenue = df_revenue.iloc[:, cols_to_keep].copy()

# Counrt unique firms in revenue data after filtering
normalized_firm_names_revenue_after = normalize_strings(
    set(df_revenue.iloc[0].dropna().astype(str).str.strip())
)
print("Number of Firms in Revenue (after filter & size threshold):", len(normalized_firm_names_revenue_after))



#==============================================================================
# 2. Data Wrangling
#==============================================================================

###################
# 2.1 Revenue Data#
###################

# Extract the first two rows for processing headers
# Take explicit copies to avoid SettingWithCopyWarning when modifying
header_rows_rev = df_revenue.iloc[:2].copy()
data_rows_rev = df_revenue.iloc[2:].copy()  # Remaining rows are the actual data

# Combine the first two rows to create a single-level column header
combined_headers = header_rows_rev.apply(lambda x: x.str.strip() if x.dtype == "object" else x);
column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0);
data_rows_rev.columns = column_headers;  # Assign proper headers to the data rows

# Rename the first column to "Date" (avoid inplace to prevent chained-assignment warnings)
data_rows_rev = data_rows_rev.rename(columns={data_rows_rev.columns[0]: 'Date'})

# Reset the index for the data rows
df_revenue = data_rows_rev.reset_index(drop=True);

### replace empty cells with NaN
df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True);



# Add total and RRR and shares (vectorized & order-preserving)
# Remove duplicate columns
df_revenue = df_revenue.loc[:, ~df_revenue.columns.duplicated()]

# Identify columns related to "New Customer" and "Returning Customer"
new_customer_cols = [col for col in df_revenue.columns if '#New_Customers' in col]
returning_customer_cols = [col for col in df_revenue.columns if '#Returning_Customers' in col]

"""
Optimizations applied:
- Pair returning/new columns by their shared base name (robust to ordering differences).
- Compute all derived series in memory (dict of Series) and concat once to avoid repeated
  DataFrame re-allocation.
- Build final column order in a single pass to avoid repeated removes/inserts.
"""

# Helper to get base name for matching (strip the suffix marker)
def _base_name(col_name: str) -> str:
    return col_name.replace('#New_Customers', '').replace('#Returning_Customers', '').strip('. ')

# Build lookup maps
new_map = { _base_name(col): col for col in new_customer_cols }
ret_map = { _base_name(col): col for col in returning_customer_cols }

# Preserve original column order
original_cols = list(df_revenue.columns)

new_cols_data = {}
final_cols = []

for col in original_cols:
    # Append the existing column
    final_cols.append(col)

    # If this is a returning-customer column and we have a matching new-customer column, compute derived columns
    if col in returning_customer_cols:
        base = _base_name(col)
        new_col = new_map.get(base)
        if new_col is None:
            # no matching new-customer column; skip
            continue

        # derived column names
        total_revenue_col = new_col.replace('#New_Customers', '#Total_Revenue')
        #share_ret_revenue_col = col.replace('#Returning_Customers', '#Share_Ret_Revenue')
        rrr_col = col.replace('#Returning_Customers', '#RRR')
        revenue_growth_col = col.replace('#Returning_Customers', '#Revenue_Growth')
        acq_rate_col = new_col.replace('#New_Customers', '#Acq_Rate')
        #gm_col = new_col.replace('#New_Customers', '#Growth_Mix')
        #growth_indicator_col = new_col.replace('#New_Customers', '#Growth_Indicator')

        # read once
        new_s = df_revenue[new_col]
        ret_s = df_revenue[col]

        # total revenue
        total = new_s + ret_s

        # safe denominators
        total_nonzero = total.replace(0, np.nan)
        total_prev_nonzero = total.shift(1).replace(0, np.nan)

        # compute derived series (vectorized)
        new_cols_data[total_revenue_col] = total
        #new_cols_data[share_ret_revenue_col] = ret_s / total_nonzero
        new_cols_data[rrr_col] = ret_s / total_prev_nonzero
        rev_growth = (total - total.shift(1)) / total_prev_nonzero
        new_cols_data[revenue_growth_col] = rev_growth
        new_cols_data[acq_rate_col] = new_s / new_s.shift(1)
        # growth mix: guard division by zero
        #denom = (total - total.shift(1)).replace(0, np.nan)
        #new_cols_data[gm_col] = (ret_s - ret_s.shift(1)) / denom
        # growth indicator: use numpy sign for speed (returns -1,0,1)
        #new_cols_data[growth_indicator_col] = np.sign(rev_growth).astype('Int64')

        # extend final_cols to include the new columns next to the returning-customer column
        final_cols.extend([
            total_revenue_col,
            #share_ret_revenue_col,
            rrr_col,
            revenue_growth_col,
            acq_rate_col,
           # gm_col,
            # growth_indicator_col,
        ])

# Concatenate the new columns once (if any)
if new_cols_data:
    df_new = pd.DataFrame(new_cols_data, index=df_revenue.index)
    df_revenue = pd.concat([df_revenue, df_new], axis=1)

# Reorder columns once: keep only columns that exist (defensive)
final_cols = [c for c in final_cols if c in df_revenue.columns]
df_revenue = df_revenue.loc[:, final_cols]


#==============================================================================
# 3. First Descriptive Analysis 
#==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

###################
# 3.1 Industry Data#
###################

# Plot top-N horizontal bar chart for readability
n_industries_after = df_industry['gics_industry_name'].nunique(dropna=True)
print("Number of Industries:", n_industries_after)

top_n = n_industries_after
to_plot = sub_counts if len(sub_counts) <= top_n else sub_counts.iloc[:top_n]

plt.figure(figsize=(10, max(4, 0.28 * len(to_plot))))
plt.barh(to_plot.index, to_plot.values, color='black')  # single neutral color
plt.xlabel('Number of Firms')
plt.ylabel('GICS Sub-Industry')
plt.title(f'Number of Firms per GICS Sub-Industry (top {len(to_plot)})')
plt.gca().invert_yaxis()   # largest on top
plt.tight_layout()
plt.show()
#plt.savefig('subindustry_counts_top{0}.png'.format(min(len(sub_counts), top_n)), dpi=200, bbox_inches='tight')





###################
# 3.1 Revenue Data#
###################


# --- Build yearly wide-format dataframe matching df_revenue column style ---
# ensure Date is datetime
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'])

# markers we want to aggregate
markers = ['#New_Customers', '#Returning_Customers', '#Total_Revenue']

# helper to extract firm base (everything before the marker)
def extract_firm(col, marker):
    return col.split(marker)[0].rstrip('. ').strip()

# container for per-marker wide tables (index = YearTimestamp, columns = "<Firm><marker>")
per_marker_wide = {}

for marker in markers:
    cols = [c for c in df_revenue.columns if marker in c]
    if not cols:
        per_marker_wide[marker] = pd.DataFrame()  # empty, may compute later
        continue

    # melt to long, compute Year (as Timestamp at year start for consistent Date-like format)
    m = df_revenue[['Date'] + cols].melt(id_vars='Date', value_vars=cols, var_name='col', value_name='value')
    m['Year'] = m['Date'].dt.to_period('Y').dt.to_timestamp()  # e.g., 2025-01-01
    m['Firm'] = m['col'].apply(lambda c: extract_firm(c, marker))
    m['value'] = pd.to_numeric(m['value'], errors='coerce').fillna(0)

    # aggregate yearly sums per firm, then pivot to wide (columns = firms)
    agg = m.groupby(['Year', 'Firm'], as_index=False)['value'].sum()
    wide = agg.pivot(index='Year', columns='Firm', values='value').sort_index(axis=1)

    # rename columns to include the same marker as original df_revenue
    wide.columns = [f"{firm}{marker}" for firm in wide.columns]
    per_marker_wide[marker] = wide

# If Total not present, compute it as sum of new + returning (per firm) after aggregation
if per_marker_wide['#Total_Revenue'].empty:
    nw = per_marker_wide['#New_Customers'] if not per_marker_wide['#New_Customers'].empty else pd.DataFrame()
    rw = per_marker_wide['#Returning_Customers'] if not per_marker_wide['#Returning_Customers'].empty else pd.DataFrame()

    # align indexes and columns, then sum where possible
    if not nw.empty or not rw.empty:
        # reindex to union of available years
        all_idx = nw.index.union(rw.index)
        nw = nw.reindex(all_idx).fillna(0)
        rw = rw.reindex(all_idx).fillna(0)

        # align firm columns by firm base: column names are "Firm#Marker" -> extract firm bases
        def strip_marker(col):
            return col.split('#')[0]

        # build dicts mapping firm->column for each
        nw_map = {strip_marker(c): c for c in nw.columns}
        rw_map = {strip_marker(c): c for c in rw.columns}
        firms = sorted(set(nw_map) | set(rw_map))

        tot_df = pd.DataFrame(index=all_idx)
        for f in firms:
            new_col = nw_map.get(f)
            ret_col = rw_map.get(f)
            new_vals = nw[new_col] if new_col is not None else 0
            ret_vals = rw[ret_col] if ret_col is not None else 0
            tot_df[f"{f}#Total_Revenue"] = (new_vals + ret_vals).fillna(0)

        per_marker_wide['#Total_Revenue'] = tot_df
    else:
        per_marker_wide['#Total_Revenue'] = pd.DataFrame()

# Combine all per-marker wide tables into one wide frame (outer join on Year index)
dfs_to_concat = [df for df in [per_marker_wide[m] for m in markers] if not df.empty]
if dfs_to_concat:
    df_yearly_wide = pd.concat(dfs_to_concat, axis=1).sort_index(axis=1)
else:
    df_yearly_wide = pd.DataFrame()

# Normalize the Year index -> proper Date column and readable Year
if not df_yearly_wide.empty:
    # reset_index creates a column named 'Year' (because index was Year timestamps)
    df_yearly_wide = df_yearly_wide.reset_index()

    # Rename the index column to 'Date' (if it is named 'Year' or something else)
    if 'Year' in df_yearly_wide.columns:
        df_yearly_wide = df_yearly_wide.rename(columns={'Year': 'Date'})
    elif 'index' in df_yearly_wide.columns:
        df_yearly_wide = df_yearly_wide.rename(columns={'index': 'Date'})

    # Ensure Date is datetime (if it's integer ns, this will convert it)
    df_yearly_wide['Date'] = pd.to_datetime(df_yearly_wide['Date'], errors='coerce')

    # Also provide a simple integer year column if useful
    df_yearly_wide['Year_int'] = df_yearly_wide['Date'].dt.year

    # Reorder columns to match df_revenue ordering where possible
    original_cols = [c for c in df_revenue.columns if c != 'Date']
    ordered_cols = ['Date'] + [c for c in original_cols if c in df_yearly_wide.columns]
    remaining = [c for c in df_yearly_wide.columns if c not in ordered_cols]
    df_yearly = df_yearly_wide.reindex(columns=(ordered_cols + remaining)).copy()

    # fill numeric NaNs with 0 (aggregations)
    for col in df_yearly.columns:
        if col != 'Date':
            df_yearly[col] = pd.to_numeric(df_yearly[col], errors='coerce').fillna(0)
else:
    df_yearly = pd.DataFrame()

# Preview
print(df_yearly.head())




# Ensure Date is datetime and df_yearly sorted by Date (ascending)
if 'Date' in df_yearly.columns:
    df_yearly['Date'] = pd.to_datetime(df_yearly['Date'], errors='coerce')
    df_yearly = df_yearly.sort_values('Date').reset_index(drop=True)
else:
    raise KeyError("df_yearly must have a 'Date' column (year-start timestamps)")

# Discover firm bases by scanning existing marker columns
marker_tags = {
    'new': '#New_Customers',
    'ret': '#Returning_Customers',
    'tot': '#Total_Revenue',
}
all_cols = [c for c in df_yearly.columns if c != 'Date']
# extract firm base before first '#' if present, else whole column as fallback
def firm_base(col):
    return col.split('#')[0] if '#' in col else col

# Build set of firm bases that appear with any marker
firms = sorted({ firm_base(c) for c in all_cols })

# Compute metrics vectorized (concat-once) for all firms: recommended and faster.
existing_cols = set(df_yearly.columns)

metrics_data = {}
for f in firms:
    rg_col = f + '#Revenue_Growth'
    rrr_col = f + '#RRR'
    ar_col = f + '#Acq_Rate'

    # if all three metrics already exist, skip this firm
    if {rg_col, rrr_col, ar_col}.issubset(existing_cols):
        continue

    new_col = f + marker_tags['new']
    ret_col = f + marker_tags['ret']
    tot_col = f + marker_tags['tot']

    new_s = pd.to_numeric(df_yearly[new_col], errors='coerce') if new_col in df_yearly.columns else pd.Series(0, index=df_yearly.index, dtype='float64')
    ret_s = pd.to_numeric(df_yearly[ret_col], errors='coerce') if ret_col in df_yearly.columns else pd.Series(0, index=df_yearly.index, dtype='float64')
    if tot_col in df_yearly.columns:
        tot_s = pd.to_numeric(df_yearly[tot_col], errors='coerce')
    else:
        tot_s = (new_s.fillna(0) + ret_s.fillna(0)).astype('float64')

    tot_prev = tot_s.shift(1)
    tot_prev_safe = tot_prev.replace(0, np.nan)

    if rg_col not in existing_cols:
        metrics_data[rg_col] = (tot_s - tot_prev) / tot_prev_safe
    if rrr_col not in existing_cols:
        metrics_data[rrr_col] = ret_s / tot_prev_safe
    if ar_col not in existing_cols:
        metrics_data[ar_col] = new_s / new_s.shift(1).replace(0, np.nan)

# concat once and defragment
if metrics_data:
    df_metrics = pd.DataFrame(metrics_data, index=df_yearly.index)
    df_yearly = pd.concat([df_yearly, df_metrics], axis=1)
    df_yearly = df_yearly.copy()
    print(f"Metrics computed vectorized for {len(firms)} firms (concat-once).")

# Optional: show a small sample of the new metric columns for a single firm
sample_firm = firms[0] if firms else None
if sample_firm:
    cols_to_show = ['Date'] + [c for c in df_yearly.columns if c.startswith(sample_firm + '#')]
    print("Sample (first 10 rows) for firm:", sample_firm)
    print(df_yearly[cols_to_show].head(10))

# Summary: how many metric columns added
metric_cols = [c for c in df_yearly.columns if c.endswith('#Revenue_Growth') or c.endswith('#RRR') or c.endswith('#Acq_Rate')]
print(f"Added {len(metric_cols)} metric columns (RG/RRR/AR) across {len(firms)} firms.")



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

'''
is the share of returning revenue time independent for a firm?
'''