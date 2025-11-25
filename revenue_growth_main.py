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
    df_industry = pd.read_excel(file2_path, header=0);  # Load with header
    print("Industry-Daten erfolgreich eingelesen.")
    print("Shape:", df_industry.shape)

except Exception as e:
    print(f'Could not run: {e}')
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
        new_cols_data[rrr_col] = ret_s / total_prev_nonzero -1
        rev_growth = (total - total.shift(1)) / total_prev_nonzero
        new_cols_data[revenue_growth_col] = rev_growth
        new_cols_data[acq_rate_col] = new_s / new_s.shift(1) -1
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
plt.title(f'Number of Firms per GICS Sub-Industry')
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
        metrics_data[rrr_col] = (ret_s / tot_prev_safe) - 1
    if ar_col not in existing_cols:
        metrics_data[ar_col] = (new_s / new_s.shift(1).replace(0, np.nan)) - 1

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


### --- Transform yearly wide dataframe to panel (long) format ---

# make sure Date is datetime
df_yearly['Date'] = pd.to_datetime(df_yearly['Date'], errors='coerce')

# metric suffixes present in your wide dataframe
metrics_suffixes = ['#Revenue_Growth', '#RRR', '#Acq_Rate']

# find all metric columns
metric_cols = [c for c in df_yearly.columns if any(c.endswith(s) for s in metrics_suffixes)]

# melt to long form: Date, col, value
df_long = df_yearly.melt(id_vars='Date', value_vars=metric_cols, var_name='col', value_name='value')

# extract firm and metric name
df_long['firm'] = df_long['col'].apply(lambda c: c.split('#')[0] if '#' in c else c)
df_long['metric'] = df_long['col'].apply(lambda c: c.split('#', 1)[1] if '#' in c else '')

# pivot so each metric becomes its own column (index = Date, firm)
df_panel = df_long.pivot_table(index=['Date', 'firm'], columns='metric', values='value', aggfunc='first').reset_index()

# optional: rename metric columns to nicer names (remove accidental trailing spaces)
df_panel.columns = [c if isinstance(c, str) else c[1] for c in df_panel.columns]  # keeps Date/firm, handles MultiIndex
# convert metric columns to numeric
for col in ['Revenue_Growth', 'RRR', 'Acq_Rate']:
    if col in df_panel.columns:
        df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce')

# ensure Date is datetime
df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')

# sort by firm, then date (ascending); reset index for a clean integer index
df_panel = df_panel.sort_values(['firm', 'Date'], ascending=[True, True]).reset_index(drop=True)

# preview
print(df_panel.head())

# --------------------------------------------------------------------
# Within-firm variance for core metrics + summary stats across firms
# --------------------------------------------------------------------
metrics_for_variance = [m for m in ['RRR', 'Acq_Rate', 'Revenue_Growth'] if m in df_panel.columns]
if metrics_for_variance:
    df_firm_variance = (
        df_panel.groupby('firm')[metrics_for_variance]
        .var(ddof=1)
        .rename(columns={m: f'{m}_var' for m in metrics_for_variance})
        .reset_index()
    )

    summary_rows = []
    for metric in metrics_for_variance:
        series = df_firm_variance[f'{metric}_var'].dropna()
        if series.empty:
            continue
        summary_rows.append({
            'metric': metric,
            'mean': series.mean(),
            'std': series.std(ddof=1),
            '25Q': series.quantile(0.25),
            '50Q': series.quantile(0.50),
            '75Q': series.quantile(0.75),
        })

    if summary_rows:
        variance_summary = pd.DataFrame(summary_rows)
        print('\nWithin-firm variance summary (variance of metric by firm):')
        print(variance_summary.to_string(index=False))

        top_n = 5  # configurable number of firms to display per metric
        for metric in metrics_for_variance:
            var_col = f'{metric}_var'
            if var_col not in df_firm_variance.columns:
                continue

            top_firms = (
                df_firm_variance[['firm', var_col]]
                .dropna()
                .sort_values(var_col, ascending=False)
                .head(top_n)
            )

            if top_firms.empty:
                continue

            print(f"\nTop {len(top_firms)} firms by within-firm variance in {metric}:")
            print(top_firms.to_string(index=False))
else:
    df_firm_variance = pd.DataFrame()
    print('No metrics available to compute within-firm variance.')


#within industry variance over all periods
metrics_for_industry_var = [m for m in ['RRR', 'Acq_Rate', 'Revenue_Growth'] if m in df_panel.columns]
if metrics_for_industry_var and 'sub_industry' in df_panel.columns:
    df_industry_variance = (
        df_panel.groupby('sub_industry')[metrics_for_industry_var]
        .var(ddof=1)
        .rename(columns={m: f'{m}_var' for m in metrics_for_industry_var})
        .reset_index()
    )

    print('\nWithin-industry variance across all periods (GICS sub-industry level):')
    print(df_industry_variance.to_string(index=False))

    industry_summary_rows = []
    for metric in metrics_for_industry_var:
        var_col = f'{metric}_var'
        if var_col not in df_industry_variance.columns:
            continue
        series = df_industry_variance[var_col].dropna()
        if series.empty:
            continue
        industry_summary_rows.append({
            'metric': metric,
            'mean': series.mean(),
            'std': series.std(ddof=1),
            '25Q': series.quantile(0.25),
            '50Q': series.quantile(0.50),
            '75Q': series.quantile(0.75),
        })

    if industry_summary_rows:
        industry_variance_summary = pd.DataFrame(industry_summary_rows)
        print('\nWithin-industry variance summary (variance of metric by sub-industry):')
        print(industry_variance_summary.to_string(index=False))

    top_n = 5
    for metric in metrics_for_industry_var:
        var_col = f'{metric}_var'
        if var_col not in df_industry_variance.columns:
            continue

        top_industries = (
            df_industry_variance[['sub_industry', var_col]]
            .dropna()
            .sort_values(var_col, ascending=False)
            .head(top_n)
        )

        if top_industries.empty:
            continue

        print(f"\nTop {len(top_industries)} sub-industries by variance in {metric}:")
        print(top_industries.to_string(index=False))
else:
    df_industry_variance = pd.DataFrame()
    print('Cannot compute within-industry variance: missing metrics or sub_industry column.')

# --------------------------------------------------------------------
# Map industry / sub-industry onto the panel (one row per Date x Firm)
# --------------------------------------------------------------------
# Build normalized lookup from df_industry (ID -> sub_industry only)
if 'ID' in df_industry.columns:
    cols_present = [
        col
        for col in ['ID', 'gics_sub_industry_name', 'gics_industry_name']
        if col in df_industry.columns
    ]
    ind_map_df = df_industry[cols_present].copy()
    ind_map_df['ID_norm'] = ind_map_df['ID'].astype(str).str.upper().str.strip()
    sub_map = dict(zip(ind_map_df['ID_norm'], ind_map_df['gics_sub_industry_name']))

    industry_map = {}
    if 'gics_industry_name' in ind_map_df.columns:
        industry_map = dict(zip(ind_map_df['ID_norm'], ind_map_df['gics_industry_name']))

    # normalize firm ids in a temporary Series and map to sub-industry (no new column)
    firm_norm_series = df_panel['firm'].astype(str).str.upper().str.strip()
    df_panel['sub_industry'] = firm_norm_series.map(sub_map).fillna('Unknown')
    if industry_map:
        df_panel['industry'] = firm_norm_series.map(industry_map).fillna('Unknown')
    else:
        df_panel['industry'] = 'Unknown'

    # diagnostics: how many unique firms in panel and how many rows map to Unknown
    n_total_firms = firm_norm_series.nunique()
    n_unknown = (df_panel['sub_industry'] == 'Unknown').sum()
    print(f"Panel mapping: {n_total_firms} unique firms in panel; {n_unknown} rows with Unknown sub_industry")
    
    # --------------------------------------------------------------------
    # Compute relative metrics (cRG, cRRR, cAR) = firm metric - sub-industry median
    # and build a per-(Date, sub_industry) summary (25Q,50Q,75Q,mean,std) for
    # the original metrics.
    # --------------------------------------------------------------------
    metrics = ['Revenue_Growth', 'RRR', 'Acq_Rate']
    available_metrics = [m for m in metrics if m in df_panel.columns]
    
    if available_metrics:
        # compute sub-industry median aligned to each row using groupby.transform
        sub_medians = df_panel.groupby(['Date', 'sub_industry'])[available_metrics].transform('median')
        
        # compute centered columns per firm
        if 'Revenue_Growth' in available_metrics:
            df_panel['cRG'] = df_panel['Revenue_Growth'] - sub_medians['Revenue_Growth']
        if 'RRR' in available_metrics:
            df_panel['cRRR'] = df_panel['RRR'] - sub_medians['RRR']
        if 'Acq_Rate' in available_metrics:
            df_panel['cAR'] = df_panel['Acq_Rate'] - sub_medians['Acq_Rate']
        
        # Build summary statistics (long format) for original metric values per Date x sub_industry
        summary_rows = []
        for m in available_metrics:
            agg = (
                df_panel
                .groupby(['Date', 'sub_industry'])[m]
                .agg(
                    q25=lambda x: x.quantile(0.25),
                    q50=lambda x: x.quantile(0.50),
                    q75=lambda x: x.quantile(0.75),
                    mean='mean',
                    std='std'
                )
                .reset_index()
            )
            agg['metric'] = m
            summary_rows.append(agg)
        
        if summary_rows:
            df_subindustry_stats = pd.concat(summary_rows, ignore_index=True)
        else:
            df_subindustry_stats = pd.DataFrame()
        
        # show small samples
        print('\nSample centered columns (first 6 rows):')
        print(df_panel[['Date', 'firm', 'sub_industry'] + [c for c in ['cRG','cRRR','cAR'] if c in df_panel.columns]].head(6))
        print('\nSample sub-industry summary (first 6 rows):')
        print(df_subindustry_stats.head(6))
    else:
        df_subindustry_stats = pd.DataFrame()
        print('No metric columns (Revenue_Growth, RRR, Acq_Rate) found in panel; skipping centered columns and summary.')

    # move sub_industry column next to firm for readability
    cols = list(df_panel.columns)
    # desired ordering: Date, firm, industry info, then others
    ordered = ['Date', 'firm', 'industry', 'sub_industry'] + [
        c for c in cols if c not in ['Date', 'firm', 'industry', 'sub_industry']
    ]
    df_panel = df_panel.reindex(columns=ordered)
else:
    print("Warning: df_industry has no 'ID' column; cannot map industry info to panel.")
    df_panel['industry'] = 'Unknown'



# --------------------------------------------------------------------
# Sub-industry lollipop charts (median metrics per GICS sub-industry)
# --------------------------------------------------------------------
sub_industry_col = 'sub_industry'

if sub_industry_col in df_panel.columns:
    metric_specs = [
        ('RRR', 'Revenue Retention Rate (RRR)'),
        ('Acq_Rate', 'Acquisition Rate (AR)'),
        ('Revenue_Growth', 'Revenue Growth (RG)')
    ]

    df_industry_metrics = df_panel[[sub_industry_col] + [m for m, _ in metric_specs if m in df_panel.columns]].copy()
    df_industry_metrics[sub_industry_col] = df_industry_metrics[sub_industry_col].fillna('Unknown')
    df_industry_metrics = df_industry_metrics[df_industry_metrics[sub_industry_col] != 'Unknown']

    for metric, pretty_label in metric_specs:
        if metric not in df_industry_metrics.columns:
            print(f"Metric {metric} not available for industry lollipop; skipping.")
            continue

        metric_series = pd.to_numeric(df_industry_metrics[metric], errors='coerce')
        medians = (
            df_industry_metrics
            .assign(metric_value=metric_series)
            .dropna(subset=['metric_value'])
            .groupby(sub_industry_col)['metric_value']
            .median()
            .sort_values(ascending=False)
        )

        if medians.empty:
            print(f"No median values available for {pretty_label}; skipping lollipop chart.")
            continue

        y_pos = np.arange(len(medians))
        med_vals = medians.values
        xmin = np.minimum(0, med_vals)
        xmax = np.maximum(0, med_vals)

        plt.figure(figsize=(10, max(4, 0.3 * len(medians))))
        plt.hlines(y=y_pos, xmin=xmin, xmax=xmax, color='black', linewidth=2)
        plt.scatter(med_vals, y_pos, color='black', s=50)
        plt.yticks(y_pos, medians.index)
        plt.xlabel(f"Median {pretty_label}")
        plt.ylabel('GICS Sub-Industry')
        plt.title(f"Median {pretty_label} by GICS Sub-Industry")
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
else:
    print("Sub-industry column missing in df_panel; skipping sub-industry lollipop charts.")







# --------------------------------------------------------------------
# Classify growth path per firm x period into one of four buckets using
# centered metrics cAR (centered Acquisition Rate) and cRRR (centered RRR)
# Rules (per your spec):
#   "Acquisition Driven": cAR >= 0 and cRRR < 0
#   "Retention Driven":    cAR < 0  and cRRR >= 0
#   "Dual Engine":         cAR >= 0 and cRRR >= 0
#   "Shrinking":           cAR < 0  and cRRR < 0
# Rows with missing cAR or cRRR will be labeled 'Unknown'.
# --------------------------------------------------------------------
if {'cAR', 'cRRR'}.issubset(set(df_panel.columns)):
    # use vectorized np.select; comparisons with NaN yield False so Unknown will be used
    cond_acq = df_panel['cAR'] >= 0
    cond_ret = df_panel['cRRR'] >= 0

    conditions = [
        (cond_acq & (~cond_ret)),   # Acquisition Driven: cAR>=0 and cRRR<0
        ((~cond_acq) & cond_ret),   # Retention Driven: cAR<0 and cRRR>=0
        (cond_acq & cond_ret),      # Dual Engine: both >= 0
        ((~cond_acq) & (~cond_ret)) # Shrinking: both < 0
    ]

    choices = [
        'Acquisition Driven',
        'Retention Driven',
        'Dual Engine',
        'Shrinking'
    ]

    df_panel['growth_path'] = np.select(conditions, choices, default='Unknown')

    # Quick diagnostics: counts per growth_path (top few)
    gp_counts = df_panel['growth_path'].value_counts(dropna=False)
    print('\nGrowth path distribution (sample):')
    print(gp_counts.head(10))
else:
    print("Cannot compute 'growth_path': required centered columns 'cAR' and/or 'cRRR' not found in df_panel.")

# --------------------------------------------------------------------
# Add revenue_growth label per your spec: 'Increasing' if cRG >= 0, else 'Decreasing'
# --------------------------------------------------------------------
if 'cRG' in df_panel.columns:
    # Comparison with NaN yields False, so NaNs become 'Decreasing' per your "else" rule
    df_panel['revenue_growth'] = np.where(df_panel['cRG'] >= 0, 'Increasing', 'Decreasing')
    rg_counts = df_panel['revenue_growth'].value_counts(dropna=False)
    print('\nRevenue growth distribution (sample):')
    print(rg_counts.head(10))
else:
    print("Cannot compute 'revenue_growth': required centered column 'cRG' not found in df_panel.")

# --------------------------------------------------------------------
# Firm-level dominant strategy: determine if a single growth_path and/or
# revenue_growth label occurs in at least 50% of (non-Unknown) periods for
# each firm. If so, declare it dominant; otherwise mark 'Mixed Strategy'.
# We'll exclude 'Unknown' rows from the denominator when judging dominance.
# Output: df_firm_strategy with one row per firm and columns:
#   - firm, n_periods_growthpath, dominant_growth_path, share_growth_path,
#   - n_periods_rev, dominant_revenue_growth, share_revenue_growth
# --------------------------------------------------------------------
def _dominant_label(series, ignore_label='Unknown', threshold=0.5):
    # series: pd.Series of categorical labels for one firm
    s = series.dropna()
    if s.empty:
        return ('Mixed Strategy', 0.0, 0)
    # drop the ignore_label from consideration
    s_non = s[s != ignore_label]
    n_total = len(s_non)
    if n_total == 0:
        return ('Mixed Strategy', 0.0, 0)
    vc = s_non.value_counts()
    top_label = vc.idxmax()
    top_share = vc.iloc[0] / n_total
    if top_share >= threshold:
        return (top_label, float(top_share), int(n_total))
    else:
        return ('Mixed Strategy', float(top_share), int(n_total))







"""
share of retained revenue time invariant?
RRR autocorrelation
AR autocorrelation

"""



#==============================================================================
# RQ:1 Firm-Level Dominant Strategy
#==============================================================================


# Build summary per firm
firms = df_panel['firm'].unique()
rows = []
for f in firms:
    sub = df_panel[df_panel['firm'] == f]

    # growth_path dominance
    if 'growth_path' in sub.columns:
        gp_label, gp_share, gp_n = _dominant_label(sub['growth_path'], ignore_label='Unknown', threshold=0.5)
    else:
        gp_label, gp_share, gp_n = ('Mixed Strategy', 0.0, 0)

    # revenue_growth dominance
    if 'revenue_growth' in sub.columns:
        rg_label, rg_share, rg_n = _dominant_label(sub['revenue_growth'], ignore_label='Unknown', threshold=0.5)
    else:
        rg_label, rg_share, rg_n = ('Mixed Strategy', 0.0, 0)

    rows.append({
        'firm': f,
        'n_periods_growthpath': gp_n,
        'dominant_growth_path': gp_label,
        'share_growth_path': gp_share,
        'n_periods_revenue_growth': rg_n,
        'dominant_revenue_growth': rg_label,
        'share_revenue_growth': rg_share,
    })

df_firm_strategy = pd.DataFrame(rows)

print('\nFirm-level dominant strategy sample (first 10 rows):')
print(df_firm_strategy.head(10))

# Count how many firms have each dominant_growth_path
if 'dominant_growth_path' in df_firm_strategy.columns:
    dom_counts = df_firm_strategy['dominant_growth_path'].value_counts(dropna=False)
    print('\nCounts of firms by dominant_growth_path:')
    print(dom_counts)
else:
    print("No 'dominant_growth_path' column in df_firm_strategy to count.")

# --------------------------------------------------------------------
# Cluster-robust OLS (no fixed effects)
# Estimate: cRG ~ cAR + cRRR and report SEs clustered by firm. This is the
# user's requested default: pooled OLS with firm-clustered standard errors.
# Graceful fallbacks: if statsmodels missing or clustering fails, print plain OLS.
# --------------------------------------------------------------------
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except Exception as e:
    print(f"Statsmodels not available; skipping clustered OLS: {e}")
else:
    reg_cols = ['Date', 'firm', 'cRG', 'cAR', 'cRRR']
    if not set(reg_cols).issubset(set(df_panel.columns)):
        print("Not all regression columns present (need Date, firm, cRG, cAR, cRRR); skipping regression.")
    else:
        # build a compact dataframe and drop rows with any missing regression values
        df_reg = df_panel[reg_cols].dropna().copy()
        if df_reg.empty:
            print("No complete observations for regression after dropping NaNs; skipping regression.")
        else:
            # add a Year column for diagnostics only (not used in model)
            df_reg['Year'] = df_reg['Date'].dt.year
            n_obs = len(df_reg)
            n_firms = df_reg['firm'].nunique()
            n_years = df_reg['Year'].nunique()
            print(f"Running pooled OLS on {n_obs} obs, {n_firms} firms, {n_years} years (clustered SEs by firm)")

            formula = 'cRG ~ cAR + cRRR'
            try:
                mod = smf.ols(formula=formula, data=df_reg)
                res = mod.fit()

                # try cluster-robust covariance (cluster by firm)
                try:
                    res_clust = res.get_robustcov_results(cov_type='cluster', groups=df_reg['firm'])

                    # tidy output with clustered SEs
                    keep_vars = list(res_clust.params.index)
                    coef_df_clust = pd.DataFrame({
                        'coef': res_clust.params.loc[keep_vars],
                        'se_cluster': res_clust.bse.loc[keep_vars],
                        't_cluster': res_clust.tvalues.loc[keep_vars],
                        'p_cluster': res_clust.pvalues.loc[keep_vars]
                    })
                    ci_cl = res_clust.conf_int().loc[keep_vars]
                    coef_df_clust['ci_lower'] = ci_cl[0]
                    coef_df_clust['ci_upper'] = ci_cl[1]

                    print('\nCluster-robust OLS estimates (no fixed effects; SEs clustered by firm):')
                    print(coef_df_clust.round(4))
                except Exception as e_clust:
                    print(f"Could not compute cluster-robust SEs (cluster by firm): {e_clust}\nFalling back to plain OLS summary:")
                    print(res.summary())
            except Exception as e_mod:
                print(f"OLS estimation failed: {e_mod}")



# --------------------------------------------------------------------
# Firm-level logistic regression (simple, compact)
# Dependent: dominant_revenue_growth (Increasing=1). Predictor: dominant_growth_path dummies.
# Uses statsmodels Logit and HC1 robust SEs when available. Minimal control flow.
# --------------------------------------------------------------------
try:
    import statsmodels.api as sm
except Exception as e:
    print(f"Statsmodels not available; skipping firm-level logistic regression: {e}")
else:
    if 'df_firm_strategy' not in globals() or df_firm_strategy is None or df_firm_strategy.empty:
        print('`df_firm_strategy` not present or empty; skipping firm-level logistic regression.')
    else:
        df_lr = df_firm_strategy[['firm', 'dominant_growth_path', 'dominant_revenue_growth']].dropna()
        df_lr = df_lr[df_lr['dominant_revenue_growth'].isin(['Increasing', 'Decreasing'])]
        if df_lr.empty:
            print('No firms with clear Increasing/Decreasing outcome; skipping firm-level logistic regression.')
        else:
            y = (df_lr['dominant_revenue_growth'] == 'Increasing').astype(int)
            X = pd.get_dummies(df_lr['dominant_growth_path'], prefix='growth_path', drop_first=True)

            # coerce, drop any non-numeric rows, then fit
            df_xy = pd.concat([X, y.rename('y')], axis=1).apply(pd.to_numeric, errors='coerce').dropna()
            if df_xy.empty:
                print('No valid rows for firm-level logistic regression after coercion; skipping.')
            else:
                X_clean = sm.add_constant(df_xy.drop(columns='y')).astype(float)
                y_clean = df_xy['y'].astype(float)
                try:
                    # For Logit, request robust SEs directly in fit() call via cov_type parameter
                    res = sm.Logit(y_clean, X_clean).fit(cov_type='HC1', disp=False)
                    tidy = pd.DataFrame({
                        'coef': res.params,
                        'std_err': res.bse,
                        'p_value': res.pvalues,
                    })
                    tidy['OR'] = np.exp(tidy['coef'])
                    print('\nFirm-level Logit (Increasing=1) — HC1 robust SEs:')
                    print(tidy.round(4))

                    lr_df = int(res.df_model)
                    lr_stat = float(res.llr)
                    lr_pvalue = float(res.llr_pvalue)
                    n_obs = int(res.nobs)
                    pseudo_r2 = float(res.prsquared)
                    print(
                        f"\nModel diagnostics: n_obs={n_obs}, LR chi2({lr_df})={lr_stat:.3f}, "
                        f"p-value={lr_pvalue:.4g}, McFadden R^2={pseudo_r2:.4f}"
                    )
                except Exception as e:
                    print(f'Firm-level Logit failed: {e}')


#### if I achieve above median acquisition path -> higher likelihood of having increasing revenue growth 
### mixed strategy is worse, shrinking of course as well 
### how persistent is the acqusition path? 


#==============================================================================
# RQ2: Persistency of Growth Paths (Markov Chain Analysis)
#==============================================================================
# Objective: Determine whether firms consistently use the same growth mechanism
# Methodology: Treat each period's growth path as a state in a Markov chain
# - Persistence = probability of staying in the same growth path
# - Transition matrix = probability of moving from state i to state j
#==============================================================================

if 'growth_path' not in df_panel.columns:
    print("\n'growth_path' column not found in df_panel; skipping Markov chain analysis.")
else:
    # Prepare panel data sorted by firm and date
    df_markov = df_panel[['firm', 'Date', 'growth_path']].sort_values(['firm', 'Date']).copy()
    df_markov = df_markov[df_markov['growth_path'] != 'Unknown'].copy()  # exclude Unknown states
    
    if df_markov.empty:
        print("\nNo valid growth path observations for Markov analysis; skipping.")
    else:
        # Get unique states (growth paths)
        states = sorted(df_markov['growth_path'].unique())
        n_states = len(states)
        print(f"\n{'='*70}")
        print(f"RQ2: Markov Chain Analysis of Growth Path Persistence")
        print(f"{'='*70}")
        print(f"States: {states}")
        print(f"Number of firms: {df_markov['firm'].nunique()}")
        print(f"Total observations: {len(df_markov)}")
        
        # Create lagged growth path (previous state) within each firm
        df_markov['growth_path_lag'] = df_markov.groupby('firm')['growth_path'].shift(1)
        
        # Drop first observation per firm (no previous state)
        df_transitions = df_markov.dropna(subset=['growth_path_lag']).copy()
        
        if df_transitions.empty:
            print("\nNo transitions observed (firms need at least 2 periods); skipping.")
        else:
            print(f"Total transitions observed: {len(df_transitions)}")
            
            # --------------------------------------------------------------------
            # 1. Per-firm persistence rates
            # --------------------------------------------------------------------
            firm_persistence = []
            for firm in df_transitions['firm'].unique():
                firm_data = df_transitions[df_transitions['firm'] == firm]
                
                # Count transitions for this firm by state
                for state in states:
                    state_data = firm_data[firm_data['growth_path_lag'] == state]
                    if len(state_data) > 0:
                        # Persistence = fraction of times stayed in same state
                        stayed = (state_data['growth_path'] == state).sum()
                        total = len(state_data)
                        persistence_rate = stayed / total
                        firm_persistence.append({
                            'firm': firm,
                            'growth_path': state,
                            'n_transitions': total,
                            'persistence_rate': persistence_rate
                        })
            
            df_persistence = pd.DataFrame(firm_persistence)
            
            # Summary statistics by growth path
            if not df_persistence.empty:
                persistence_summary = df_persistence.groupby('growth_path')['persistence_rate'].agg([
                    ('mean', 'mean'),
                    ('std', 'std'),
                    ('q25', lambda x: x.quantile(0.25)),
                    ('median', lambda x: x.quantile(0.50)),
                    ('q75', lambda x: x.quantile(0.75)),
                    ('n_firms', 'count')
                ]).reset_index()
                
                print("\n" + "="*70)
                print("PERSISTENCE SUMMARY BY GROWTH PATH")
                print("="*70)
                print("(Persistence = Probability of staying in same state next period)")
                print("\n" + persistence_summary.to_string(index=False))
                print("="*70)
            
            # --------------------------------------------------------------------
            # 2. Global transition matrix (averaged across all firms)
            # --------------------------------------------------------------------
            # Count all transitions globally
            transition_counts = pd.crosstab(
                df_transitions['growth_path_lag'],
                df_transitions['growth_path'],
                margins=False
            )
            
            # Ensure all states appear (fill missing with 0)
            for state in states:
                if state not in transition_counts.index:
                    transition_counts.loc[state] = 0
                if state not in transition_counts.columns:
                    transition_counts[state] = 0
            
            # Reorder to match state order
            transition_counts = transition_counts.reindex(index=states, columns=states, fill_value=0)
            
            # Convert counts to probabilities (row-normalize)
            transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
            
            print("\n" + "="*70)
            print("GLOBAL TRANSITION PROBABILITY MATRIX")
            print("="*70)
            print("(Rows = current state, Columns = next state)")
            print("\n" + transition_probs.round(3).to_string())
            print("="*70)
            
            # --------------------------------------------------------------------
            # 3. Heatmap of transition probabilities
            # --------------------------------------------------------------------
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    transition_probs,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Transition Probability'},
                    xticklabels=states,
                    yticklabels=states,
                    vmin=0,
                    vmax=1
                )
                plt.title('Growth Path Transition Probabilities\n(Averaged Across All Firms)', fontsize=14, fontweight='bold')
                plt.xlabel('Next Period State', fontsize=12)
                plt.ylabel('Current Period State', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()
                
                print("\nHeatmap displayed. Diagonal = persistence (staying in same state).")
                print("Off-diagonal = transition probabilities to other states.")
                
            except Exception as e:
                print(f"\nCould not create heatmap: {e}")


#==============================================================================
# RQ3: Trend Analysis - ARIMA(0,1,0) for Growth Path Trajectories
#==============================================================================
# Objective: Identify trajectories of cAR, cRRR, and cRG over time
# Methodology: ARIMA(0,1,0) = Random Walk = First Differences
#   - Models ΔcAR, ΔcRRR, ΔcRG (change from t-1 to t)
#   - Tests if trends differ by growth path
#   - Examines how component trends (ΔcAR, ΔcRRR) impact revenue growth trend (ΔcRG)
#==============================================================================

if 'cRG' not in df_panel.columns or 'cAR' not in df_panel.columns or 'cRRR' not in df_panel.columns:
    print("\nCentered metrics (cRG, cAR, cRRR) not found in df_panel; skipping RQ3 trend analysis.")
else:
    # --------------------------------------------------------------------
    # 1. Create first differences (ARIMA 0,1,0)
    # --------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"RQ3: Trend Analysis Using ARIMA(0,1,0) - First Differences")
    print(f"{'='*80}\n")
    
    # Start with panel data containing centered metrics
    df_trend = df_panel[['firm', 'Date', 'cRG', 'cAR', 'cRRR', 'growth_path']].copy()
    
    # Compute first differences within each firm (sorted by date)
    df_trend = df_trend.sort_values(['firm', 'Date'])
    for var in ['cRG', 'cAR', 'cRRR']:
        df_trend[f'delta_{var}'] = df_trend.groupby('firm')[var].diff()
    
    # Drop rows with missing deltas (first observation per firm)
    df_trend_clean = df_trend.dropna(subset=['delta_cRG', 'delta_cAR', 'delta_cRRR']).copy()
    
    print(f"Trend analysis sample: {len(df_trend_clean)} observations, {df_trend_clean['firm'].nunique()} firms")
    
    # Descriptive statistics of first differences by growth path
    if 'growth_path' in df_trend_clean.columns:
        print("\nMean first differences by growth path:")
        for path in ['Acquisition Driven', 'Retention Driven', 'Dual Engine', 'Shrinking']:
            sub = df_trend_clean[df_trend_clean['growth_path'] == path]
            if len(sub) > 0:
                print(f"\n{path} (n={len(sub)}):")
                print(f"  ΔcAR:  {sub['delta_cAR'].mean():.4f}")
                print(f"  ΔcRRR: {sub['delta_cRRR'].mean():.4f}")
                print(f"  ΔcRG:  {sub['delta_cRG'].mean():.4f}")
    
    # --------------------------------------------------------------------
    # 2. Test significance of trends: Three independent regressions
    # --------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("TREND SIGNIFICANCE TESTS (ARIMA 0,1,0)")
    print(f"{'='*80}\n")
    
    # Prepare regression data
    df_reg_trend = df_trend_clean.dropna(subset=['delta_cRG', 'delta_cAR', 'delta_cRRR', 'firm', 'Date']).copy()
    
    if len(df_reg_trend) > 50:  # Minimum sample size
        try:
            from linearmodels.panel import PanelOLS
            import statsmodels.api as sm
            
            # Set panel index
            df_reg_trend = df_reg_trend.set_index(['firm', 'Date'])
            
            # Regression 1: ΔcAR ~ 1 with firm fixed effects
            print("Regression 1: ΔcAR ~ constant + firm FE")
            y_reg = df_reg_trend[['delta_cAR']]
            X_reg = sm.add_constant(pd.DataFrame(index=y_reg.index))
            model1 = PanelOLS(y_reg, X_reg, entity_effects=True, time_effects=False, drop_absorbed=True)
            results1 = model1.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
            print(results1.summary)
            print("\n")
            
            # Regression 2: ΔcRRR ~ 1 with firm fixed effects
            print("Regression 2: ΔcRRR ~ constant + firm FE")
            y_reg = df_reg_trend[['delta_cRRR']]
            X_reg = sm.add_constant(pd.DataFrame(index=y_reg.index))
            model2 = PanelOLS(y_reg, X_reg, entity_effects=True, time_effects=False, drop_absorbed=True)
            results2 = model2.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
            print(results2.summary)
            print("\n")
            
            # Regression 3: ΔcRG ~ 1 with firm fixed effects
            print("Regression 3: ΔcRG ~ constant + firm FE")
            y_reg = df_reg_trend[['delta_cRG']]
            X_reg = sm.add_constant(pd.DataFrame(index=y_reg.index))
            model3 = PanelOLS(y_reg, X_reg, entity_effects=True, time_effects=False, drop_absorbed=True)
            results3 = model3.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
            print(results3.summary)
            print(f"\n{'='*80}\n")
            
            # --------------------------------------------------------------------
            # 3. Extract firm fixed effects and create histograms
            # --------------------------------------------------------------------
            print(f"\n{'='*80}")
            print("FIRM FIXED EFFECTS HISTOGRAMS")
            print(f"{'='*80}\n")
            
            # Extract firm fixed effects from each model
            # Note: PanelOLS with entity_effects=True estimates firm fixed effects
            # The estimated_effects attribute contains these effects
            try:
                import matplotlib.pyplot as plt
                from scipy import stats
                
                fe_data = {}
                models = [
                    (results1, model1, 'ΔcAR', 'delta_cAR'),
                    (results2, model2, 'ΔcRRR', 'delta_cRRR'),
                    (results3, model3, 'ΔcRG', 'delta_cRG')
                ]
                
                def _extract_entity_effects(res_obj):
                    effects = getattr(res_obj, 'estimated_effects', None)
                    if effects is None:
                        return None

                    eff_df = effects.copy()

                    # If columns are multi-level (e.g., ('entity', 'effect')), select entity level
                    if isinstance(eff_df, pd.DataFrame):
                        if isinstance(eff_df.columns, pd.MultiIndex):
                            level0 = eff_df.columns.get_level_values(0)
                            if 'entity' in level0:
                                eff_df = eff_df.xs('entity', axis=1, level=0)
                            # collapse remaining multi-index columns if only one column at lower level
                            if isinstance(eff_df, pd.DataFrame) and eff_df.shape[1] == 1:
                                eff_df = eff_df.iloc[:, 0]
                        elif eff_df.shape[1] == 1:
                            eff_df = eff_df.iloc[:, 0]

                    if isinstance(eff_df, pd.DataFrame):
                        # If still DataFrame, take first column as fixed effect values
                        eff_series = eff_df.iloc[:, 0]
                    else:
                        eff_series = eff_df

                    if eff_series is None:
                        return None

                    if isinstance(eff_series, pd.DataFrame):
                        eff_series = eff_series.iloc[:, 0]

                    eff_series = pd.Series(eff_series, copy=True)

                    if isinstance(eff_series.index, pd.MultiIndex):
                        names = list(eff_series.index.names)
                        if 'firm' in names:
                            drop_levels = [lvl for lvl in names if lvl != 'firm']
                            eff_series = eff_series.droplevel(drop_levels)
                        else:
                            # fallback: keep the last level (assumed to be entity id)
                            eff_series = eff_series.droplevel(list(range(eff_series.index.nlevels - 1)))

                    eff_series.name = 'firm_effect'
                    eff_series = eff_series.astype(float)

                    if not eff_series.index.is_unique:
                        eff_series = eff_series.groupby(level=0).mean()

                    return eff_series

                for results, model, label, var in models:
                    fe = _extract_entity_effects(results)
                    if fe is None or fe.empty:
                        print(f"{label}: No entity fixed effects available")
                        fe_data[label] = None
                        continue

                    # Residual variance for approximate SEs
                    residual_var = results.resid_ss / results.df_resid

                    firm_counts = (
                        df_reg_trend.reset_index()
                        .groupby('firm')
                        .size()
                    )

                    counts_map = firm_counts.to_dict()
                    n_obs = fe.index.map(counts_map)

                    fe_df = pd.DataFrame({
                        'effect': fe,
                        'n_obs': n_obs,
                    })
                    fe_df = fe_df.dropna()
                    fe_df = fe_df[fe_df['n_obs'] > 1]

                    if fe_df.empty:
                        print(f"{label}: No firms with sufficient observations for FE histogram")
                        fe_data[label] = None
                        continue

                    fe_df['se'] = np.sqrt(residual_var / fe_df['n_obs'])
                    fe_df['t_stat'] = fe_df['effect'] / fe_df['se']

                    t_stats = fe_df['t_stat']
                    df_resid = results.df_resid
                    p_values = pd.Series(
                        2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid)),
                        index=fe_df.index,
                    )

                    fe_filtered = fe_df['effect'].copy()
                    fe_filtered.loc[p_values > 0.1] = 0

                    fe_filtered.name = 'effect'
                    fe_filtered.index.name = 'firm'
                    fe_data[label] = fe_filtered

                    n_total = len(fe_df)
                    n_significant = (p_values <= 0.1).sum()
                    n_zero = (fe_filtered == 0).sum()
                    print(
                        f"{label}: {n_total} firms, {n_significant} significant (p≤0.1), {n_zero} set to zero"
                    )
                
                # Create histograms
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                for idx, (label, fe_filtered) in enumerate(fe_data.items()):
                    if fe_filtered is not None and len(fe_filtered) > 0:
                        ax = axes[idx]
                        
                        # Plot histogram
                        ax.hist(fe_filtered, bins=30, color='black', edgecolor='white', alpha=0.7)
                        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
                        ax.set_xlabel('Firm Fixed Effect')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'{label} Firm Fixed Effects\n(p>0.1 set to 0)')
                        ax.legend()
                        ax.grid(axis='y', alpha=0.3)
                        
                        # Add statistics text
                        mean_fe = fe_filtered.mean()
                        median_fe = fe_filtered.median()
                        ax.text(0.02, 0.98, f'Mean: {mean_fe:.4f}\nMedian: {median_fe:.4f}',
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    else:
                        axes[idx].text(0.5, 0.5, f'{label}\nNo data',
                                      ha='center', va='center', transform=axes[idx].transAxes)
                        axes[idx].set_title(f'{label} Firm Fixed Effects')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n{'='*80}\n")
                
            except Exception as e:
                print(f"Could not create firm fixed effects histograms: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"\nTrend significance tests failed: {e}")
    else:
        print(f"\nInsufficient observations for regression (n={len(df_reg_trend)}); skipping.")
    


#==============================================================================
# RQ4: Which growth path yields the highest long-run revenue growth?
#==============================================================================

long_run = (
    df_panel[['firm', 'cRG', 'growth_path']]
    .dropna(subset=['cRG', 'growth_path'])
    .groupby(['growth_path'])['cRG']
    .median()
    .sort_values(ascending=False)
)

print(long_run)

