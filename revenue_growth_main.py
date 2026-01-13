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
# helper to normalize firm identifiers consistently throughout the pipeline
def normalize_strings(values):
    if values is None:
        return []
    normalized = []
    for val in values:
        if val is None:
            continue
        normalized.append(str(val).upper().strip())
    return normalized

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

        # read once and coerce to numeric to prevent string arithmetic issues
        new_s = pd.to_numeric(df_revenue[new_col], errors='coerce')
        ret_s = pd.to_numeric(df_revenue[col], errors='coerce')

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

# Post-filter industry coverage summary
n_industries_after = df_industry['gics_industry_name'].nunique(dropna=True)
print("Number of Industries:", n_industries_after)

sub_counts = df_industry['gics_sub_industry_name'].value_counts(dropna=True)
n_subindustries_after = df_industry['gics_sub_industry_name'].nunique(dropna=True)
print("Number of Sub-Industries:", n_subindustries_after)

df_subindustry_counts = (
    sub_counts.rename_axis('Subindustry')
    .reset_index(name='Number of firms')
)

total_firms = n_firms_industry_after

if total_firms > 0:
    df_subindustry_counts['Share of firms (%)'] = (
        df_subindustry_counts['Number of firms'] / total_firms * 100
    ).round(2)
else:
    df_subindustry_counts['Share of firms (%)'] = np.nan

output_excel_path = 'subindustry_firm_counts.xlsx'
try:
    df_subindustry_counts.to_excel(output_excel_path, index=False)
    print(f"Saved sub-industry coverage to {output_excel_path}")
except Exception as excel_err:
    print(f"Could not save Excel summary: {excel_err}")

print(f"\nSub-industry coverage (post filter, total firms = {total_firms}):")
display_df = df_subindustry_counts.copy()
display_df['Share of firms'] = display_df['Share of firms (%)'].apply(
    lambda x: f"{x:.0f}%" if pd.notnull(x) and float(x).is_integer() else (f"{x:.2f}%" if pd.notnull(x) else 'NA')
)
print(display_df[['Subindustry', 'Number of firms', 'Share of firms']].to_string(index=False))






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

# metric suffixes present in your wide dataframe (include level for size control)
metrics_suffixes = ['#Revenue_Growth', '#RRR', '#Acq_Rate', '#Total_Revenue']

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
for col in ['Revenue_Growth', 'RRR', 'Acq_Rate', 'Total_Revenue']:
    if col in df_panel.columns:
        df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce')

# ensure Date is datetime
df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')

# sort by firm, then date (ascending); reset index for a clean integer index
df_panel = df_panel.sort_values(['firm', 'Date'], ascending=[True, True]).reset_index(drop=True)

if not df_panel.empty:
    df_panel['period_year'] = df_panel['Date'].dt.year.astype(float)
else:
    df_panel['period_year'] = pd.Series(dtype='float64')

# preview
print(df_panel.head())


def print_top_metric_leaderboard(panel_df: pd.DataFrame, metric_col: str, pretty_label: str, value_label: str | None = None, top_n: int = 20) -> None:
    """Print top-N firm-period observations for a metric."""
    required_cols = {'firm', 'Date', metric_col}
    if not required_cols.issubset(panel_df.columns):
        missing = sorted(required_cols.difference(panel_df.columns))
        print(f"Top {pretty_label} leaderboard skipped: missing columns {missing} in df_panel.")
        return

    leaderboard = panel_df[['firm', 'Date', metric_col]].dropna().copy()
    if leaderboard.empty:
        print(f"Top {pretty_label} leaderboard skipped: no complete observations after dropping NaNs.")
        return

    leaderboard = leaderboard.sort_values(metric_col, ascending=False).head(top_n).copy()
    leaderboard.insert(0, 'Position', range(1, len(leaderboard) + 1))
    leaderboard['Period'] = pd.to_datetime(leaderboard['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    leaderboard['Firm'] = leaderboard['firm']
    value_col = value_label or pretty_label
    leaderboard[value_col] = leaderboard[metric_col]

    display_cols = ['Position', 'Firm', 'Period', value_col]
    print(f"\nTop {top_n} period-firm observations by {pretty_label}:")
    print(leaderboard[display_cols].to_string(index=False))


def get_top_firms_by_metric(panel_df: pd.DataFrame, metric_col: str, top_n: int = 5) -> set[str]:
    """Return a set of firm names that appear in the top-N observations for the given metric."""
    required_cols = {'firm', 'Date', metric_col}
    if not required_cols.issubset(panel_df.columns):
        return set()

    leaderboard = panel_df[['firm', 'Date', metric_col]].dropna().copy()
    if leaderboard.empty:
        return set()

    top_rows = leaderboard.sort_values(metric_col, ascending=False).head(top_n)
    return set(top_rows['firm'].astype(str))


# Leaderboards for key metrics
print_top_metric_leaderboard(df_panel, 'Revenue_Growth', 'Revenue Growth')
print_top_metric_leaderboard(df_panel, 'Acq_Rate', 'Acquisition Rate')
print_top_metric_leaderboard(df_panel, 'RRR', 'Revenue Retention Rate (RRR)', value_label='RRR Value')


# Remove firms that rank in the top 5 for any core metric (sequential pooling)
pool_top_firms: set[str] = set()
selection_log = []

metric_sequence = [
    ('Revenue_Growth', 'Revenue Growth'),
    ('Acq_Rate', 'Acquisition Rate'),
    ('RRR', 'Revenue Retention Rate (RRR)')
]

for metric_col, pretty in metric_sequence:
    top_set = get_top_firms_by_metric(df_panel, metric_col, top_n=5)
    new_firms = sorted(top_set - pool_top_firms)
    pool_top_firms |= top_set
    selection_log.append((pretty, list(sorted(top_set)), new_firms, len(pool_top_firms)))

if pool_top_firms:
    firms_before = df_panel['firm'].nunique()
    rows_before = len(df_panel)

    df_panel = df_panel[~df_panel['firm'].isin(pool_top_firms)].copy()

    firms_after = df_panel['firm'].nunique()
    rows_after = len(df_panel)

    print("\nTop-5 firm pooling by metric (sequential):")
    for pretty, top_list, new_firms, cum_count in selection_log:
        print(f"  {pretty}: picked {len(top_list)} (new this step: {len(new_firms)} -> {new_firms}); cumulative unique: {cum_count}")

    print(f"\nFiltered out pooled top firms ({len(pool_top_firms)} unique).")
    print(f"Rows: {rows_before} -> {rows_after}; Firms: {firms_before} -> {firms_after}")
else:
    print("\nNo top-5 firms identified for removal (missing data or columns).")




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
# Firm-level average RRR and sub-industry shares of positive average RRR
# --------------------------------------------------------------------
if 'RRR' in df_panel.columns:
    df_firm_rrr_avg = (
        df_panel[['firm', 'RRR']]
        .dropna(subset=['RRR'])
        .groupby('firm', as_index=False)['RRR']
        .mean()
        .rename(columns={'RRR': 'avg_RRR'})
    )

    total_firms_rrr = len(df_firm_rrr_avg)
    positive_firms_rrr = (df_firm_rrr_avg['avg_RRR'] > 0).sum()
    print(f"\nFirms with positive average RRR: {positive_firms_rrr} of {total_firms_rrr}")

    if 'sub_industry' in df_panel.columns:
        firm_sub = (
            df_panel[['firm', 'sub_industry']]
            .dropna(subset=['firm', 'sub_industry'])
            .groupby('firm')['sub_industry']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .reset_index()
        )

        df_firm_rrr_avg = df_firm_rrr_avg.merge(firm_sub, on='firm', how='left')

        sub_shares = (
            df_firm_rrr_avg.groupby('sub_industry')
            .agg(
                n_firms=('firm', 'size'),
                n_positive=('avg_RRR', lambda s: (s > 0).sum())
            )
            .reset_index()
        )
        sub_shares['share_positive'] = sub_shares['n_positive'] / sub_shares['n_firms']

        if positive_firms_rrr > 0:
            sub_shares['share_of_total_positive'] = sub_shares['n_positive'] / positive_firms_rrr
        else:
            sub_shares['share_of_total_positive'] = np.nan

        sub_shares = sub_shares.sort_values('share_positive', ascending=False)

        print('\nShare of firms with positive average RRR by sub-industry:')
        print(sub_shares[['sub_industry', 'n_firms', 'n_positive', 'share_positive', 'share_of_total_positive']].to_string(index=False))
    else:
        print("Sub-industry column missing; cannot compute sub-industry shares of positive average RRR.")

    # Industry-level shares (using industry column if available)
    if 'industry' in df_panel.columns:
        firm_ind = (
            df_panel[['firm', 'industry']]
            .dropna(subset=['firm', 'industry'])
            .groupby('firm')['industry']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .reset_index()
        )

        df_firm_rrr_ind = df_firm_rrr_avg.merge(firm_ind, on='firm', how='left')

        ind_shares = (
            df_firm_rrr_ind.groupby('industry')
            .agg(
                n_firms=('firm', 'size'),
                n_positive=('avg_RRR', lambda s: (s > 0).sum())
            )
            .reset_index()
        )
        ind_shares['share_positive'] = ind_shares['n_positive'] / ind_shares['n_firms']

        if positive_firms_rrr > 0:
            ind_shares['share_of_total_positive'] = ind_shares['n_positive'] / positive_firms_rrr
        else:
            ind_shares['share_of_total_positive'] = np.nan

        ind_shares = ind_shares.sort_values('share_positive', ascending=False)

        print('\nShare of firms with positive average RRR by industry:')
        print(ind_shares[['industry', 'n_firms', 'n_positive', 'share_positive', 'share_of_total_positive']].to_string(index=False))
    else:
        print("Industry column missing; cannot compute industry shares of positive average RRR.")
else:
    print('RRR column missing; cannot compute firm-level average RRR summary.')


# --------------------------------------------------------------------
# Randomness diagnostics: Ljung–Box (lag 1) and AR(1) vs mean AIC, per metric
# Metrics tested: Revenue_Growth/cRG, Acq_Rate/cAR, RRR/cRRR
# --------------------------------------------------------------------
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception as e:
    print(f"Statsmodels diagnostic import failed; skipping randomness diagnostics: {e}")
else:
    # Test only raw (non-centered) metrics as requested
    metric_groups = [
        ['Revenue_Growth'],
        ['Acq_Rate'],
        ['RRR']
    ]

    for candidates in metric_groups:
        metric_for_randomness = candidates[0]
        if metric_for_randomness not in df_panel.columns:
            print(f"Randomness diagnostics skipped: {metric_for_randomness} not found in df_panel.")
            continue

        tests = []
        grouped = df_panel[['firm', metric_for_randomness]].dropna().copy().groupby('firm')

        for firm, sub in grouped:
            series = pd.to_numeric(sub[metric_for_randomness], errors='coerce').dropna()
            if len(series) < 6:
                continue  # need a few observations to test

            # Ljung–Box at lag 1
            try:
                lb_res = acorr_ljungbox(series, lags=[1], return_df=True)
                lb_p = lb_res['lb_pvalue'].iloc[0]
            except Exception:
                lb_p = np.nan

            # AR(1) vs mean-only model AIC comparison
            aic_mean = np.nan
            aic_ar1 = np.nan
            try:
                mean_fit = sm.tsa.arima.model.ARIMA(series, order=(0, 0, 0)).fit()
                aic_mean = mean_fit.aic
            except Exception:
                pass

            try:
                ar1_fit = sm.tsa.arima.model.ARIMA(series, order=(1, 0, 0)).fit()
                aic_ar1 = ar1_fit.aic
            except Exception:
                pass

            tests.append({
                'firm': firm,
                'n_obs': len(series),
                'lb_p': lb_p,
                'aic_mean': aic_mean,
                'aic_ar1': aic_ar1
            })

        if tests:
            df_tests = pd.DataFrame(tests)
            tested_firms = len(df_tests)
            lb_reject = (df_tests['lb_p'] < 0.05).sum()
            ar1_better = (df_tests['aic_ar1'] < df_tests['aic_mean']).sum()

            print(f"\nRandomness diagnostics on '{metric_for_randomness}' (firms with >=6 observations): {tested_firms} firms")
            print(f"  Ljung–Box lag-1 rejects (p<0.05): {lb_reject} ({lb_reject/tested_firms:.1%})")
            print(f"  AR(1) AIC better than mean-only: {ar1_better} ({ar1_better/tested_firms:.1%})")

            top_lb = df_tests.sort_values('lb_p').head(5)
            print('\nTop firms by lowest Ljung–Box p-values (structure indication):')
            print(top_lb[['firm', 'n_obs', 'lb_p']].to_string(index=False))
        else:
            print(f"Randomness diagnostics skipped: not enough data (need >=6 observations per firm) for '{metric_for_randomness}'.")


# --------------------------------------------------------------------
# Generic AR(1) helper to keep code compact
# --------------------------------------------------------------------
try:
    import statsmodels.api as sm
except Exception as e:
    sm = None
    print(f"Statsmodels not available; skipping AR(1) tests: {e}")


def run_ar1_metric(panel: pd.DataFrame, metric: str, pretty: str) -> None:
    if sm is None:
        return
    if metric not in panel.columns:
        print(f"{metric} column missing; cannot run {pretty} AR(1) test.")
        return

    if 'period_year' not in panel.columns:
        print(f"period_year column missing; cannot add time trend for {pretty} AR(1) test.")
        return

    if 'Total_Revenue' not in panel.columns:
        print(f"Total_Revenue column missing; cannot add size control for {pretty} AR(1) test.")
        return

    metric_counts = panel[['firm', metric]].dropna(subset=[metric]).groupby('firm').size()
    n_ge2 = (metric_counts >= 2).sum()
    n_ge3 = (metric_counts >= 3).sum()
    print(f"{pretty} availability: firms with >=2 obs: {n_ge2}; firms with >=3: {n_ge3}")

    grouped = (
        panel[['firm', 'Date', 'period_year', metric, 'Total_Revenue']]
        .dropna(subset=[metric, 'period_year', 'Total_Revenue'])
        .sort_values(['firm', 'Date'])
        .groupby('firm')
    )

    rows = []
    for firm, sub in grouped:
        sub = sub.copy()
        sub['lag'] = sub[metric].shift(1)
        sub['log_size'] = np.log1p(sub['Total_Revenue'])
        sub = sub.dropna(subset=[metric, 'lag', 'period_year', 'log_size'])
        if len(sub) < 2:
            continue
        try:
            X = sm.add_constant(pd.DataFrame({
                'lag': sub['lag'],
                'trend': sub['period_year'],
                'log_size': sub['log_size']
            }))
            y = sub[metric]
            res = sm.OLS(y, X, missing='drop').fit()
            lag_beta = res.params.get('lag', np.nan)
            lag_pval = res.pvalues.get('lag', np.nan)
            trend_beta = res.params.get('trend', np.nan)
            trend_pval = res.pvalues.get('trend', np.nan)
            size_beta = res.params.get('log_size', np.nan)
            size_pval = res.pvalues.get('log_size', np.nan)
            const_beta = res.params.get('const', np.nan)
            const_pval = res.pvalues.get('const', np.nan)
            rows.append({
                'firm': firm,
                'n_obs': len(sub),
                'lag_beta': lag_beta,
                'lag_pval': lag_pval,
                'trend_beta': trend_beta,
                'trend_pval': trend_pval,
                'size_beta': size_beta,
                'size_pval': size_pval,
                'const_beta': const_beta,
                'const_pval': const_pval
            })
        except Exception:
            continue

    if rows:
        df_res = pd.DataFrame(rows)
        tested = len(df_res)
        lag_signif_10 = (df_res['lag_pval'] < 0.10).sum()
        trend_signif_10 = (df_res['trend_pval'] < 0.10).sum()
        size_signif_10 = (df_res['size_pval'] < 0.10).sum()
        const_signif_10 = (df_res['const_pval'] < 0.10).sum()
        print(f"\n{pretty} AR(1) test (firms with >=2 usable observations): {tested} firms")
        print(f"  Firms with lag beta significant at 10%: {lag_signif_10} ({lag_signif_10/tested:.1%})")
        print(f"  Firms with time-trend significant at 10%: {trend_signif_10} ({trend_signif_10/tested:.1%})")
        print(f"  Firms with size (log revenue) significant at 10%: {size_signif_10} ({size_signif_10/tested:.1%})")
        print(f"  Firms with constant significant at 10%: {const_signif_10} ({const_signif_10/tested:.1%})")
    else:
        print(f"{pretty} AR(1) test skipped: not enough data to estimate for any firm.")


# Run AR(1) for the three metrics without per-firm listings
run_ar1_metric(df_panel, 'RRR', 'RRR')
run_ar1_metric(df_panel, 'Revenue_Growth', 'Revenue_Growth')
run_ar1_metric(df_panel, 'Acq_Rate', 'Acq_Rate')


# --------------------------------------------------------------------
# PanelOLS regressions with lag, time trend, and size (log total revenue)
# --------------------------------------------------------------------
def run_panel_reg_metric(panel: pd.DataFrame, metric: str, pretty: str) -> None:
    try:
        from linearmodels.panel import PanelOLS
    except Exception as e:
        print(f"{pretty} panel regression skipped: linearmodels not available ({e}).")
        return

    required = {'firm', 'Date', 'period_year', 'Total_Revenue', metric}
    if not required.issubset(panel.columns):
        missing = required.difference(panel.columns)
        print(f"{pretty} panel regression skipped: missing columns {sorted(missing)}")
        return

    df = panel[['firm', 'Date', 'period_year', 'Total_Revenue', metric]].dropna().copy()
    if df.empty:
        print(f"{pretty} panel regression skipped: no complete observations after dropping NaNs.")
        return

    df = df.sort_values(['firm', 'Date'])
    df['lag'] = df.groupby('firm')[metric].shift(1)
    df['trend'] = df['period_year']
    df['log_size'] = np.log1p(df['Total_Revenue'])
    df = df.dropna(subset=['lag', 'trend', 'log_size', metric])
    if df.empty:
        print(f"{pretty} panel regression skipped: no rows after requiring lag/trend/size.")
        return

    df = df.set_index(['firm', 'Date'])
    y = df[metric]
    X = sm.add_constant(df[['lag', 'trend', 'log_size']])

    try:
        model = PanelOLS(y, X, entity_effects=True, time_effects=False, drop_absorbed=True)
        results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=False)
        print(f"\n{pretty} panel regression (PanelOLS pooled with linear time trend + size; clustered by firm)")
        print(f"  N obs: {results.nobs} | Firms: {df.index.get_level_values(0).nunique()}")
        print(results.summary)
    except Exception as e:
        print(f"{pretty} panel regression failed: {e}")


# Run panel regressions on the three metrics
run_panel_reg_metric(df_panel, 'RRR', 'RRR')
run_panel_reg_metric(df_panel, 'Revenue_Growth', 'Revenue_Growth')
run_panel_reg_metric(df_panel, 'Acq_Rate', 'Acq_Rate')



# --------------------------------------------------------------------
# Histograms for core metrics (distribution across firm-periods)
# --------------------------------------------------------------------
metric_specs = [
    ('RRR', 'Revenue Retention Rate (RRR)'),
    ('Acq_Rate', 'Acquisition Rate (AR)'),
    ('Revenue_Growth', 'Revenue Growth (RG)')
]

available_hist_metrics = [m for m, _ in metric_specs if m in df_panel.columns]

if available_hist_metrics:
    # summary table (mean, std, quartiles) for each available metric
    summary_rows = []
    for metric, pretty_label in metric_specs:
        if metric not in available_hist_metrics:
            continue
        series = pd.to_numeric(df_panel[metric], errors='coerce').dropna()
        if series.empty:
            continue
        summary_rows.append({
            'metric': pretty_label,
            'mean': series.mean(),
            'std': series.std(ddof=1),
            '25Q': series.quantile(0.25),
            '50Q': series.quantile(0.50),
            '75Q': series.quantile(0.75),
            'n_obs': len(series)
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print('\nDistribution summary for core metrics:')
        print(summary_df.to_string(index=False))

    for metric, pretty_label in metric_specs:
        if metric not in available_hist_metrics:
            print(f"Metric {metric} not available for histogram; skipping.")
            continue

        metric_values = pd.to_numeric(df_panel[metric], errors='coerce').dropna()
        metric_values = metric_values[(metric_values >= -2) & (metric_values <= 3)]
        if metric_values.empty:
            print(f"No data available for {pretty_label}; skipping histogram.")
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(metric_values, bins=40, kde=True, color='black')
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f"Distribution of {pretty_label}")
        plt.xlabel(pretty_label)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
else:
    print('No core metrics available for histogram plotting.')



# --------------------------------------------------------------------
# Sub-industry boxplots for core metrics (hide outliers)
# --------------------------------------------------------------------

if 'sub_industry' in df_panel.columns:
    available_specs = [(m, label) for m, label in metric_specs if m in df_panel.columns]
    if available_specs:
        base_cols = ['sub_industry'] + [m for m, _ in available_specs]
        df_box = df_panel[base_cols].copy()
        df_box = df_box[df_box['sub_industry'].notna()]

        def _plot_boxplots(data, subset_label):
            for metric, pretty_label in available_specs:
                plot_df = data[['sub_industry', metric]].dropna()
                if plot_df.empty:
                    print(f"No data available for {pretty_label} ({subset_label}); skipping boxplot.")
                    continue

                n_subs = plot_df['sub_industry'].nunique()
                # dynamic height: 0.4 per subindustry, clipped to sensible bounds
                height = min(max(0.4 * n_subs, 4), 25)

                plt.figure(figsize=(10, height))
                sns.boxplot(
                    data=plot_df,
                    y='sub_industry',
                    x=metric,
                    orient='h',
                    showfliers=False,
                    color='#1f77b4'
                )
                plt.title(f"{pretty_label} by Sub-industry ({subset_label})")
                plt.xlabel(pretty_label)
                plt.ylabel('Sub-industry')
                plt.tight_layout()
                plt.show()

        print('\nBoxplots by sub-industry (all sub-industries):')
        _plot_boxplots(df_box, 'All Sub-industries')

        firm_counts = (
            df_panel[['firm', 'sub_industry']]
            .dropna()
            .drop_duplicates()
            .groupby('sub_industry')['firm']
            .nunique()
            .sort_values(ascending=False)
        )
        top5_subindustries = firm_counts.head(5).index.tolist()

        if top5_subindustries:
            df_box_top5 = df_box[df_box['sub_industry'].isin(top5_subindustries)].copy()
            print('\nBoxplots by sub-industry (top 5 by firm count):')
            _plot_boxplots(df_box_top5, 'Top 5 Sub-industries by Firm Count')
        else:
            print('Top-5 sub-industry list is empty; skipping focused boxplots.')
    else:
        print('Core metrics missing in df_panel; skipping sub-industry boxplots.')
else:
    print("Cannot create sub-industry boxplots: 'sub_industry' column missing in df_panel.")





    # --------------------------------------------------------------------
    # Firm-level time-trend regressions (metric ~ calendar year)
    # --------------------------------------------------------------------
    metrics_trend = [
        ('Revenue_Growth', 'Revenue Growth (RG)'),
        ('Acq_Rate', 'Acquisition Rate (AR)'),
        ('RRR', 'Revenue Retention Rate (RRR)')
    ]

    if 'period_year' not in df_panel.columns:
        print("Cannot run time-trend regressions: 'period_year' column missing.")
    else:
        try:
            from scipy import stats
        except Exception as e:
            print(f"Scipy not available; skipping time-trend regressions: {e}")
        else:
            def _compute_trend_stats(metric: str, pretty_label: str) -> pd.DataFrame:
                if metric not in df_panel.columns:
                    print(f"{pretty_label} column missing; skipping time-trend regression.")
                    return pd.DataFrame()

                sub = df_panel[['firm', 'period_year', metric]].dropna()
                if sub.empty:
                    print(f"No data available for {pretty_label}; skipping time-trend regression.")
                    return pd.DataFrame()

                sub = sub.rename(columns={metric: 'y', 'period_year': 'x'})
                sub['y'] = pd.to_numeric(sub['y'], errors='coerce')
                sub['x'] = pd.to_numeric(sub['x'], errors='coerce')
                sub = sub.dropna(subset=['x', 'y'])
                if sub.empty:
                    print(f"No numeric data for {pretty_label}; skipping time-trend regression.")
                    return pd.DataFrame()

                sub['x2'] = sub['x'] ** 2
                sub['y2'] = sub['y'] ** 2
                sub['xy'] = sub['x'] * sub['y']

                agg = (
                    sub.groupby('firm')
                    .agg(
                        n=('x', 'size'),
                        sum_x=('x', 'sum'),
                        sum_y=('y', 'sum'),
                        sum_x2=('x2', 'sum'),
                        sum_y2=('y2', 'sum'),
                        sum_xy=('xy', 'sum')
                    )
                    .reset_index()
                )

                agg['Sxx'] = agg['sum_x2'] - (agg['sum_x'] ** 2) / agg['n']
                agg['Sxy'] = agg['sum_xy'] - (agg['sum_x'] * agg['sum_y']) / agg['n']
                agg['Syy'] = agg['sum_y2'] - (agg['sum_y'] ** 2) / agg['n']

                agg = agg[(agg['n'] >= 3) & (agg['Sxx'] > 0)]
                if agg.empty:
                    print(f"Not enough observations per firm for {pretty_label}; skipping time-trend regression.")
                    return pd.DataFrame()

                agg['beta'] = agg['Sxy'] / agg['Sxx']
                agg['SSE'] = (agg['Syy'] - agg['beta'] * agg['Sxy']).clip(lower=0)
                agg['df'] = agg['n'] - 2
                agg = agg[agg['df'] > 0]
                if agg.empty:
                    print(f"Insufficient degrees of freedom for {pretty_label}; skipping time-trend regression.")
                    return pd.DataFrame()

                agg['sigma2'] = agg['SSE'] / agg['df']
                agg['se_beta'] = np.sqrt((agg['sigma2'] / agg['Sxx']).replace({np.inf: np.nan, -np.inf: np.nan}))
                agg = agg[agg['se_beta'].notna() & (agg['se_beta'] > 0)]
                if agg.empty:
                    print(f"Could not compute standard errors for {pretty_label}; skipping time-trend regression.")
                    return pd.DataFrame()

                agg['t_stat'] = agg['beta'] / agg['se_beta']
                agg['p_value'] = 2 * stats.t.sf(np.abs(agg['t_stat']), agg['df'])

                return agg[['firm', 'n', 'beta', 'se_beta', 't_stat', 'p_value']]

            for metric, pretty_label in metrics_trend:
                trend_stats = _compute_trend_stats(metric, pretty_label)
                if trend_stats is None or trend_stats.empty:
                    continue

                sig_betas = trend_stats[trend_stats['p_value'] < 0.1]
                total_firms = len(trend_stats)
                sig_count = len(sig_betas)
                print(
                    f"Time-trend regression ({pretty_label}): {sig_count} of {total_firms} firms with significant slope (p<0.1)"
                )

                if sig_betas.empty:
                    print(f"No significant time trends detected for {pretty_label}.")
                    continue

                plt.figure(figsize=(8, 4))
                plt.hist(sig_betas['beta'], bins=30, color='#1f77b4', edgecolor='white')
                plt.axvline(0, color='black', linestyle='--', linewidth=1)
                plt.title(f"Significant time-trend slopes for {pretty_label}")
                plt.xlabel('Slope (beta)')
                plt.ylabel('Number of firms')
                plt.tight_layout()
                plt.show()



# --------------------------------------------------------------------
# AR vs. RRR correlation decomposition and sub-industry-aware models
# --------------------------------------------------------------------

if {'Acq_Rate', 'RRR', 'sub_industry'}.issubset(df_panel.columns):
    corr_df = df_panel[['sub_industry', 'Acq_Rate', 'RRR']].copy()
    corr_df[['Acq_Rate', 'RRR']] = corr_df[['Acq_Rate', 'RRR']].apply(pd.to_numeric, errors='coerce')
    corr_df = corr_df.dropna()

    if corr_df.empty:
        print('AR/RRR correlation analysis skipped: no overlapping observations after dropping NaNs.')
    else:
        overall_corr = corr_df['Acq_Rate'].corr(corr_df['RRR'])
        print(f"\nOverall AR–RRR correlation (pooled): {overall_corr:.3f}")

        sub_corr = (
            corr_df.groupby('sub_industry')
            .apply(lambda g: g['Acq_Rate'].corr(g['RRR']) if len(g) >= 3 else np.nan)
            .dropna()
        )

        if not sub_corr.empty:
            print('Per-sub-industry correlation (n>=3 periods) summary:')
            print(sub_corr.describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        else:
            print('Not enough observations per sub-industry to compute within-industry correlations.')

        # Decompose into within- and between-sub-industry correlations
        group_means = corr_df.groupby('sub_industry')[['Acq_Rate', 'RRR']].transform('mean')
        corr_df['Acq_Rate_within'] = corr_df['Acq_Rate'] - group_means['Acq_Rate']
        corr_df['RRR_within'] = corr_df['RRR'] - group_means['RRR']
        within_corr = corr_df['Acq_Rate_within'].corr(corr_df['RRR_within'])

        sub_means = corr_df.groupby('sub_industry')[['Acq_Rate', 'RRR']].mean().dropna()
        between_corr = sub_means['Acq_Rate'].corr(sub_means['RRR']) if len(sub_means) > 1 else np.nan

        print(
            f"Within-sub-industry correlation (demeaned): {within_corr:.3f} | "
            f"Between-sub-industry correlation (centroids): {between_corr:.3f}"
        )

        # Regression-based models removed per latest request; relying on correlation summaries above.
else:
    print("Cannot assess AR/RRR correlation by industry: required columns missing.")


# --------------------------------------------------------------------
# Pooled OLS: Revenue_Growth ~ Acq_Rate + RRR (clustered SEs by firm)
# --------------------------------------------------------------------
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except Exception as e:
    print(f"Statsmodels not available; skipping pooled Revenue_Growth regression: {e}")
else:
    reg_cols = ['Date', 'firm', 'Revenue_Growth', 'Acq_Rate', 'RRR']
    if not set(reg_cols).issubset(set(df_panel.columns)):
        print("Not all regression columns present (need Date, firm, Revenue_Growth, Acq_Rate, RRR); skipping regression.")
    else:
        df_reg_rg = df_panel[reg_cols].dropna().copy()
        if df_reg_rg.empty:
            print("Revenue_Growth regression skipped: no complete observations after dropping NaNs.")
        else:
            df_reg_rg['Year'] = df_reg_rg['Date'].dt.year
            n_obs_rg = len(df_reg_rg)
            n_firms_rg = df_reg_rg['firm'].nunique()
            n_years_rg = df_reg_rg['Year'].nunique()
            print(f"\nRunning pooled OLS for Revenue_Growth on {n_obs_rg} obs, {n_firms_rg} firms, {n_years_rg} years (clustered SEs by firm)")

            formula_rg = 'Revenue_Growth ~ Acq_Rate + RRR'
            try:
                mod_rg = smf.ols(formula=formula_rg, data=df_reg_rg)
                res_rg = mod_rg.fit()

                try:
                    res_rg_clust = res_rg.get_robustcov_results(cov_type='cluster', groups=df_reg_rg['firm'])
                    keep_vars_rg = list(res_rg_clust.params.index)
                    coef_df_rg = pd.DataFrame({
                        'coef': res_rg_clust.params.loc[keep_vars_rg],
                        'se_cluster': res_rg_clust.bse.loc[keep_vars_rg],
                        't_cluster': res_rg_clust.tvalues.loc[keep_vars_rg],
                        'p_cluster': res_rg_clust.pvalues.loc[keep_vars_rg]
                    })
                    ci_rg = res_rg_clust.conf_int().loc[keep_vars_rg]
                    coef_df_rg['ci_lower'] = ci_rg[0]
                    coef_df_rg['ci_upper'] = ci_rg[1]

                    print('\nCluster-robust OLS (Revenue_Growth ~ Acq_Rate + RRR):')
                    print(coef_df_rg.round(4))
                except Exception as e_clust_rg:
                    print(f"Could not compute clustered SEs (firm level) for Revenue_Growth regression: {e_clust_rg}\nFalling back to plain OLS summary:")
                    print(res_rg.summary())
            except Exception as e_mod_rg:
                print(f"Revenue_Growth regression estimation failed: {e_mod_rg}")







#==============================================================================
# RQ:2 Firm-Level Strategy
#==============================================================================

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




"""
share of retained revenue time invariant?
RRR autocorrelation
AR autocorrelation

"""






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
# Firm-level share regression:
# share_positive_revenue_growth ~ growth-path shares (baseline = Shrinking share)
# --------------------------------------------------------------------
required_cols = {'firm', 'growth_path', 'revenue_growth'}
if not required_cols.issubset(df_panel.columns):
    print('Cannot run share regression: missing firm/growth_path/revenue_growth columns in df_panel.')
else:
    gp_df = df_panel[['firm', 'growth_path', 'revenue_growth']].dropna()
    if gp_df.empty:
        print('Share regression skipped: no overlapping rows after dropping NaNs.')
    else:
        growth_states = ['Acquisition Driven', 'Retention Driven', 'Dual Engine', 'Shrinking']
        share_rows = []
        for firm, sub in gp_df.groupby('firm'):
            total = len(sub)
            if total == 0:
                continue
            share_positive = (sub['revenue_growth'] == 'Increasing').mean()
            path_shares = (sub['growth_path'].value_counts(normalize=True)
                           .reindex(growth_states, fill_value=0.0))
            share_rows.append({
                'firm': firm,
                'share_positive': share_positive,
                'share_acq': path_shares['Acquisition Driven'],
                'share_ret': path_shares['Retention Driven'],
                'share_dual': path_shares['Dual Engine'],
                'share_shrink': path_shares['Shrinking'],
                'n_periods': total
            })

        df_shares = pd.DataFrame(share_rows).dropna()
        df_shares = df_shares[df_shares['n_periods'] >= 3]

        if df_shares.empty:
            print('Share regression skipped: no firms with at least 3 periods of data.')
        else:
            try:
                import statsmodels.formula.api as smf
            except Exception as e:
                print(f"Statsmodels formula API unavailable; skipping share regression: {e}")
            else:
                try:
                    formula = 'share_positive ~ share_acq + share_ret + share_dual'
                    share_mod = smf.ols(formula=formula, data=df_shares).fit()
                    print('\nShare regression (Shrinking share = baseline):')
                    print(share_mod.summary().tables[1])

                    diag_stats = {
                        'n_obs': int(share_mod.nobs),
                        'r_squared': share_mod.rsquared,
                        'adj_r_squared': share_mod.rsquared_adj,
                        'f_stat': share_mod.fvalue,
                        'f_pvalue': share_mod.f_pvalue,
                        'rmse': np.sqrt(share_mod.scale) if hasattr(share_mod, 'scale') else np.nan
                    }
                    print('\nShare regression diagnostics:')
                    for key, value in diag_stats.items():
                        print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) and not pd.isna(value)
                              else f"  {key}: {value}")
                except Exception as e:
                    print(f'Share regression failed: {e}')

    


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
                    annot_kws={'size': 10, 'weight': 'bold'},
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

