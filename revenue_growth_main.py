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
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Ensure plots render in VS Code Interactive/Jupyter contexts
try:
    from IPython import get_ipython

    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic('matplotlib', 'inline')
except Exception:
    pass

plt.ion()

ANALYSIS_END_YEAR = 2025
DISPLAY_N_FIRMS = 591

# Quarterly Revenue
file1_path = "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)\\Privat\\Research\\Revenue_Growth\\RG_Data\\2026-02-18-Firm-Year-Revenue-Python.xlsx"
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

# Keep only firms with valid sub-industry and sub-industries with at least 2 firms
df_industry['gics_sub_industry_name'] = (
    df_industry['gics_sub_industry_name']
    .astype(str)
    .str.strip()
    .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
)

n_before_industry_filter = df_industry['ID'].nunique()

# 1) drop firms without sub-industry
df_industry = df_industry.dropna(subset=['gics_sub_industry_name']).copy()

# 2) drop sub-industries with only one firm
sub_counts = df_industry.groupby('gics_sub_industry_name')['ID'].nunique()
keep_subs = sub_counts[sub_counts >= 2].index
df_industry = df_industry[df_industry['gics_sub_industry_name'].isin(keep_subs)].copy()

n_after_industry_filter = df_industry['ID'].nunique()
print(
    "Industry filter applied: "
    f"removed {n_before_industry_filter - n_after_industry_filter} firms; "
    f"{n_after_industry_filter} remain."
)

# #of Firms
n_firms_industry_after = df_industry["ID"].nunique()
print("Number of Firms:", n_firms_industry_after)

# Normalized set of valid IDs for the Revenue dataset
def _norm_id(x) -> str:
    s = str(x).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def attach_industry_labels(panel_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    """Attach industry to panel by robust firm-ID matching."""
    if panel_df is None or panel_df.empty:
        return panel_df
    if industry_df is None or industry_df.empty:
        print("Warning: industry merge skipped (industry dataset empty).")
        return panel_df
    if 'firm' not in panel_df.columns:
        print("Warning: industry merge skipped ('firm' column missing in panel).")
        return panel_df
    if not {'ID', 'gics_sub_industry_name'}.issubset(industry_df.columns):
        print("Warning: industry merge skipped (industry file missing ID/gics_sub_industry_name).")
        return panel_df

    out = panel_df.copy()
    ind = industry_df[['ID', 'gics_sub_industry_name']].copy()
    ind['id_norm'] = ind['ID'].apply(_norm_id)
    ind = ind.dropna(subset=['id_norm', 'gics_sub_industry_name']).drop_duplicates('id_norm')
    id_to_industry = ind.set_index('id_norm')['gics_sub_industry_name']

    firm_raw = out['firm'].astype(str)
    key_direct = firm_raw.apply(_norm_id)
    key_before_dot = firm_raw.str.split('.').str[0].apply(_norm_id)
    key_after_dot = firm_raw.str.split('.').str[-1].apply(_norm_id)

    out['industry'] = key_direct.map(id_to_industry)
    out['industry'] = out['industry'].fillna(key_before_dot.map(id_to_industry))
    out['industry'] = out['industry'].fillna(key_after_dot.map(id_to_industry))

    matched = int(out['industry'].notna().sum())
    if matched == 0:
        sample_firms = out['firm'].dropna().astype(str).head(5).tolist()
        print(f"Warning: industry merge found 0 matches. Sample firm values: {sample_firms}")

    return out

valid_ids_norm = set(_norm_id(x) for x in df_industry["ID"].dropna())

# Select columns: always keep date column (index 0) + firm columns
# Robust: try matching IDs from BOTH header rows (row 0 and row 1)
cols_to_keep = [0]
max_header_row = min(2, len(df_revenue.index))
for col_idx in df_revenue.columns:
    if col_idx == 0:
        continue

    header_candidates = []
    for r in range(max_header_row):
        v = df_revenue.iloc[r, col_idx]
        if pd.notna(v):
            header_candidates.append(_norm_id(v))

    if any(h in valid_ids_norm for h in header_candidates):
        cols_to_keep.append(col_idx)

# Fallback: if nothing matched, keep non-empty columns so pipeline can proceed
if len(cols_to_keep) == 1:
    fallback_cols = [0]
    for col_idx in df_revenue.columns[1:]:
        vals = [df_revenue.iloc[r, col_idx] for r in range(max_header_row)]
        if any(pd.notna(v) and str(v).strip() != "" for v in vals):
            fallback_cols.append(col_idx)
    cols_to_keep = fallback_cols
    print(
        "Warning: no firm columns matched industry IDs in header rows; "
        "using non-empty fallback columns."
    )

df_revenue = df_revenue.iloc[:, cols_to_keep].copy()

# Count unique firms in revenue data after filtering
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
print("Number of Firms in Revenue:", len(normalized_firm_names_revenue_after))





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
combined_headers = header_rows_rev.apply(lambda x: x.astype(str).str.strip())
combined_headers = combined_headers.replace({'nan': pd.NA, 'NaT': pd.NA, 'None': pd.NA, '': pd.NA})
column_headers = combined_headers.apply(
    lambda x: '.'.join(x.dropna()),
    axis=0
)
data_rows_rev.columns = column_headers;  # Assign proper headers to the data rows

# Rename the first column to "Date" (avoid inplace to prevent chained-assignment warnings)
data_rows_rev = data_rows_rev.rename(columns={data_rows_rev.columns[0]: 'Date'})

# Reset the index for the data rows
df_revenue = data_rows_rev.reset_index(drop=True);

### replace empty cells with NaN
df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True)

# Ensure valid yearly rows only
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'], errors='coerce')
df_revenue = df_revenue[df_revenue['Date'].notna()].copy()
df_revenue = df_revenue[df_revenue['Date'].dt.year <= ANALYSIS_END_YEAR].copy()
df_revenue = df_revenue.sort_values('Date').reset_index(drop=True)

# Identify columns related to New/Returning
new_customer_cols = [c for c in df_revenue.columns if '#New_Customers' in c or '#New_Customer' in c]
returning_customer_cols = [c for c in df_revenue.columns if '#Returning_Customers' in c or '#Returning_Customer' in c]

def _base_name(col_name: str) -> str:
    return (
        str(col_name)
        .replace('#New_Customers', '')
        .replace('#New_Customer', '')
        .replace('#Returning_Customers', '')
        .replace('#Returning_Customer', '')
        .lstrip('#')
        .strip('. ')
    )

# Simple strict filter: drop firm if any cell is blank/non-numeric or literal 'missing value'
candidate_firms = sorted({_base_name(c) for c in (new_customer_cols + returning_customer_cols)})
firms_keep = []

for f in candidate_firms:
    new_col = next((c for c in new_customer_cols if _base_name(c) == f), None)
    ret_col = next((c for c in returning_customer_cols if _base_name(c) == f), None)
    if new_col is None or ret_col is None:
        continue

    new_raw = df_revenue[new_col].astype(str).str.strip().str.lower()
    ret_raw = df_revenue[ret_col].astype(str).str.strip().str.lower()

    new_num = pd.to_numeric(df_revenue[new_col], errors='coerce')
    ret_num = pd.to_numeric(df_revenue[ret_col], errors='coerce')

    invalid_new = new_raw.eq('missing value') | new_num.isna()
    invalid_ret = ret_raw.eq('missing value') | ret_num.isna()

    if not (invalid_new.any() or invalid_ret.any()):
        firms_keep.append(f)

firms_drop = sorted(set(candidate_firms) - set(firms_keep))
if firms_drop:
    cols_to_drop = [c for c in df_revenue.columns if c != 'Date' and _base_name(c) in firms_drop]
    df_revenue = df_revenue.drop(columns=cols_to_drop)
    print(f"Dropped firms with blanks/'missing value': removed {len(firms_drop)}; {len(firms_keep)} remain.")
else:
    print("No firms dropped for blanks/'missing value'.")

# refresh lists after filtering
new_customer_cols = [c for c in df_revenue.columns if '#New_Customers' in c or '#New_Customer' in c]
returning_customer_cols = [c for c in df_revenue.columns if '#Returning_Customers' in c or '#Returning_Customer' in c]

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
        rrr_col = col.replace('#Returning_Customers', '#RRR')
        revenue_growth_col = col.replace('#Returning_Customers', '#Revenue_Growth')
        acq_rate_col = new_col.replace('#New_Customers', '#Acq_Rate')
        q_t_col = col.replace('#Returning_Customers', '#Q_t')
        sonr_col = new_col.replace('#New_Customers', '#SoNR')
        nrlc_col = new_col.replace('#New_Customers', '#NRLC')

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
        new_cols_data[rrr_col] = ret_s / total_prev_nonzero
        rev_growth = (total - total.shift(1)) / total_prev_nonzero
        new_cols_data[revenue_growth_col] = rev_growth
        acq_rate = new_s / new_s.shift(1)
        new_cols_data[acq_rate_col] = acq_rate
        new_cols_data[q_t_col] = (ret_s / total_prev_nonzero) / acq_rate.replace(0, np.nan)
        sonr = new_s / total_nonzero
        new_cols_data[sonr_col] = sonr
        new_cols_data[nrlc_col] = acq_rate * sonr.shift(1)

        # extend final_cols to include the new columns next to the returning-customer column
        final_cols.extend([
            total_revenue_col,
            rrr_col,
            revenue_growth_col,
            acq_rate_col,
            q_t_col,
            sonr_col,
            nrlc_col,
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

###################
# 3.1 Revenue Data#
###################

# --- YEARLY INPUT: df_revenue is already yearly wide-format ---
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'], errors='coerce')
df_revenue = df_revenue.sort_values('Date').reset_index(drop=True)

# Metrics are already calculated in section 2.1; no separate df_yearly needed.
metrics_suffixes = ['#Revenue_Growth', '#RRR', '#Acq_Rate', '#Q_t', '#Total_Revenue', '#SoNR', '#NRLC']
present_metric_cols = [c for c in df_revenue.columns if any(c.endswith(s) for s in metrics_suffixes)]
if not present_metric_cols:
    print("Warning: no metric columns found in df_revenue (expected from section 2.1).")
else:
    print(f"Confirmed: {len(present_metric_cols)} metric columns present in df_revenue.")

### --- Transform yearly wide dataframe to panel (long) format ---

df_wide = df_revenue.copy()
df_wide['Date'] = pd.to_datetime(df_wide['Date'], errors='coerce')
metric_cols = [c for c in df_wide.columns if c in present_metric_cols]

df_long = df_wide.melt(id_vars='Date', value_vars=metric_cols, var_name='col', value_name='value')
df_long['firm'] = df_long['col'].apply(lambda c: c.split('#')[0] if '#' in c else c)
df_long['metric'] = df_long['col'].apply(lambda c: c.split('#', 1)[1] if '#' in c else '')

df_panel = df_long.pivot_table(index=['Date', 'firm'], columns='metric', values='value', aggfunc='first').reset_index()
df_panel.columns = [c if isinstance(c, str) else c[1] for c in df_panel.columns]
for col in ['Revenue_Growth', 'RRR', 'Acq_Rate', 'Q_t', 'Total_Revenue', 'SoNR', 'NRLC']:
    if col in df_panel.columns:
        df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce')
df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')
df_panel = df_panel.sort_values(['firm', 'Date'], ascending=[True, True]).reset_index(drop=True)
if not df_panel.empty:
    df_panel['period_year'] = df_panel['Date'].dt.year.astype(float)
else:
    df_panel['period_year'] = pd.Series(dtype='float64')
df_panel = df_panel[df_panel['period_year'] <= ANALYSIS_END_YEAR].copy()

# Attach industry label per firm (normalized IDs)
if 'industry' not in df_panel.columns:
    if 'df_industry' in globals():
        df_panel = attach_industry_labels(df_panel, df_industry)
    else:
        print("Warning: df_industry not available; cannot attach industry labels.")

### --- Filtering for outliners ---

# ============================================================
# Distribution of observations per firm
# ============================================================



# ============================================================
# Revenue_Growth observation counts per firm (non-NaN only)
# ============================================================
print(f"\n{'='*60}")
print("REVENUE_GROWTH OBSERVATION COUNTS PER FIRM")
print(f"{'='*60}")

if 'Revenue_Growth' in df_panel.columns:
    revenue_growth_counts = df_panel.groupby('firm')['Revenue_Growth'].count()





def plot_metric_histograms(df_panel: pd.DataFrame, bins: int = 50, show: bool = True):
    """Plot histograms for the core metrics; returns {metric: figure}."""
    metrics = {
        "Revenue_Growth": "Revenue Growth",
        "Acq_Rate": "Acquisition Rate",
        "RRR": "Revenue Retention Rate",
        "SoNR": "Share of New Revenue",
        "NRLC": "New Revenue Level Contribution",
    }
    figs = {}
    for m, label in metrics.items():
        if m not in df_panel.columns:
            continue
        x = pd.to_numeric(df_panel[m], errors="coerce").dropna()
        if x.empty:
            continue

        mean_val = x.mean()
        std_val = x.std(ddof=1)
        if 'firm' in df_panel.columns:
            n_firms = int(df_panel.loc[x.index, 'firm'].nunique())
        else:
            n_firms = int(x.shape[0])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(x, bins=bins, alpha=0.7, edgecolor="black")
        ax.axvline(
            mean_val,
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f} | SD: {std_val:.4f} | Firms: 591",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of Firm-Year {label}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        figs[m] = fig
        if show:
            plt.show()
    return figs

# Example usage (run in your Interactive window after df_panel exists):
# %matplotlib inline
figs = plot_metric_histograms(df_panel, bins=50, show=True)


# ============================================================
# 2) Trim metrics to NaN based on quantiles
# ============================================================
def trim_metrics_to_nan_from_original(
    panel_df: pd.DataFrame,
    metrics: list[str],
    q_low: float = 0.01,
    q_high: float = 0.98,
):
    """
    Trim extreme values in specified metrics to NaN based on quantile thresholds.
    Modifies panel_df in-place.
    """
    for metric in metrics:
        if metric not in panel_df.columns:
            continue
        
        x = pd.to_numeric(panel_df[metric], errors='coerce')
        lower = x.quantile(q_low)
        upper = x.quantile(q_high)
        
        panel_df.loc[x < lower, metric] = np.nan
        panel_df.loc[x > upper, metric] = np.nan


# ============================================================
# 3) Convenience wrapper: trim + plot (rerun with new quantiles)
# ============================================================
def trim_and_plot(
    panel_df: pd.DataFrame,
    q_low: float = 0.01,
    q_high: float = 0.98,
    bins: int = 50,
    show: bool = True,
):
    # Trim in-place so downstream tables/regressions use the trimmed dataset
    metrics = ['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR', 'NRLC']
    trim_metrics_to_nan_from_original(panel_df, metrics=metrics, q_low=q_low, q_high=q_high)
    plot_metric_histograms(panel_df, bins=bins, show=show)

# Trim and plot distributions (quantile-based) and keep trimmed data for all downstream analysis
print(f"\n{'='*60}")
print("TRIMMED METRIC DISTRIBUTIONS (QUANTILE TRIM)")
print(f"{'='*60}\n")
trim_and_plot(df_panel, q_low=0.01, q_high=0.98, bins=50, show=True)


# ============================================================
# DATA QUALITY CHECK - Diagnostic before evolution table
# ============================================================
print(f"\n{'='*60}")
print("DATA QUALITY CHECK - Non-NaN counts by metric")
print(f"{'='*60}")
for metric in ['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR', 'NRLC']:
    if metric in df_panel.columns:
        non_nan_count = df_panel[metric].notna().sum()
        nan_count = df_panel[metric].isna().sum()
        total_count = len(df_panel)
        pct_valid = (non_nan_count / total_count) * 100 if total_count > 0 else 0
        print(f"  {metric:20s}: {non_nan_count:6d} non-NaN ({pct_valid:5.1f}%) | {nan_count:6d} NaN")

print(f"\nTotal rows in df_panel: {len(df_panel)}")
print(f"Total unique firms: {df_panel['firm'].nunique()}")
print(f"{'='*60}\n")


# ============================================================
# Q_t pattern by firm: always below/above 1 vs both sides
# ============================================================
print(f"\n{'='*60}")
print("Q_t PATTERN BY FIRM RELATIVE TO 1")
print(f"{'='*60}")

if {'firm', 'Q_t'}.issubset(df_panel.columns):
    q_df = df_panel[['firm', 'Q_t']].copy()
    q_df['Q_t'] = pd.to_numeric(q_df['Q_t'], errors='coerce')
    q_df = q_df.dropna(subset=['Q_t'])

    if q_df.empty:
        print("No non-missing Q_t observations found.")
    else:
        by_firm = q_df.groupby('firm')['Q_t']

        always_below = by_firm.apply(lambda s: (s < 1).all())
        always_above = by_firm.apply(lambda s: (s > 1).all())
        below_and_above = by_firm.apply(lambda s: (s < 1).any() and (s > 1).any())

        total_firms_q = q_df['firm'].nunique()
        n_always_below = int(always_below.sum())
        n_always_above = int(always_above.sum())
        n_below_and_above = int(below_and_above.sum())

        print(f"Firms with non-missing Q_t: {total_firms_q}")
        print(f"  Always Q_t < 1: {n_always_below} ({n_always_below/total_firms_q:.1%})")
        print(f"  Always Q_t > 1: {n_always_above} ({n_always_above/total_firms_q:.1%})")
        print(f"  Q_t both < 1 and > 1 over time: {n_below_and_above} ({n_below_and_above/total_firms_q:.1%})")
else:
    print("Cannot compute Q_t pattern: required columns (firm, Q_t) not found.")

print(f"{'='*60}\n")






# ============================================================
# Evolution Table (metrics over time periods)
# ============================================================
def create_metrics_evolution_table(panel_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Create a table showing Mean, SD, 10%, 25%, Median, 75%, 90% percentiles,
    and Obs count for each metric across different time periods (years).
    Prints metric name before each table (metric as header, not column).
    """
    if 'period_year' not in panel_df.columns:
        print("Warning: 'period_year' column not found. Cannot create evolution table.")
        return pd.DataFrame()
    
    # Get unique years and sort
    years = sorted(panel_df['period_year'].dropna().unique())
    
    for metric in metrics:
        if metric not in panel_df.columns:
            continue
        
        # Print metric name as header
        print(f"\n{metric}")
        
        # Build table rows for this metric (no Metric column)
        table_rows = []
        
        # Add rows for each year
        for year in years:
            year_data = pd.to_numeric(
                panel_df[panel_df['period_year'] == year][metric],
                errors='coerce'
            ).dropna()
            
            if len(year_data) == 0:
                continue
            
            table_rows.append({
                'Year': f'{int(year)}',
                'Mean': f"{year_data.mean():.4f}",
                'SD': f"{year_data.std():.4f}",
                '10%': f"{year_data.quantile(0.10):.4f}",
                '25%': f"{year_data.quantile(0.25):.4f}",
                'Median': f"{year_data.median():.4f}",
                '75%': f"{year_data.quantile(0.75):.4f}",
                '90%': f"{year_data.quantile(0.90):.4f}",
                'Obs': f"{len(year_data)}"
            })
        
        metric_table = pd.DataFrame(table_rows)
        
        # Print as TSV (tab-delimited, easy copy-paste to Excel/PowerPoint)
        print(metric_table.to_csv(sep='\t', index=False))
    
    return pd.DataFrame()


# Generate and display the evolution table as tab-separated format
if 'period_year' in df_panel.columns and len(df_panel['period_year'].dropna().unique()) > 0:
    evolution_table = create_metrics_evolution_table(
        df_panel, 
    metrics=['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR', 'NRLC']
    )
else:
    print("Cannot create evolution table: 'period_year' not available or no time variation.")

# --------------------------------------------------------------------
# Within-firm variance for core metrics + summary stats across firms
# --------------------------------------------------------------------
metrics_for_variance = [m for m in ['Revenue_Growth', 'Acq_Rate',  'SoNR', 'NRLC', 'RRR'] if m in df_panel.columns]
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
            'Metric': metric,
            'Mean': f"{series.mean():.4f}",
            'SD': f"{series.std(ddof=1):.4f}",
            '10%': f"{series.quantile(0.10):.4f}",
            '25%': f"{series.quantile(0.25):.4f}",
            'Median': f"{series.median():.4f}",
            '75%': f"{series.quantile(0.75):.4f}",
            '90%': f"{series.quantile(0.90):.4f}",
        })

    if summary_rows:
        variance_summary = pd.DataFrame(summary_rows)
        print('\nWithin-firm variance summary (variance of metric by firm):')
        print(variance_summary.to_string(index=False))

 
else:
    df_firm_variance = pd.DataFrame()
    print('No metrics available to compute within-firm variance.')



# --------------------------------------------------------------------
# AR vs. RRR correlation 
# --------------------------------------------------------------------

# ============================================================
# AR (Acquisition Rate) vs. RRR (Revenue Retention Rate) Correlation
# Calculate correlation by firm across the sample
# ============================================================

if {'Acq_Rate', 'RRR', 'firm'}.issubset(df_panel.columns):
    corr_df = df_panel[['firm', 'Acq_Rate', 'RRR']].copy()
    corr_df[['Acq_Rate', 'RRR']] = corr_df[['Acq_Rate', 'RRR']].apply(pd.to_numeric, errors='coerce')
    corr_df = corr_df.dropna(subset=['Acq_Rate', 'RRR'])

    if corr_df.empty:
        print('AR/RRR correlation analysis skipped: no overlapping observations after dropping NaNs.')
    else:
        # Overall correlation across all firm-period observations
        overall_corr = corr_df['Acq_Rate'].corr(corr_df['RRR'])
        n_obs = len(corr_df)
        n_firms = corr_df['firm'].nunique()
        
        print(f"\n{'='*60}")
        print(f"AR–RRR Correlation Analysis")
        print(f"{'='*60}")
        print(f"Overall correlation (pooled across {n_obs} observations, {n_firms} firms): {overall_corr:.4f}")
        
        # Firm-level correlations
        firm_corrs = (
            corr_df.groupby('firm')
            .apply(lambda g: g['Acq_Rate'].corr(g['RRR']) if len(g) >= 2 else np.nan)
            .dropna()
        )
        
        if not firm_corrs.empty:
            print(f"\nFirm-level correlation summary (n={len(firm_corrs)} firms with ≥2 observations):")
            print(f"  Mean: {firm_corrs.mean():.4f}")
            print(f"  Std:  {firm_corrs.std():.4f}")
            print(f"  10%:  {firm_corrs.quantile(0.1):.4f}")
            print(f"  25%:  {firm_corrs.quantile(0.25):.4f}")
            print(f"  Median: {firm_corrs.median():.4f}")
            print(f"  75%:  {firm_corrs.quantile(0.75):.4f}")
            print(f"  90%:  {firm_corrs.quantile(0.9):.4f}")

            # Histogram of firm-level correlations
            plt.figure(figsize=(10, 6))
            plt.hist(firm_corrs, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
            plt.axvline(firm_corrs.mean(), color='blue', linestyle='--', linewidth=2,
                        label=f"Mean: {firm_corrs.mean():.4f} | Firms: 591")
            plt.xlabel('Within-Firm Correlation (Acq_Rate vs RRR)')
            plt.ylabel('Number of Firms')
            plt.title('Within-Firm Correlation Between AR & RRR')
            plt.legend(loc='best', fontsize=9)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close()
        
        print(f"{'='*60}\n")
else:
    print("Cannot calculate AR/RRR correlation: required columns (firm, Acq_Rate, RRR) not all present.")


# --------------------------------------------------------------------
# Industry-time adjusted AR vs. RRR correlation
# --------------------------------------------------------------------
if {'firm', 'Date', 'Acq_Rate', 'RRR'}.issubset(df_panel.columns):
    adj_df = df_panel[['firm', 'Date', 'Acq_Rate', 'RRR']].copy()
    adj_df['Date'] = pd.to_datetime(adj_df['Date'], errors='coerce')
    adj_df['Acq_Rate'] = pd.to_numeric(adj_df['Acq_Rate'], errors='coerce')
    adj_df['RRR'] = pd.to_numeric(adj_df['RRR'], errors='coerce')

    if 'industry' not in adj_df.columns:
        if 'df_industry' in globals():
            adj_df = attach_industry_labels(adj_df, df_industry)

    if 'industry' not in adj_df.columns:
        print("Cannot calculate industry-adjusted AR/RRR correlation: 'industry' not available.")
    else:
        adj_df['AR_industry_median'] = adj_df.groupby(['industry', 'Date'])['Acq_Rate'].transform('median')
        adj_df['RRR_industry_median'] = adj_df.groupby(['industry', 'Date'])['RRR'].transform('median')
        adj_df['AR_adj'] = adj_df['AR_industry_median'] - adj_df['Acq_Rate']
        adj_df['RRR_adj'] = adj_df['RRR_industry_median'] - adj_df['RRR']
        adj_df = adj_df.dropna(subset=['AR_adj', 'RRR_adj'])

        if adj_df.empty:
            print('Industry-adjusted AR/RRR correlation skipped: no overlapping observations after adjustment.')
        else:
            overall_corr_adj = adj_df['AR_adj'].corr(adj_df['RRR_adj'])
            n_obs_adj = len(adj_df)
            n_firms_adj = adj_df['firm'].nunique()

            print(f"\n{'='*60}")
            print("Industry-Adjusted AR–RRR Correlation Analysis")
            print(f"{'='*60}")
            print(
                f"Overall correlation (pooled across {n_obs_adj} observations, {n_firms_adj} firms): "
                f"{overall_corr_adj:.4f}"
            )

            firm_corrs_adj = (
                adj_df.groupby('firm')
                .apply(lambda g: g['AR_adj'].corr(g['RRR_adj']) if len(g) >= 2 else np.nan)
                .dropna()
            )

            if not firm_corrs_adj.empty:
                print(f"\nFirm-level adjusted correlation summary (n={len(firm_corrs_adj)} firms with ≥2 observations):")
                print(f"  Mean: {firm_corrs_adj.mean():.4f}")
                print(f"  Std:  {firm_corrs_adj.std():.4f}")
                print(f"  10%:  {firm_corrs_adj.quantile(0.1):.4f}")
                print(f"  25%:  {firm_corrs_adj.quantile(0.25):.4f}")
                print(f"  Median: {firm_corrs_adj.median():.4f}")
                print(f"  75%:  {firm_corrs_adj.quantile(0.75):.4f}")
                print(f"  90%:  {firm_corrs_adj.quantile(0.9):.4f}")

                plt.figure(figsize=(10, 6))
                plt.hist(firm_corrs_adj, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
                plt.axvline(
                    firm_corrs_adj.mean(),
                    color='blue',
                    linestyle='--',
                    linewidth=2,
                    label=f"Mean: {firm_corrs_adj.mean():.4f} | Firms: 591"
                )
                plt.xlabel('Within-Firm Correlation (Industry-adjusted AR vs Industry-adjusted RRR)')
                plt.ylabel('Number of Firms')
                plt.title('Within-Firm Correlation: Industry-Adjusted AR & RRR')
                plt.legend(loc='best', fontsize=9)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()
                plt.close()

            print(f"{'='*60}\n")
else:
    print("Cannot calculate industry-adjusted AR/RRR correlation: required columns missing.")


# --------------------------------------------------------------------
# NRLC vs. RRR correlation
# --------------------------------------------------------------------

if {'NRLC', 'RRR', 'firm'}.issubset(df_panel.columns):
    nrlc_rrr_df = df_panel[['firm', 'NRLC', 'RRR']].copy()
    nrlc_rrr_df[['NRLC', 'RRR']] = nrlc_rrr_df[['NRLC', 'RRR']].apply(pd.to_numeric, errors='coerce')
    nrlc_rrr_df = nrlc_rrr_df.dropna(subset=['NRLC', 'RRR'])

    if nrlc_rrr_df.empty:
        print('NRLC/RRR correlation analysis skipped: no overlapping observations after dropping NaNs.')
    else:
        overall_corr = nrlc_rrr_df['NRLC'].corr(nrlc_rrr_df['RRR'])
        n_obs = len(nrlc_rrr_df)
        n_firms = nrlc_rrr_df['firm'].nunique()

        print(f"\n{'='*60}")
        print("NRLC–RRR Correlation Analysis")
        print(f"{'='*60}")
        print(f"Overall correlation (pooled across {n_obs} observations, {n_firms} firms): {overall_corr:.4f}")

        firm_corrs = (
            nrlc_rrr_df.groupby('firm')
            .apply(lambda g: g['NRLC'].corr(g['RRR']) if len(g) >= 2 else np.nan)
            .dropna()
        )

        if not firm_corrs.empty:
            print(f"\nFirm-level correlation summary (n={len(firm_corrs)} firms with ≥2 observations):")
            print(f"  Mean: {firm_corrs.mean():.4f}")
            print(f"  Std:  {firm_corrs.std():.4f}")
            print(f"  10%:  {firm_corrs.quantile(0.1):.4f}")
            print(f"  25%:  {firm_corrs.quantile(0.25):.4f}")
            print(f"  Median: {firm_corrs.median():.4f}")
            print(f"  75%:  {firm_corrs.quantile(0.75):.4f}")
            print(f"  90%:  {firm_corrs.quantile(0.9):.4f}")

            # Histogram of firm-level correlations
            plt.figure(figsize=(10, 6))
            plt.hist(firm_corrs, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
            plt.axvline(firm_corrs.mean(), color='blue', linestyle='--', linewidth=2,
                        label=f"Mean: {firm_corrs.mean():.4f} | Firms: 591")
            plt.xlabel('Within-Firm Correlation (NRLC vs RRR)')
            plt.ylabel('Number of Firms')
            plt.title('Within-Firm Correlation Between NRLC & RRR')
            plt.legend(loc='best', fontsize=9)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close()

        print(f"{'='*60}\n")
else:
    print("Cannot calculate NRLC/RRR correlation: required columns (firm, NRLC, RRR) not all present.")


# --------------------------------------------------------------------
# Log(size) vs. SoNR correlation
# --------------------------------------------------------------------
if {'Total_Revenue', 'SoNR'}.issubset(df_panel.columns):
    size_corr_df = df_panel[['firm', 'Total_Revenue', 'SoNR']].copy()
    size_corr_df['Total_Revenue'] = pd.to_numeric(size_corr_df['Total_Revenue'], errors='coerce')
    size_corr_df['SoNR'] = pd.to_numeric(size_corr_df['SoNR'], errors='coerce')
    size_corr_df['log_size'] = np.log1p(size_corr_df['Total_Revenue'])
    size_corr_df = size_corr_df.dropna(subset=['log_size', 'SoNR'])

    if size_corr_df.empty:
        print('Log(size)/SoNR correlation skipped: no overlapping observations after dropping NaNs.')
    else:
        overall_corr = size_corr_df['log_size'].corr(size_corr_df['SoNR'])
        n_obs = len(size_corr_df)
        n_firms = size_corr_df['firm'].nunique()

        print(f"\n{'='*60}")
        print("Log(Size) – SoNR Correlation Analysis")
        print(f"{'='*60}")
        print(f"Overall correlation (pooled across {n_obs} observations, {n_firms} firms): {overall_corr:.4f}")

        firm_corrs = (
            size_corr_df.groupby('firm')
            .apply(lambda g: g['log_size'].corr(g['SoNR']) if len(g) >= 2 else np.nan)
            .dropna()
        )

        if not firm_corrs.empty:
            print(f"\nFirm-level correlation summary (n={len(firm_corrs)} firms with ≥2 observations):")
            print(f"  Mean: {firm_corrs.mean():.4f}")
            print(f"  Std:  {firm_corrs.std():.4f}")
            print(f"  10%:  {firm_corrs.quantile(0.1):.4f}")
            print(f"  25%:  {firm_corrs.quantile(0.25):.4f}")
            print(f"  Median: {firm_corrs.median():.4f}")
            print(f"  75%:  {firm_corrs.quantile(0.75):.4f}")
            print(f"  90%:  {firm_corrs.quantile(0.9):.4f}")

            # Histogram of firm-level correlations
            plt.figure(figsize=(10, 6))
            plt.hist(firm_corrs, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
            plt.axvline(firm_corrs.mean(), color='blue', linestyle='--', linewidth=2,
                        label=f"Mean: {firm_corrs.mean():.4f} | Firms: 591")
            plt.xlabel('Within-Firm Correlation (log size vs SoNR)')
            plt.ylabel('Number of Firms')
            plt.title('Within-Firm Correlation Between Log(Size) & SoNR')
            plt.legend(loc='best', fontsize=9)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close()

        print(f"{'='*60}\n")
else:
    print("Cannot calculate log(size)/SoNR correlation: required columns (Total_Revenue, SoNR) not all present.")


# --------------------------------------------------------------------
# Firm-level average SoNR histogram
# --------------------------------------------------------------------
if {'firm', 'SoNR'}.issubset(df_panel.columns):
    firm_sonr = (
        df_panel[['firm', 'SoNR']]
        .assign(SoNR=lambda d: pd.to_numeric(d['SoNR'], errors='coerce'))
        .dropna(subset=['SoNR'])
        .groupby('firm', as_index=False)['SoNR']
        .mean()
        .rename(columns={'SoNR': 'SoNR_mean'})
    )

    if firm_sonr.empty:
        print('Firm-level SoNR histogram skipped: no non-missing SoNR values.')
    else:
        n_firms_total = len(firm_sonr)
        n_below_half = (firm_sonr['SoNR_mean'] < 0.5).sum()
        pct_below_half = (n_below_half / n_firms_total) * 100 if n_firms_total > 0 else 0

        print(f"\nFirm-level average SoNR: {n_firms_total} firms")
        print(f"Firms with average SoNR < 0.5: {n_below_half} ({pct_below_half:.1f}%)")

        plt.figure(figsize=(10, 6))
        plt.hist(firm_sonr['SoNR_mean'], bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
        plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold: 0.5 | Firms: 591')
        plt.xlabel('Average SoNR by Firm')
        plt.ylabel('Number of Firms')
        plt.title('Distribution of Firm-Level Average SoNR')
        plt.legend(loc='best', fontsize=9)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()
else:
    print("Firm-level SoNR histogram skipped: required columns (firm, SoNR) not found.")






# --------------------------------------------------------------------
# Firm-level median histograms for Acq_Rate, RRR, SoNR
# --------------------------------------------------------------------
print(f"\n{'='*60}")
print("FIRM-LEVEL MEDIAN HISTOGRAMS")
print(f"{'='*60}")

metric_labels_firm_median = {
    'Revenue_Growth': 'Revenue Growth',
    'Acq_Rate': 'Acquisition Rate',
    'RRR': 'Revenue Retention Rate',
    'SoNR': 'Share of New Revenue',
}

for metric, label in metric_labels_firm_median.items():
    if not {'firm', metric}.issubset(df_panel.columns):
        print(f"{metric} skipped: required columns not found.")
        continue

    firm_metric = (
        df_panel[['firm', metric]]
        .assign(**{metric: lambda d: pd.to_numeric(d[metric], errors='coerce')})
        .dropna(subset=[metric])
        .groupby('firm', as_index=False)[metric]
        .median()
        .rename(columns={metric: f'{metric}_firm_median'})
    )

    if firm_metric.empty:
        print(f"{metric} skipped: no non-missing values.")
        continue

    x = firm_metric[f'{metric}_firm_median']
    n_firms = len(firm_metric)
    mean_val = x.mean()
    sd_val = x.std(ddof=1)

    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    plt.axvline(
        mean_val,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f"Mean: {mean_val:.4f} | SD: {sd_val:.4f} | Firms: 591",
    )
    plt.xlabel(f'Firm-level Median {label}')
    plt.ylabel('Number of Firms')
    plt.title(f'Distribution of Firm-level Median {label}')
    plt.legend(loc='best', fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

print(f"{'='*60}\n")


# --------------------------------------------------------------------
# Industry-level metric medians: top 5 and bottom 5 industries
# --------------------------------------------------------------------
print(f"\n{'='*60}")
print("INDUSTRY-LEVEL MEDIAN METRICS (TOP/BOTTOM 5)")
print(f"{'='*60}")

if 'industry' not in df_panel.columns and 'df_industry' in globals():
    df_panel = attach_industry_labels(df_panel, df_industry)

industry_metrics_to_show = [m for m in ['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR'] if m in df_panel.columns]

if {'industry'}.issubset(df_panel.columns) and industry_metrics_to_show:
    for metric in industry_metrics_to_show:
        metric_df = df_panel[['industry', metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors='coerce')
        metric_df = metric_df.dropna(subset=['industry', metric])

        if metric_df.empty:
            print(f"{metric}: no valid industry observations.")
            continue

        industry_median = (
            metric_df.groupby('industry', as_index=False)[metric]
            .median()
            .rename(columns={metric: 'industry_median'})
            .sort_values('industry_median', ascending=False)
        )

        top5 = industry_median.head(5).copy()
        bottom5 = industry_median.tail(5).sort_values('industry_median', ascending=True).copy()

        print(f"\n{metric} - Top 5 industries by median")
        print(top5.to_string(index=False))
        print(f"\n{metric} - Bottom 5 industries by median")
        print(bottom5.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False)

        axes[0].barh(top5['industry'], top5['industry_median'], color='#1f77b4', alpha=0.8)
        axes[0].invert_yaxis()
        axes[0].set_title(f'Top 5 Industries - Median {metric}')
        axes[0].set_xlabel('Median value')
        axes[0].grid(axis='x', alpha=0.3)

        axes[1].barh(bottom5['industry'], bottom5['industry_median'], color='#d62728', alpha=0.8)
        axes[1].invert_yaxis()
        axes[1].set_title(f'Bottom 5 Industries - Median {metric}')
        axes[1].set_xlabel('Median value')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()
else:
    print("Industry-level median graphics skipped: missing 'industry' or metric columns.")

print(f"{'='*60}\n")


# ============================================================
# Industry-median regressions (firm-level) and significance histograms
# ============================================================
def prepare_panel_with_industry_medians(
    panel_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Assign industry to each firm and attach per-(industry, Date) medians for selected metrics."""
    out = attach_industry_labels(panel_df, industry_df)
    if 'industry' not in out.columns:
        return out

    out = out.copy()
    for metric in metrics:
        if metric not in out.columns:
            continue
        out[metric] = pd.to_numeric(out[metric], errors='coerce')
        med_col = f'industry_median_{metric}'
        out[med_col] = out.groupby(['industry', 'Date'])[metric].transform('median')

    return out


def run_industry_median_regressions(panel_df: pd.DataFrame, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """For each metric, run firm-level OLS: (industry_median_t - metric_it) = alpha_i."""

    required_cols = {'firm', 'industry', 'Date'}
    if not required_cols.issubset(panel_df.columns):
        print("Industry-median regressions skipped: missing firm/industry/Date columns.")
        return {}

    results: dict[str, pd.DataFrame] = {}

    for metric in metrics:
        if metric not in panel_df.columns:
            print(f"{metric} not in panel; skipping.")
            continue

        med_col = f'industry_median_{metric}'
        if med_col not in panel_df.columns:
            print(f"{med_col} not in panel; skipping {metric}.")
            continue

        sub = panel_df[['firm', 'industry', 'Date', metric, med_col]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors='coerce')
        sub[med_col] = pd.to_numeric(sub[med_col], errors='coerce')
        sub = sub.dropna(subset=['firm', 'industry', 'Date', metric, med_col])
        if sub.empty:
            metric_non_nan = pd.to_numeric(panel_df[metric], errors='coerce').notna().sum()
            industry_non_nan = panel_df['industry'].notna().sum() if 'industry' in panel_df.columns else 0
            median_non_nan = pd.to_numeric(panel_df[med_col], errors='coerce').notna().sum()
            print(
                f"No data for {metric}; skipping. "
                f"Non-NaN {metric}: {metric_non_nan}, "
                f"Non-NaN industry: {industry_non_nan}, "
                f"Non-NaN {med_col}: {median_non_nan}"
            )
            continue

        rows = []
        for firm, g in sub.groupby('firm'):
            if len(g) < 3:
                continue

            y_gap = g[med_col] - g[metric]
            X = np.ones((len(g), 1))  # constant-only regression
            try:
                res = sm.OLS(y_gap, X, missing='drop').fit()
            except Exception:
                continue

            rows.append({
                'firm': firm,
                'n_obs': int(res.nobs),
                'alpha': float(res.params.iloc[0]) if hasattr(res.params, 'iloc') else float(res.params[0]),
                'alpha_pval': float(res.pvalues.iloc[0]) if hasattr(res.pvalues, 'iloc') else float(res.pvalues[0]),
            })

        if not rows:
            print(f"No estimable firm regressions for {metric} (need >=3 obs per firm).")
            continue

        df_res = pd.DataFrame(rows)
        results[metric] = df_res

        sig_alpha = df_res[df_res['alpha_pval'] < 0.10]
        sig_alpha_pos = sig_alpha[sig_alpha['alpha'] > 0]

        print(f"\n{'='*60}")
        print(f"Industry-median regression (firm-level): {metric}")
        print(f"{'='*60}")
        print(f"Firms estimated: {len(df_res)} | Mean obs per firm: {df_res['n_obs'].mean():.1f}")
        print(f"Significant alpha (p<0.10): {len(sig_alpha)} firms")
        print(f"Significant positive alpha (p<0.10 & alpha>0): {len(sig_alpha_pos)} firms")

        if not sig_alpha.empty:
            alpha_std = sig_alpha['alpha'].std(ddof=1)
            plt.figure(figsize=(10, 6))
            plt.hist(sig_alpha['alpha'], bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
            plt.axvline(sig_alpha['alpha'].mean(), color='blue', linestyle='--', linewidth=2,
                        label=f"Mean alpha: {sig_alpha['alpha'].mean():.4f} | SD: {alpha_std:.4f} | Firms: 591")
            plt.xlabel('Alpha (industry median - firm metric)')
            plt.ylabel('Number of firms')
            plt.title(f"Significant alpha estimates (p<0.10) for {metric}")
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        if not sig_alpha.empty:
            alpha_stats = sig_alpha['alpha'].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
            print("Significant alpha (p<0.10) summary:")
            print(
                "  "
                f"Mean: {alpha_stats['mean']:.4f}, "
                f"SD: {alpha_stats['std']:.4f}, "
                f"10%: {alpha_stats['10%']:.4f}, "
                f"25%: {alpha_stats['25%']:.4f}, "
                f"Median: {alpha_stats['50%']:.4f}, "
                f"75%: {alpha_stats['75%']:.4f}, "
                f"90%: {alpha_stats['90%']:.4f}"
            )

    return results


industry_metrics = [m for m in ['Acq_Rate', 'RRR'] if m in df_panel.columns]

if 'df_industry' in globals():
    df_panel = prepare_panel_with_industry_medians(df_panel, df_industry, industry_metrics)
else:
    print("Industry median prep skipped: df_industry not available in memory.")

industry_median_reg_results = run_industry_median_regressions(df_panel, metrics=industry_metrics)

# 3x3 summary table of alpha status (AR rows x RRR columns)
if {'Acq_Rate', 'RRR'}.issubset(industry_median_reg_results.keys()):
    ar_all = industry_median_reg_results['Acq_Rate'][['firm', 'alpha', 'alpha_pval']].copy()
    rrr_all = industry_median_reg_results['RRR'][['firm', 'alpha', 'alpha_pval']].copy()
    both_all = ar_all.merge(rrr_all, on='firm', how='inner', suffixes=('_ar', '_rrr'))

    print(f"\n{'='*60}")
    print("3x3 TABLE: ALPHA STATUS (INCLUDING NON-SIGNIFICANT)")
    print(f"{'='*60}")

    if both_all.empty:
        print("No common firms with estimable Acq_Rate and RRR regressions.")
    else:
        alpha_corr = both_all['alpha_ar'].corr(both_all['alpha_rrr'])
        print(f"Correlation between alpha estimates (AR vs RRR): {alpha_corr:.4f}")

        def _alpha_status(alpha: float, pval: float, metric_label: str) -> str:
            if pd.isna(alpha) or pd.isna(pval):
                return f"{metric_label} non-sig"
            if pval >= 0.10:
                return f"{metric_label} non-sig"
            return f"{metric_label} neg sig" if alpha < 0 else f"{metric_label} pos sig"

        both_all['ar_status'] = both_all.apply(
            lambda r: _alpha_status(r['alpha_ar'], r['alpha_pval_ar'], 'AR'),
            axis=1
        )
        both_all['rrr_status'] = both_all.apply(
            lambda r: _alpha_status(r['alpha_rrr'], r['alpha_pval_rrr'], 'RRR'),
            axis=1
        )

        table_counts = pd.crosstab(
            both_all['ar_status'],
            both_all['rrr_status']
        ).reindex(
            index=['AR neg sig', 'AR non-sig', 'AR pos sig'],
            columns=['RRR neg sig', 'RRR non-sig', 'RRR pos sig'],
            fill_value=0,
        )

        n_common = len(both_all)
        table_pct = (table_counts / n_common) * 100

        print(f"Common estimable firms (Acq_Rate & RRR): {n_common}")
        print("\nCount table:")
        print(table_counts.to_string())
        print("\nPercent table (% of common estimable firms):")
        print(table_pct.round(1).to_string())

    print(f"{'='*60}\n")



# ============================================================
# 2025 firm-level comparison: 1+RG_AR vs 1+RG_RRR
# ============================================================
print(f"\n{'='*60}")
print("FIRM-LEVEL 2025 COMPARISON: 1+RG_AR VS 1+RG_RRR")
print(f"{'='*60}")

required_cols_rg_cmp = {'firm', 'Date', 'period_year', 'SoNR', 'Acq_Rate', 'RRR'}
if required_cols_rg_cmp.issubset(df_panel.columns):
    cmp_df = df_panel[list(required_cols_rg_cmp)].copy()
    for c in ['SoNR', 'Acq_Rate', 'RRR', 'period_year']:
        cmp_df[c] = pd.to_numeric(cmp_df[c], errors='coerce')
    cmp_df['Date'] = pd.to_datetime(cmp_df['Date'], errors='coerce')

    # last observation in 2025 per firm
    last_2025 = (
        cmp_df[cmp_df['period_year'] == ANALYSIS_END_YEAR]
        .sort_values(['firm', 'Date'])
        .groupby('firm', as_index=False)
        .tail(1)
    )

    # firm-level standard deviations over the full sample
    firm_sds = (
        cmp_df.groupby('firm')[['Acq_Rate', 'RRR']]
        .std(ddof=1)
        .rename(columns={'Acq_Rate': 'sd_ar', 'RRR': 'sd_rrr'})
        .reset_index()
    )

    comp = last_2025.merge(firm_sds, on='firm', how='left')

    # User-requested formulas
    # 1+RG_AR  = SoNR_2025 * (AR_2025 + sd(AR)_firm) + RRR_2025
    # 1+RG_RRR = SoNR_2025 * (AR_2025 + RRR_2025 + sd(RRR)_firm)
    comp['one_plus_RG_AR'] = comp['SoNR'] * (comp['Acq_Rate'] + comp['sd_ar']) + comp['RRR']
    comp['one_plus_RG_RRR'] = comp['SoNR'] * (comp['Acq_Rate'] + comp['RRR'] + comp['sd_rrr'])

    valid_comp = comp.dropna(subset=['one_plus_RG_AR', 'one_plus_RG_RRR'])
    if valid_comp.empty:
        print("No valid firm-level observations for the 2025 comparison.")
    else:
        n_firms_valid = len(valid_comp)
        n_gt = int((valid_comp['one_plus_RG_AR'] > valid_comp['one_plus_RG_RRR']).sum())
        n_le = int((valid_comp['one_plus_RG_AR'] <= valid_comp['one_plus_RG_RRR']).sum())
        pct_gt = (n_gt / n_firms_valid) * 100

        print(f"Firms with valid 2025 comparison: {n_firms_valid}")
        print(f"1+RG_AR > 1+RG_RRR: {n_gt} firms ({pct_gt:.1f}%)")
        print(f"1+RG_AR <= 1+RG_RRR: {n_le} firms ({100 - pct_gt:.1f}%)")
else:
    missing = sorted(required_cols_rg_cmp.difference(df_panel.columns))
    print(f"Cannot compute 2025 comparison: missing columns {missing}")

print(f"{'='*60}\n")



# ============================================================
# Firm-Level Time-Trend Regressions (metric ~ calendar year)
# ============================================================
# Strategy: For each metric and firm:
#   1. Run regression: metric ~ const + period_year (calendar year)
#   2. Extract and test significance of all coefficients
#   3. Count firms by observation count and significance levels
#   4. Visualize significant constants and calendar-year betas
# ============================================================

metrics_trend = [
    ('Revenue_Growth', 'Revenue Growth'),
    ('Acq_Rate', 'Acquisition Rate'),
    ('RRR', 'Revenue Retention Rate'),
    ('SoNR', 'Share of New Revenue'),
]

if 'period_year' not in df_panel.columns:
    print("Cannot run time-trend regressions: 'period_year' column missing.")
else:
    try:
        from scipy import stats
    except Exception as e:
        print(f"Scipy not available; skipping time-trend regressions: {e}")
        stats = None
    
    if stats is not None:
        def _compute_comprehensive_trend_stats(metric: str, pretty_label: str) -> dict:
            """
            Compute time-trend regressions for each firm with constant + calendar year only.
            Returns dict with results, firm counts by obs, and significance counts.
            """
            if metric not in df_panel.columns:
                print(f"{pretty_label} column missing; skipping.")
                return None

            # Prepare data
            sub = df_panel[['firm', 'Date', 'period_year', metric, 'Total_Revenue']].dropna()
            if sub.empty:
                print(f"No data available for {pretty_label}; skipping.")
                return None

            sub = sub.sort_values(['firm', 'Date']).copy()

            # Count observations per firm
            firm_obs_counts = sub.groupby('firm').size()
            
            obs_dist = {}
            for n in range(2, 8):
                obs_dist[f'{n}_obs'] = (firm_obs_counts == n).sum()
            total_firms_with_data = len(firm_obs_counts)

            # Run regressions by firm
            rows = []
            for firm in sub['firm'].unique():
                firm_data = sub[sub['firm'] == firm].copy()
                
                if len(firm_data) < 2:
                    continue

                firm_data = firm_data.dropna(subset=[metric, 'period_year'])

                if len(firm_data) < 2:
                    continue

                try:
                    X = sm.add_constant(pd.DataFrame({
                        'trend': firm_data['period_year']
                    }))
                    y = firm_data[metric]
                    res = sm.OLS(y, X, missing='drop').fit()

                    rows.append({
                        'firm': firm,
                        'n_obs': len(firm_data),
                        'trend_beta': res.params.get('trend', np.nan),
                        'trend_pval': res.pvalues.get('trend', np.nan),
                        'const_beta': res.params.get('const', np.nan),
                        'const_pval': res.pvalues.get('const', np.nan)
                    })
                except Exception:
                    continue

            if not rows:
                print(f"Could not estimate regressions for {pretty_label}.")
                return None

            df_res = pd.DataFrame(rows)
            
            # Count significance
            tested = len(df_res)
            trend_signif = (df_res['trend_pval'] < 0.10).sum()
            const_signif = (df_res['const_pval'] < 0.10).sum()

            return {
                'results': df_res,
                'tested': tested,
                'total_firms': total_firms_with_data,
                'obs_dist': obs_dist,
                'trend_signif': trend_signif,
                'const_signif': const_signif
            }

        # Run analysis for each metric
        for metric, pretty_label in metrics_trend:
            result = _compute_comprehensive_trend_stats(metric, pretty_label)
            if result is None:
                continue

            df_res = result['results']
            tested = result['tested']
            total_firms = result['total_firms']
            obs_dist = result['obs_dist']
            trend_signif = result['trend_signif']
            const_signif = result['const_signif']

            # Print summary
            print(f"\n{'='*80}")
            print(f"FIRM-LEVEL TREND ANALYSIS: {pretty_label}")
            print(f"{'='*80}")
            print(f"Total firms with {metric} data: {total_firms}")
            print(f"Firms tested (≥2 observations): {tested}")
            print(f"\nDistribution of observations per firm:")
            for n in range(2, 8):
                key = f'{n}_obs'
                if key in obs_dist:
                    count = obs_dist[key]
                    pct = (count / total_firms * 100) if total_firms > 0 else 0
                    print(f"  {n} observations: {count:3d} firms ({pct:5.1f}%)")

            print(f"\nRegression results (Const + Period_Year):") 
            print(f"  Firms with significant TREND (p<0.10): {trend_signif:3d} ({trend_signif/tested:.1%})")
            print(f"  Firms with significant CONST (p<0.10): {const_signif:3d} ({const_signif/tested:.1%})")

            

            # Visualize trend coefficients with non-significant coefficients set to 0
            trend_beta_with_zeros = pd.Series(
                np.where(df_res['trend_pval'] < 0.10, df_res['trend_beta'], 0.0),
                index=df_res.index,
                dtype='float64',
            ).dropna()
            if not trend_beta_with_zeros.empty:
                trend_mean = trend_beta_with_zeros.mean()
                trend_std = trend_beta_with_zeros.std(ddof=1)
                n_trend_zero = int((trend_beta_with_zeros == 0).sum())

                plt.figure(figsize=(10, 6))
                plt.hist(trend_beta_with_zeros, bins=25, color='#1f77b4', alpha=0.7, edgecolor='black')
                plt.axvline(
                    trend_mean,
                    color='blue',
                    linestyle=':',
                    linewidth=2,
                    label=(
                        f'Mean: {trend_mean:.4f} | SD: {trend_std:.4f} '
                        f'| Firms: 591'
                    ),
                )

                plt.xlabel('Trend Coefficient (beta; non-significant set to 0)')
                plt.ylabel('Number of Firms')
                plt.title(f"Time-Trend Slopes for {pretty_label}")
                plt.legend(loc='best', fontsize=9)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()

            # Visualize constant coefficients with non-significant coefficients set to 0
            const_beta_with_zeros = pd.Series(
                np.where(df_res['const_pval'] < 0.10, df_res['const_beta'], 0.0),
                index=df_res.index,
                dtype='float64',
            ).dropna()
            if not const_beta_with_zeros.empty:
                const_mean = const_beta_with_zeros.mean()
                const_std = const_beta_with_zeros.std(ddof=1)
                n_const_zero = int((const_beta_with_zeros == 0).sum())

                plt.figure(figsize=(10, 6))
                plt.hist(const_beta_with_zeros, bins=25, color='#1f77b4', alpha=0.7, edgecolor='black')
                plt.axvline(
                    const_mean,
                    color='blue',
                    linestyle=':',
                    linewidth=2,
                    label=(
                        f"Mean: {const_mean:.4f} | SD: {const_std:.4f} "
                        f"| Firms: 591"
                    ),
                )

                plt.xlabel('Constant Coefficient (beta; non-significant set to 0)')
                plt.ylabel('Number of Firms')
                plt.title(f"Constant Coefficients for {pretty_label}")
                plt.legend(loc='best', fontsize=9)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()

            print(f"{'='*80}\n")

