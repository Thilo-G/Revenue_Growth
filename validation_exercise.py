"""
validation_exercise.py — Validation analyses for the Revenue Growth paper.

Three analyses:
  1. Out-of-sample revenue growth forecasting (in-sample OLS + expanding-window
     RMSE + Clark-West test for nested model comparison).
  2. Future revenue volatility (forecast-error regression, cross-sectional SD(RG)
     regression, quadrant comparison of absolute forecast errors).
  3. Q_t robustness excluding 2020-2021 (updated Panel C figures for the
     robustness table).

Outputs two LaTeX table files and prints all key numbers to console.
"""

# ==============================================================================
# Imports
# ==============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ==============================================================================
# Constants — paths and analysis settings
# ==============================================================================
DATA_FILE_REVENUE = (
    "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft"
    "\\Thilo(privat)\\Privat\\Research\\Revenue_Growth\\RG_Data"
    "\\2026-02-18-Firm-Year-Revenue-Python.xlsx"
)
DATA_FILE_INDUSTRY = (
    "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft"
    "\\Thilo(privat)\\Privat\\Research\\Revenue_Growth\\RG_Data"
    "\\Firms-Industry-2025-11-21-Python.xlsx"
)
LATEX_TABLE_DIR = (
    "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft"
    "\\Thilo(privat)\\Privat\\Research\\Revenue_Growth\\RG_LaTeX\\tables"
)

ANALYSIS_END_YEAR = 2025

# Trimming percentiles (1st / 98th — matches v2 pipeline)
TRIM_LOWER = 0.01
TRIM_UPPER = 0.98

# Expanding-window OOS: first training window covers these years, then extends
OOS_FIRST_TRAIN_END = 2019   # train on 2018–2019, predict 2020
COVID_YEARS = [2020, 2021]

os.makedirs(LATEX_TABLE_DIR, exist_ok=True)


# ==============================================================================
# Helper utilities
# ==============================================================================

def _norm_id(x: object) -> str:
    """Normalize a firm/industry ID string for robust matching."""
    s = str(x).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _base_name(col_name: str) -> str:
    """Strip metric suffix from a column name to recover the firm base ID."""
    return (
        str(col_name)
        .replace('#New_Customers', '')
        .replace('#New_Customer', '')
        .replace('#Returning_Customers', '')
        .replace('#Returning_Customer', '')
        .lstrip('#')
        .strip('. ')
    )


def _stars(pval: float) -> str:
    """Return significance star string."""
    if pval < 0.01:
        return '***'
    if pval < 0.05:
        return '**'
    if pval < 0.10:
        return '*'
    return ''


def _fmt_coef(coef: float, pval: float) -> str:
    """Format coefficient with significance stars."""
    return f"{coef:.3f}{_stars(pval)}"


def _fmt_tstat(tval: float) -> str:
    """Format t-statistic in parentheses."""
    return f"({tval:.2f})"


# ==============================================================================
# DATA LOADING — exact mirror of the new_analyses_v2.py pipeline
# ==============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df_revenue = pd.read_excel(DATA_FILE_REVENUE, header=None, engine='openpyxl')
df_industry = pd.read_excel(DATA_FILE_INDUSTRY, header=0, engine='openpyxl')

# Strip column whitespace in industry file
df_industry.columns = df_industry.columns.str.strip()

# Industry filtering: keep only sub-industries with >= 2 firms
df_industry['gics_sub_industry_name'] = (
    df_industry['gics_sub_industry_name']
    .astype(str).str.strip()
    .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
)
df_industry = df_industry.dropna(subset=['gics_sub_industry_name']).copy()
sub_counts = df_industry.groupby('gics_sub_industry_name')['ID'].nunique()
keep_subs = sub_counts[sub_counts >= 2].index
df_industry = df_industry[df_industry['gics_sub_industry_name'].isin(keep_subs)].copy()

valid_ids_norm = set(_norm_id(x) for x in df_industry["ID"].dropna())

# Select revenue columns that match valid industry IDs
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

if len(cols_to_keep) == 1:
    fallback_cols = [0]
    for col_idx in df_revenue.columns[1:]:
        vals = [df_revenue.iloc[r, col_idx] for r in range(max_header_row)]
        if any(pd.notna(v) and str(v).strip() != "" for v in vals):
            fallback_cols.append(col_idx)
    cols_to_keep = fallback_cols

df_revenue = df_revenue.iloc[:, cols_to_keep].copy()

# Build single-level column headers from the first two rows
header_rows_rev = df_revenue.iloc[:2].copy()
data_rows_rev = df_revenue.iloc[2:].copy()

combined_headers = header_rows_rev.apply(lambda x: x.astype(str).str.strip())
combined_headers = combined_headers.replace({'nan': pd.NA, 'NaT': pd.NA, 'None': pd.NA, '': pd.NA})
column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
data_rows_rev.columns = column_headers
data_rows_rev = data_rows_rev.rename(columns={data_rows_rev.columns[0]: 'Date'})
df_revenue = data_rows_rev.reset_index(drop=True)

df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True)
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'], errors='coerce')
df_revenue = df_revenue[df_revenue['Date'].notna()].copy()
df_revenue = df_revenue[df_revenue['Date'].dt.year <= ANALYSIS_END_YEAR].copy()
df_revenue = df_revenue.sort_values('Date').reset_index(drop=True)

# Identify New / Returning customer columns
new_customer_cols = [c for c in df_revenue.columns if '#New_Customers' in c or '#New_Customer' in c]
returning_customer_cols = [c for c in df_revenue.columns if '#Returning_Customers' in c or '#Returning_Customer' in c]

# Drop firms with any blank or 'missing value' cell
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

new_customer_cols = [c for c in df_revenue.columns if '#New_Customers' in c or '#New_Customer' in c]
returning_customer_cols = [c for c in df_revenue.columns if '#Returning_Customers' in c or '#Returning_Customer' in c]

# Compute derived metrics (RRR, RG, AR, Q_t, SoNR, NRLC)
new_map = {_base_name(col): col for col in new_customer_cols}
ret_map = {_base_name(col): col for col in returning_customer_cols}
original_cols = list(df_revenue.columns)
new_cols_data = {}
final_cols = []

for col in original_cols:
    final_cols.append(col)
    if col in returning_customer_cols:
        base = _base_name(col)
        new_col = new_map.get(base)
        if new_col is None:
            continue
        new_s = pd.to_numeric(df_revenue[new_col], errors='coerce')
        ret_s = pd.to_numeric(df_revenue[col], errors='coerce')
        total = new_s + ret_s
        total_nonzero = total.replace(0, np.nan)
        total_prev_nonzero = total.shift(1).replace(0, np.nan)

        total_revenue_col = new_col.replace('#New_Customers', '#Total_Revenue')
        rrr_col = col.replace('#Returning_Customers', '#RRR')
        revenue_growth_col = col.replace('#Returning_Customers', '#Revenue_Growth')
        acq_rate_col = new_col.replace('#New_Customers', '#Acq_Rate')
        q_t_col = col.replace('#Returning_Customers', '#Q_t')
        sonr_col = new_col.replace('#New_Customers', '#SoNR')
        nrlc_col = new_col.replace('#New_Customers', '#NRLC')

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
        final_cols.extend([total_revenue_col, rrr_col, revenue_growth_col,
                           acq_rate_col, q_t_col, sonr_col, nrlc_col])

if new_cols_data:
    df_new = pd.DataFrame(new_cols_data, index=df_revenue.index)
    df_revenue = pd.concat([df_revenue, df_new], axis=1)

final_cols = [c for c in final_cols if c in df_revenue.columns]
df_revenue = df_revenue.loc[:, final_cols]

# Reshape to long (panel) format
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'], errors='coerce')
df_revenue = df_revenue.sort_values('Date').reset_index(drop=True)

metrics_suffixes = ['#Revenue_Growth', '#RRR', '#Acq_Rate', '#Q_t', '#Total_Revenue', '#SoNR', '#NRLC']
present_metric_cols = [c for c in df_revenue.columns if any(c.endswith(s) for s in metrics_suffixes)]

df_wide = df_revenue.copy()
df_wide['Date'] = pd.to_datetime(df_wide['Date'], errors='coerce')
metric_cols = [c for c in df_wide.columns if c in present_metric_cols]
df_long = df_wide.melt(id_vars='Date', value_vars=metric_cols, var_name='col', value_name='value')
df_long['firm'] = df_long['col'].apply(lambda c: c.split('#')[0] if '#' in c else c)
df_long['metric'] = df_long['col'].apply(lambda c: c.split('#', 1)[1] if '#' in c else '')

df_panel = (
    df_long
    .pivot_table(index=['Date', 'firm'], columns='metric', values='value', aggfunc='first')
    .reset_index()
)
df_panel.columns = [c if isinstance(c, str) else c[1] for c in df_panel.columns]

for col in ['Revenue_Growth', 'RRR', 'Acq_Rate', 'Q_t', 'Total_Revenue', 'SoNR', 'NRLC']:
    if col in df_panel.columns:
        df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce')

df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')
df_panel = df_panel.sort_values(['firm', 'Date']).reset_index(drop=True)
df_panel['period_year'] = df_panel['Date'].dt.year.astype(float)
df_panel = df_panel[df_panel['period_year'] <= ANALYSIS_END_YEAR].copy()

# Attach industry labels
def attach_industry_labels(panel_df, industry_df):
    """Attach GICS sub-industry label to panel by robust firm-ID matching."""
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
    return out

df_panel = attach_industry_labels(df_panel, df_industry)

# Trim outliers at 1st / 98th percentiles (values set to NaN, not dropped)
for metric in ['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR', 'NRLC']:
    if metric not in df_panel.columns:
        continue
    x = pd.to_numeric(df_panel[metric], errors='coerce')
    lower = x.quantile(TRIM_LOWER)
    upper = x.quantile(TRIM_UPPER)
    df_panel.loc[x < lower, metric] = np.nan
    df_panel.loc[x > upper, metric] = np.nan

n_firms = df_panel['firm'].nunique()
years_available = sorted(df_panel['period_year'].dropna().unique().astype(int))
print(f"Panel loaded: {len(df_panel)} rows, {n_firms} firms")
print(f"Years: {years_available}")

# Compute firm-level quadrant classification (used in both Analysis 1 and 2)
# Quadrant based on whether firm-median AR and firm-median RRR are above/below
# the cross-firm medians — classification uses the full sample
firm_medians_full = (
    df_panel.groupby('firm')[['Acq_Rate', 'RRR', 'Revenue_Growth']]
    .median()
    .dropna()
)
ar_cross_median = firm_medians_full['Acq_Rate'].median()
rrr_cross_median = firm_medians_full['RRR'].median()

firm_medians_full['AR_high'] = firm_medians_full['Acq_Rate'] >= ar_cross_median
firm_medians_full['RRR_high'] = firm_medians_full['RRR'] >= rrr_cross_median
firm_medians_full['quadrant'] = np.where(
    firm_medians_full['AR_high'] & firm_medians_full['RRR_high'], 'HH',
    np.where(
        firm_medians_full['AR_high'] & ~firm_medians_full['RRR_high'], 'HL',
        np.where(
            ~firm_medians_full['AR_high'] & firm_medians_full['RRR_high'], 'LH', 'LL'
        )
    )
)
df_panel = df_panel.merge(
    firm_medians_full[['quadrant']].reset_index(),
    on='firm', how='left'
)


# ==============================================================================
# ANALYSIS 1: Out-of-Sample Revenue Growth Forecasting
# ==============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: OUT-OF-SAMPLE REVENUE GROWTH FORECASTING")
print("=" * 70)

# Build a one-period-ahead panel: align RG_{i,t+1} with predictors at time t.
# Sort by firm and year, then shift RG forward within each firm.
df_panel = df_panel.sort_values(['firm', 'period_year']).reset_index(drop=True)
df_panel['RG_next'] = (
    df_panel.groupby('firm')['Revenue_Growth'].shift(-1)
)
df_panel['SoNR_lag'] = (
    df_panel.groupby('firm')['SoNR'].shift(1)
)

# Working sample: observations where all predictors and the outcome are observed
# Include period_year as the "training year" (i.e., year t; outcome is year t+1)
reg_cols = ['firm', 'period_year', 'Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR', 'RG_next']
df_reg = df_panel[reg_cols].dropna().copy()
df_reg['period_year'] = df_reg['period_year'].astype(int)
print(f"  Regression sample: {len(df_reg)} obs, {df_reg['firm'].nunique()} firms")
print(f"  Predictor years (t): {sorted(df_reg['period_year'].unique())}")


def _cluster_se_ols(endog, exog, cluster_var):
    """
    Fit OLS and return result with firm-clustered standard errors.
    Uses statsmodels OLS with cov_type='cluster'.
    """
    res = sm.OLS(endog, exog, missing='drop').fit(
        cov_type='cluster',
        cov_kwds={'groups': cluster_var}
    )
    return res


def _add_year_dummies(df, year_col='period_year'):
    """Return a DataFrame of year dummy columns (drop-first)."""
    year_dummies = pd.get_dummies(df[year_col], prefix='yr', drop_first=True)
    return year_dummies.astype(float)


# ── In-sample regressions ─────────────────────────────────────────────────────

year_dummies_full = _add_year_dummies(df_reg)

# Baseline: RG_{t+1} = alpha + beta1*RG_t + year FE
X_baseline = sm.add_constant(
    pd.concat([df_reg[['Revenue_Growth']], year_dummies_full], axis=1),
    has_constant='add'
)
res_baseline = _cluster_se_ols(df_reg['RG_next'], X_baseline, df_reg['firm'])

# Decomposition: RG_{t+1} = alpha + beta1*AR_t + beta2*RRR_t + beta3*SoNR_t + year FE
X_decomp = sm.add_constant(
    pd.concat([df_reg[['Acq_Rate', 'RRR', 'SoNR']], year_dummies_full], axis=1),
    has_constant='add'
)
res_decomp = _cluster_se_ols(df_reg['RG_next'], X_decomp, df_reg['firm'])

# Combined: RG_{t+1} = alpha + beta1*RG_t + beta2*AR_t + beta3*RRR_t + beta4*SoNR_t + year FE
X_combined = sm.add_constant(
    pd.concat([df_reg[['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR']], year_dummies_full], axis=1),
    has_constant='add'
)
res_combined = _cluster_se_ols(df_reg['RG_next'], X_combined, df_reg['firm'])

print(f"\n  In-sample results:")
print(f"    Baseline    Adj-R2={res_baseline.rsquared_adj:.4f}  N={int(res_baseline.nobs)}")
print(f"    Decomp      Adj-R2={res_decomp.rsquared_adj:.4f}  N={int(res_decomp.nobs)}")
print(f"    Combined    Adj-R2={res_combined.rsquared_adj:.4f}  N={int(res_combined.nobs)}")

# F-test: incremental explanatory power of decomposition variables over lagged RG alone.
# H0: coefficients on AR, RRR, SoNR are jointly zero in the Combined model.
# Build restriction matrix manually to avoid API version differences in f_test().
decomp_var_names = [v for v in ['Acq_Rate', 'RRR', 'SoNR'] if v in res_combined.params.index]
try:
    param_names = list(res_combined.params.index)
    R_mat = np.zeros((len(decomp_var_names), len(param_names)))
    for row_i, vname in enumerate(decomp_var_names):
        col_i = param_names.index(vname)
        R_mat[row_i, col_i] = 1.0
    ftest = res_combined.f_test(R_mat)
    # .statistic may be a 2-D array or a scalar depending on statsmodels version
    ftest_stat_raw = ftest.statistic
    ftest_stat = float(np.asarray(ftest_stat_raw).flat[0])
    ftest_pval = float(ftest.pvalue)
    print(f"\n  F-test (decomp vars = 0 in Combined): F={ftest_stat:.3f}, p={ftest_pval:.4f}")
except Exception as e:
    ftest_stat = np.nan
    ftest_pval = np.nan
    print(f"  F-test failed: {e}")

# ── Out-of-sample expanding-window forecasting ────────────────────────────────

train_years = sorted(df_reg['period_year'].unique())
# First prediction is for year OOS_FIRST_TRAIN_END+1; train ends at OOS_FIRST_TRAIN_END
oos_results = []

predict_years = [y for y in train_years if y > OOS_FIRST_TRAIN_END]

for pred_year in predict_years:
    # Training data: all observations with period_year < pred_year
    df_train = df_reg[df_reg['period_year'] < pred_year].copy()
    df_test = df_reg[df_reg['period_year'] == pred_year - 1].copy()
    # Note: df_test rows have period_year = pred_year-1 and RG_next = RG at pred_year

    if len(df_train) < 5 or len(df_test) == 0:
        continue

    year_dummies_train = _add_year_dummies(df_train)
    year_dummies_test = _add_year_dummies(df_test)

    # Align dummy columns so test set has the same columns as train set
    for col in year_dummies_train.columns:
        if col not in year_dummies_test.columns:
            year_dummies_test[col] = 0.0
    year_dummies_test = year_dummies_test[year_dummies_train.columns]

    y_train = df_train['RG_next']
    y_test = df_test['RG_next']

    # Baseline model
    X_tr_b = sm.add_constant(
        pd.concat([df_train[['Revenue_Growth']], year_dummies_train], axis=1),
        has_constant='add'
    )
    X_te_b = sm.add_constant(
        pd.concat([df_test[['Revenue_Growth']], year_dummies_test], axis=1),
        has_constant='add'
    )
    X_te_b = X_te_b.reindex(columns=X_tr_b.columns, fill_value=0.0)

    # Decomposition model
    X_tr_d = sm.add_constant(
        pd.concat([df_train[['Acq_Rate', 'RRR', 'SoNR']], year_dummies_train], axis=1),
        has_constant='add'
    )
    X_te_d = sm.add_constant(
        pd.concat([df_test[['Acq_Rate', 'RRR', 'SoNR']], year_dummies_test], axis=1),
        has_constant='add'
    )
    X_te_d = X_te_d.reindex(columns=X_tr_d.columns, fill_value=0.0)

    # Combined model
    X_tr_c = sm.add_constant(
        pd.concat([df_train[['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR']], year_dummies_train], axis=1),
        has_constant='add'
    )
    X_te_c = sm.add_constant(
        pd.concat([df_test[['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR']], year_dummies_test], axis=1),
        has_constant='add'
    )
    X_te_c = X_te_c.reindex(columns=X_tr_c.columns, fill_value=0.0)

    try:
        m_b = sm.OLS(y_train, X_tr_b, missing='drop').fit()
        m_d = sm.OLS(y_train, X_tr_d, missing='drop').fit()
        m_c = sm.OLS(y_train, X_tr_c, missing='drop').fit()

        pred_b = m_b.predict(X_te_b)
        pred_d = m_d.predict(X_te_d)
        pred_c = m_c.predict(X_te_c)

        for idx in df_test.index:
            if idx not in pred_b.index:
                continue
            actual = y_test.loc[idx]
            if pd.isna(actual):
                continue
            oos_results.append({
                'pred_year': pred_year,
                'firm': df_test.loc[idx, 'firm'],
                'actual': actual,
                'pred_baseline': pred_b.loc[idx],
                'pred_decomp': pred_d.loc[idx],
                'pred_combined': pred_c.loc[idx],
            })
    except Exception as e:
        print(f"    OOS window {pred_year}: {e}")
        continue

df_oos = pd.DataFrame(oos_results)
print(f"\n  OOS sample: {len(df_oos)} obs across {df_oos['pred_year'].nunique()} forecast years "
      f"({df_oos['pred_year'].min()}-{df_oos['pred_year'].max()})")

# RMSE for each model
rmse_baseline = np.sqrt(((df_oos['actual'] - df_oos['pred_baseline']) ** 2).mean())
rmse_decomp = np.sqrt(((df_oos['actual'] - df_oos['pred_decomp']) ** 2).mean())
rmse_combined = np.sqrt(((df_oos['actual'] - df_oos['pred_combined']) ** 2).mean())

print(f"\n  OOS RMSE:")
print(f"    Baseline:     {rmse_baseline:.4f}")
print(f"    Decomposition:{rmse_decomp:.4f}")
print(f"    Combined:     {rmse_combined:.4f}")

# ── Clark-West test: baseline vs combined (nested model comparison) ────────────
# Reference: Clark and West (2007), Journal of Econometrics.
# For each OOS observation compute:
#   f_hat_t = (y - f1)^2 - [(y - f2)^2 - (f1 - f2)^2]
# Regress f_hat on a constant; t-stat on the constant is the CW statistic.
# One-sided test (H1: f_hat > 0, i.e., combined beats baseline after adjustment).

y_oos = df_oos['actual'].values
f1 = df_oos['pred_baseline'].values   # restricted (baseline)
f2 = df_oos['pred_combined'].values   # unrestricted (combined)

f_hat = (y_oos - f1) ** 2 - ((y_oos - f2) ** 2 - (f1 - f2) ** 2)
cw_reg = sm.OLS(f_hat, np.ones(len(f_hat))).fit(cov_type='HC1')
cw_stat = cw_reg.tvalues[0]
# One-sided p-value from standard normal
cw_pval = 1.0 - scipy_stats.norm.cdf(cw_stat)

print(f"\n  Clark-West test (baseline vs combined):")
print(f"    CW statistic: {cw_stat:.3f}")
print(f"    p-value (one-sided): {cw_pval:.4f}")


# ── LaTeX table: tab_forecasting.tex ─────────────────────────────────────────

def _coef_row(label, param_name_b, param_name_d, param_name_c,
               res_b, res_d, res_c):
    """Build two LaTeX rows (coef + t-stat) for one predictor across three models."""
    def _cell(res, name):
        if name in res.params.index:
            return _fmt_coef(res.params[name], res.pvalues[name])
        return ''

    def _tcell(res, name):
        if name in res.tvalues.index:
            return _fmt_tstat(res.tvalues[name])
        return ''

    coef_row = (
        f"  {label} & "
        f"{_cell(res_b, param_name_b)} & "
        f"{_cell(res_d, param_name_d)} & "
        f"{_cell(res_c, param_name_c)} \\\\"
    )
    tstat_row = (
        f"  & "
        f"{_tcell(res_b, param_name_b)} & "
        f"{_tcell(res_d, param_name_d)} & "
        f"{_tcell(res_c, param_name_c)} \\\\"
    )
    return coef_row + "\n" + tstat_row


n_obs_baseline = int(res_baseline.nobs)
n_obs_decomp = int(res_decomp.nobs)
n_obs_combined = int(res_combined.nobs)

adj_r2_b = res_baseline.rsquared_adj
adj_r2_d = res_decomp.rsquared_adj
adj_r2_c = res_combined.rsquared_adj

cw_stars = _stars(cw_pval)

forecasting_latex = r"""\begin{threeparttable}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
& \multicolumn{3}{c}{Dependent variable: $\RG_{i,t+1}$} \\
\cmidrule(lr){2-4}
& (1) Baseline & (2) Decomposition & (3) Combined \\
\midrule
"""

# Intercept row
forecasting_latex += _coef_row(
    r'Constant', 'const', 'const', 'const',
    res_baseline, res_decomp, res_combined
) + "\n"

# RG_t row (baseline and combined only)
forecasting_latex += _coef_row(
    r'$\RG_{i,t}$', 'Revenue_Growth', None, 'Revenue_Growth',
    res_baseline,
    type('R', (), {'params': pd.Series(dtype=float),
                   'pvalues': pd.Series(dtype=float),
                   'tvalues': pd.Series(dtype=float)})(),
    res_combined
) + "\n"

# AR_t row (decomposition and combined)
forecasting_latex += _coef_row(
    r'$\AR_{i,t}$', None, 'Acq_Rate', 'Acq_Rate',
    type('R', (), {'params': pd.Series(dtype=float),
                   'pvalues': pd.Series(dtype=float),
                   'tvalues': pd.Series(dtype=float)})(),
    res_decomp, res_combined
) + "\n"

# RRR_t row
forecasting_latex += _coef_row(
    r'$\RRR_{i,t}$', None, 'RRR', 'RRR',
    type('R', (), {'params': pd.Series(dtype=float),
                   'pvalues': pd.Series(dtype=float),
                   'tvalues': pd.Series(dtype=float)})(),
    res_decomp, res_combined
) + "\n"

# SoNR_t row
forecasting_latex += _coef_row(
    r'$\SoNR_{i,t}$', None, 'SoNR', 'SoNR',
    type('R', (), {'params': pd.Series(dtype=float),
                   'pvalues': pd.Series(dtype=float),
                   'tvalues': pd.Series(dtype=float)})(),
    res_decomp, res_combined
) + "\n"

forecasting_latex += r"""  Year FE & Yes & Yes & Yes \\
\midrule
"""

# Model fit stats
forecasting_latex += f"  Adj.~$R^2$ & {adj_r2_b:.3f} & {adj_r2_d:.3f} & {adj_r2_c:.3f} \\\\\n"
forecasting_latex += f"  $N$ & {n_obs_baseline:,} & {n_obs_decomp:,} & {n_obs_combined:,} \\\\\n"

forecasting_latex += r"""\midrule
\multicolumn{4}{l}{\textit{Out-of-sample performance (expanding window)}} \\
"""

forecasting_latex += (
    f"  OOS RMSE & {rmse_baseline:.4f} & {rmse_decomp:.4f} & {rmse_combined:.4f} \\\\\n"
)

# Clark-West test (baseline vs combined)
cw_pval_str = f"{cw_pval:.4f}" if cw_pval >= 0.0001 else "<0.0001"
forecasting_latex += (
    f"  Clark--West stat (1 vs 3) & \\multicolumn{{3}}{{c}}"
    f"{{{cw_stat:.3f}{cw_stars}}} \\\\\n"
    f"  CW $p$-value (one-sided) & \\multicolumn{{3}}{{c}}{{{cw_pval_str}}} \\\\\n"
)

forecasting_latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item \textit{Notes:} Dependent variable is revenue growth $\RG_{i,t+1}$ in the following period.
All models estimated by pooled OLS with year fixed effects. Standard errors clustered by firm;
$t$-statistics in parentheses.
The out-of-sample (OOS) RMSE uses an expanding-window design: training on all years up to $t$,
predicting year $t+1$, starting with a training window of 2018--2019 predicting 2020.
The Clark--West (2007) test adjusts for parameter estimation error in the larger model;
the statistic is the $t$-ratio from regressing $\hat{f}_t$ on a constant (one-sided normal $p$-value).
$\hat{f}_t = (y_t - \hat{f}^1_t)^2 - [(y_t - \hat{f}^3_t)^2 - (\hat{f}^1_t - \hat{f}^3_t)^2]$.
\item[***] $p < 0.01$; \item[**] $p < 0.05$; \item[*] $p < 0.10$.
\end{tablenotes}
\end{threeparttable}"""

tab_forecasting_path = os.path.join(LATEX_TABLE_DIR, 'tab_forecasting.tex')
with open(tab_forecasting_path, 'w', encoding='utf-8') as fh:
    fh.write(forecasting_latex)
print(f"\n  LaTeX table saved: {tab_forecasting_path}")


# ==============================================================================
# ANALYSIS 2: Future Revenue Volatility
# ==============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: FUTURE REVENUE VOLATILITY")
print("=" * 70)

# ── Panel A: Absolute forecast error regression ───────────────────────────────
# Dependent variable: |actual RG_{t+1} - predicted RG_{t+1}| from baseline model.
# Predictors: RRR_{i,t}, AR_{i,t}, year FE. Standard errors clustered by firm.

df_oos['abs_fe_baseline'] = (df_oos['actual'] - df_oos['pred_baseline']).abs()

# Merge predictors (AR and RRR at time t=pred_year-1) back into the OOS frame
df_oos_predictors = df_reg[['firm', 'period_year', 'Acq_Rate', 'RRR']].copy()
df_oos_predictors = df_oos_predictors.rename(columns={'period_year': 'pred_year_minus1'})
df_oos_with_pred_year = df_oos.copy()
# The OOS row at pred_year was generated from data at period_year = pred_year - 1
df_oos_with_pred_year['pred_year_minus1'] = df_oos_with_pred_year['pred_year'] - 1

df_fe_reg = df_oos_with_pred_year.merge(
    df_oos_predictors,
    left_on=['firm', 'pred_year_minus1'],
    right_on=['firm', 'pred_year_minus1'],
    how='inner'
)
df_fe_reg = df_fe_reg.dropna(subset=['abs_fe_baseline', 'RRR', 'Acq_Rate'])
print(f"  Panel A sample: {len(df_fe_reg)} obs, {df_fe_reg['firm'].nunique()} firms")

year_dummies_fe = pd.get_dummies(df_fe_reg['pred_year'], prefix='yr', drop_first=True).astype(float)
X_fe = sm.add_constant(
    pd.concat([df_fe_reg[['RRR', 'Acq_Rate']], year_dummies_fe], axis=1),
    has_constant='add'
)
res_fe = sm.OLS(df_fe_reg['abs_fe_baseline'], X_fe, missing='drop').fit(
    cov_type='cluster',
    cov_kwds={'groups': df_fe_reg['firm']}
)

print(f"    RRR coef: {res_fe.params.get('RRR', np.nan):.4f} "
      f"(t={res_fe.tvalues.get('RRR', np.nan):.2f}, "
      f"p={res_fe.pvalues.get('RRR', np.nan):.4f})")
print(f"    AR  coef: {res_fe.params.get('Acq_Rate', np.nan):.4f} "
      f"(t={res_fe.tvalues.get('Acq_Rate', np.nan):.2f}, "
      f"p={res_fe.pvalues.get('Acq_Rate', np.nan):.4f})")
print(f"    Adj-R2:   {res_fe.rsquared_adj:.4f}  N={int(res_fe.nobs)}")

# ── Panel B: Cross-sectional firm-level SD(RG) regression ─────────────────────
# Within-firm SD(RG) over the full sample regressed on mean RRR and mean AR.
# OLS with HC1 (heteroskedasticity-robust) standard errors.

df_firm_stats = (
    df_panel.groupby('firm')[['Revenue_Growth', 'RRR', 'Acq_Rate']]
    .agg(sd_rg=('Revenue_Growth', 'std'),
         mean_rrr=('RRR', 'mean'),
         mean_ar=('Acq_Rate', 'mean'))
    .dropna()
    .reset_index()
)
print(f"\n  Panel B sample: {len(df_firm_stats)} firms")

X_sd = sm.add_constant(df_firm_stats[['mean_rrr', 'mean_ar']], has_constant='add')
res_sd = sm.OLS(df_firm_stats['sd_rg'], X_sd, missing='drop').fit(cov_type='HC1')

print(f"    mean_RRR coef: {res_sd.params.get('mean_rrr', np.nan):.4f} "
      f"(t={res_sd.tvalues.get('mean_rrr', np.nan):.2f}, "
      f"p={res_sd.pvalues.get('mean_rrr', np.nan):.4f})")
print(f"    mean_AR  coef: {res_sd.params.get('mean_ar', np.nan):.4f} "
      f"(t={res_sd.tvalues.get('mean_ar', np.nan):.2f}, "
      f"p={res_sd.pvalues.get('mean_ar', np.nan):.4f})")
print(f"    Adj-R2:        {res_sd.rsquared_adj:.4f}  N={int(res_sd.nobs)}")

# ── Panel C: Quadrant comparison of absolute forecast errors ──────────────────
# Merge OOS absolute forecast errors with quadrant labels

df_oos_quad = df_oos_with_pred_year.merge(
    firm_medians_full[['quadrant']].reset_index(),
    on='firm', how='left'
)

quad_fe_stats = []
for q_label in ['HH', 'HL', 'LH', 'LL']:
    sub = df_oos_quad[df_oos_quad['quadrant'] == q_label]['abs_fe_baseline'].dropna()
    quad_fe_stats.append({
        'Quadrant': q_label,
        'Mean abs FE': sub.mean(),
        'Median abs FE': sub.median(),
        'N': len(sub),
    })

df_quad_fe = pd.DataFrame(quad_fe_stats)
print(f"\n  Panel C: Quadrant absolute forecast errors")
print(df_quad_fe.to_string(index=False, float_format='{:.4f}'.format))


# ── LaTeX table: tab_forward_volatility.tex ───────────────────────────────────

def _vol_coef_row(label, param_name, res, n_models=2):
    """Build coef + t-stat rows for a volatility regression."""
    coef = res.params.get(param_name, np.nan)
    pval = res.pvalues.get(param_name, np.nan)
    tval = res.tvalues.get(param_name, np.nan)
    stars = _stars(pval) if not np.isnan(pval) else ''
    coef_str = f"{coef:.3f}{stars}" if not np.isnan(coef) else ''
    tval_str = f"({tval:.2f})" if not np.isnan(tval) else ''
    return (
        f"  {label} & {coef_str} \\\\\n"
        f"  & {tval_str} \\\\"
    )


n_obs_fe = int(res_fe.nobs)
n_obs_sd = int(res_sd.nobs)

vol_latex = r"""\begin{threeparttable}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}
\toprule
& Panel A & Panel B \\
& Abs.\ forecast error & Firm SD(\RG) \\
& (OOS, baseline model) & (cross-sectional) \\
\midrule
\multicolumn{3}{l}{\textit{Panel A: Forecast error regression}} \\[2pt]
"""

# Panel A rows
for pname, label in [('const', 'Constant'), ('RRR', r'$\RRR_{i,t}$'), ('Acq_Rate', r'$\AR_{i,t}$')]:
    coef = res_fe.params.get(pname, np.nan)
    pval = res_fe.pvalues.get(pname, np.nan)
    tval = res_fe.tvalues.get(pname, np.nan)
    stars = _stars(pval) if not np.isnan(pval) else ''
    coef_str = f"{coef:.3f}{stars}" if not np.isnan(coef) else ''
    tval_str = f"({tval:.2f})" if not np.isnan(tval) else ''
    vol_latex += f"  {label} & {coef_str} & \\\\\n"
    vol_latex += f"  & {tval_str} & \\\\\n"

vol_latex += (
    f"  Year FE & Yes & --- \\\\\n"
    f"  Clustered SE & Yes & No (HC1) \\\\\n"
    f"  Adj.~$R^2$ & {res_fe.rsquared_adj:.3f} & \\\\\n"
    f"  $N$ & {n_obs_fe:,} & \\\\\n"
)

vol_latex += r"""\midrule
\multicolumn{3}{l}{\textit{Panel B: Cross-sectional firm-level SD(\RG) regression}} \\[2pt]
"""

# Panel B rows
for pname, label in [('const', 'Constant'), ('mean_rrr', r'Mean $\RRR_i$'), ('mean_ar', r'Mean $\AR_i$')]:
    coef = res_sd.params.get(pname, np.nan)
    pval = res_sd.pvalues.get(pname, np.nan)
    tval = res_sd.tvalues.get(pname, np.nan)
    stars = _stars(pval) if not np.isnan(pval) else ''
    coef_str = f"{coef:.3f}{stars}" if not np.isnan(coef) else ''
    tval_str = f"({tval:.2f})" if not np.isnan(tval) else ''
    vol_latex += f"  {label} & & {coef_str} \\\\\n"
    vol_latex += f"  & & {tval_str} \\\\\n"

vol_latex += (
    f"  Adj.~$R^2$ & & {res_sd.rsquared_adj:.3f} \\\\\n"
    f"  $N$ (firms) & & {n_obs_sd:,} \\\\\n"
)

vol_latex += r"""\midrule
\multicolumn{3}{l}{\textit{Panel C: Absolute forecast errors by quadrant}} \\[2pt]
  Quadrant & Mean $|\hat{e}|$ & Median $|\hat{e}|$ \\
\cmidrule(lr){1-3}
"""

for _, row in df_quad_fe.iterrows():
    n_q = int(row['N'])
    vol_latex += (
        f"  {row['Quadrant']} & {row['Mean abs FE']:.4f} & {row['Median abs FE']:.4f} \\\\\n"
    )

vol_latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item \textit{Notes:}
Panel~A regresses the absolute forecast error from the out-of-sample baseline model on
$\RRR_{i,t}$ and $\AR_{i,t}$ with year fixed effects.
Standard errors are clustered by firm. $t$-statistics in parentheses.
Panel~B regresses within-firm standard deviation of revenue growth (computed over the full sample)
on the firm-level mean of $\RRR$ and mean of $\AR$.
Standard errors use the HC1 heteroskedasticity-robust estimator.
Panel~C reports mean and median absolute forecast errors (from the OOS baseline model)
by firm quadrant (HH/HL/LH/LL), where quadrant membership is based on firm-median
$\AR$ and firm-median $\RRR$ relative to cross-firm medians.
\item[***] $p < 0.01$; \item[**] $p < 0.05$; \item[*] $p < 0.10$.
\end{tablenotes}
\end{threeparttable}"""

tab_vol_path = os.path.join(LATEX_TABLE_DIR, 'tab_forward_volatility.tex')
with open(tab_vol_path, 'w', encoding='utf-8') as fh:
    fh.write(vol_latex)
print(f"\n  LaTeX table saved: {tab_vol_path}")


# ==============================================================================
# ANALYSIS 3: Q_t Robustness Excluding 2020-2021
# ==============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: Q_t ROBUSTNESS (EXCLUDING 2020-2021)")
print("=" * 70)

if 'Q_t' not in df_panel.columns:
    print("  Q_t column not found in panel — skipping.")
else:
    df_excl_covid = df_panel[~df_panel['period_year'].isin(COVID_YEARS)].copy()

    qt_excl = df_excl_covid[['firm', 'Q_t']].copy()
    qt_excl['Q_t'] = pd.to_numeric(qt_excl['Q_t'], errors='coerce')
    qt_excl = qt_excl.dropna(subset=['Q_t'])

    by_firm_excl = qt_excl.groupby('firm')['Q_t']

    always_below_excl = by_firm_excl.apply(lambda s: (s < 1).all())
    always_above_excl = by_firm_excl.apply(lambda s: (s > 1).all())
    mixed_excl = by_firm_excl.apply(lambda s: (s < 1).any() and (s > 1).any())

    total_q_excl = qt_excl['firm'].nunique()

    n_acq_excl = always_below_excl.sum()
    n_ret_excl = always_above_excl.sum()
    n_mix_excl = mixed_excl.sum()

    pct_acq_excl = n_acq_excl / total_q_excl * 100
    pct_ret_excl = n_ret_excl / total_q_excl * 100
    pct_mix_excl = n_mix_excl / total_q_excl * 100

    print(f"  Firms with Q_t data (excl 2020-21): {total_q_excl}")
    print(f"  Always Q_t < 1 (acquisition-driven): {n_acq_excl} ({pct_acq_excl:.1f}%)")
    print(f"  Always Q_t > 1 (retention-driven):   {n_ret_excl} ({pct_ret_excl:.1f}%)")
    print(f"  Mixed (both sides of 1):              {n_mix_excl} ({pct_mix_excl:.1f}%)")

    # Full-sample Q_t classification for comparison
    qt_full = df_panel[['firm', 'Q_t']].copy()
    qt_full['Q_t'] = pd.to_numeric(qt_full['Q_t'], errors='coerce')
    qt_full = qt_full.dropna(subset=['Q_t'])
    by_firm_full = qt_full.groupby('firm')['Q_t']
    always_below_full = by_firm_full.apply(lambda s: (s < 1).all())
    always_above_full = by_firm_full.apply(lambda s: (s > 1).all())
    mixed_full = by_firm_full.apply(lambda s: (s < 1).any() and (s > 1).any())
    total_q_full = qt_full['firm'].nunique()

    print(f"\n  Full-sample Q_t for comparison:")
    print(f"  Firms: {total_q_full}")
    print(f"  Always acquisition-driven: {always_below_full.sum()} ({always_below_full.sum()/total_q_full*100:.1f}%)")
    print(f"  Always retention-driven:   {always_above_full.sum()} ({always_above_full.sum()/total_q_full*100:.1f}%)")
    print(f"  Mixed:                     {mixed_full.sum()} ({mixed_full.sum()/total_q_full*100:.1f}%)")

    print("\n  *** Updated Panel C for tab_robustness_excl_covid.tex ***")
    print(f"  Acquisition-driven (full):  {always_below_full.sum()/total_q_full*100:.1f}%  "
          f"| Excl 2020-21: {pct_acq_excl:.1f}%")
    print(f"  Retention-driven (full):    {always_above_full.sum()/total_q_full*100:.1f}%  "
          f"| Excl 2020-21: {pct_ret_excl:.1f}%")
    print(f"  Mixed (full):               {mixed_full.sum()/total_q_full*100:.1f}%  "
          f"| Excl 2020-21: {pct_mix_excl:.1f}%")

    print("\n  *** LaTeX snippet for Panel C of tab_robustness_excl_covid.tex ***")
    print(r"  \multicolumn{3}{l}{\textit{Panel C: $q_t$ Classification}} \\")
    print(f"  Acquisition-driven ($q_t < 1$ always) & "
          f"{always_below_full.sum()/total_q_full*100:.1f}\\% & {pct_acq_excl:.1f}\\% \\\\")
    print(f"  Retention-driven ($q_t > 1$ always) & "
          f"{always_above_full.sum()/total_q_full*100:.1f}\\% & {pct_ret_excl:.1f}\\% \\\\")
    print(f"  Mixed & "
          f"{mixed_full.sum()/total_q_full*100:.1f}\\% & {pct_mix_excl:.1f}\\% \\\\")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 70)
print("ALL VALIDATION ANALYSES COMPLETED")
print("=" * 70)
print(f"LaTeX tables written to: {LATEX_TABLE_DIR}")
print(f"  tab_forecasting.tex")
print(f"  tab_forward_volatility.tex")
print(f"(Panel C Q_t numbers printed to console above for manual update of")
print(f" tab_robustness_excl_covid.tex)")
