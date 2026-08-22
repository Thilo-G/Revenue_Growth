"""
new_analyses_v3.py -- Additional analyses for the Revenue Growth paper (v3 extension)
Extends new_analyses_v2.py with 4 new outputs:
  1. tab_sample_construction    -- LaTeX table of sample construction steps
  2. F_histograms_metrics       -- 2x2 histogram figure (RG, AR, RRR, SoNR)
  3. tab_uncertainty_regression -- OLS of within-firm RG variance on AR, RRR, SoNR, SoNR*AR
  4. tab_downside_risk_maxdrawdown -- OLS of worst-year RG and max drawdown on predictors
Data loading is copied exactly from new_analyses_v2.py.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# -- Paths --------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'RG_LaTeX', 'figures')
LATEX_TABLE_DIR = os.path.join(
    "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft"
    "\\Thilo(privat)\\Privat\\Research\\Revenue_Growth\\RG_LaTeX\\tables"
)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LATEX_TABLE_DIR, exist_ok=True)

ANALYSIS_END_YEAR = 2025

# -- Plot style (match v2 script) ---------------------------------------------
plt.rcParams.update({
    'figure.figsize': (9, 5.5),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'acq': '#d62728',
    'ret': '#1f77b4',
    'median_line': '#d62728',
    'mean_line': '#2ca02c',
}


def save_fig(fig, name):
    """Save figure as PDF and PNG to FIG_DIR."""
    path_pdf = os.path.join(FIG_DIR, f'{name}.pdf')
    path_png = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight', dpi=150)
    print(f"  Saved: {path_pdf}")


# =============================================================================
# DATA LOADING (copied exactly from new_analyses_v2.py)
# =============================================================================

file1_path = (
    "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)"
    "\\Privat\\Research\\Revenue_Growth\\RG_Data"
    "\\2026-02-18-Firm-Year-Revenue-Python.xlsx"
)
file2_path = (
    "C:\\Users\\thkraft\\eCommerce-Goethe Dropbox\\Thilo Kraft\\Thilo(privat)"
    "\\Privat\\Research\\Revenue_Growth\\RG_Data"
    "\\Firms-Industry-2025-11-21-Python.xlsx"
)

print("Loading data...")
df_revenue = pd.read_excel(file1_path, header=None)
df_industry = pd.read_excel(file2_path, header=0)

# Record raw universe count before any filtering (for sample construction table)
n_raw_universe = df_revenue.shape[1] - 1   # subtract the Date column (col 0)

# -- Industry filtering (match v2) --------------------------------------------
df_industry['gics_sub_industry_name'] = (
    df_industry['gics_sub_industry_name']
    .astype(str).str.strip()
    .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
)
df_industry = df_industry.dropna(subset=['gics_sub_industry_name']).copy()

# Count after GICS mapping (before sub-industry size filter)
n_after_gics_map = df_industry['ID'].nunique()

sub_counts = df_industry.groupby('gics_sub_industry_name')['ID'].nunique()
keep_subs = sub_counts[sub_counts >= 2].index
df_industry = df_industry[df_industry['gics_sub_industry_name'].isin(keep_subs)].copy()

# Count after sub-industry size filter
n_after_subind_filter = df_industry['ID'].nunique()


def _norm_id(x):
    """Normalise firm ID to uppercase string without trailing .0."""
    s = str(x).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    return s


valid_ids_norm = set(_norm_id(x) for x in df_industry["ID"].dropna())

# -- Filter revenue columns to match industry IDs -----------------------------
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

# -- Header processing (match v2) ---------------------------------------------
header_rows_rev = df_revenue.iloc[:2].copy()
data_rows_rev = df_revenue.iloc[2:].copy()

combined_headers = header_rows_rev.apply(lambda x: x.astype(str).str.strip())
combined_headers = combined_headers.replace(
    {'nan': pd.NA, 'NaT': pd.NA, 'None': pd.NA, '': pd.NA}
)
column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
data_rows_rev.columns = column_headers
data_rows_rev = data_rows_rev.rename(columns={data_rows_rev.columns[0]: 'Date'})
df_revenue = data_rows_rev.reset_index(drop=True)
df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True)
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'], errors='coerce')
df_revenue = df_revenue[df_revenue['Date'].notna()].copy()
df_revenue = df_revenue[df_revenue['Date'].dt.year <= ANALYSIS_END_YEAR].copy()
df_revenue = df_revenue.sort_values('Date').reset_index(drop=True)

# -- Identify New/Returning columns -------------------------------------------
new_customer_cols = [
    c for c in df_revenue.columns
    if '#New_Customers' in c or '#New_Customer' in c
]
returning_customer_cols = [
    c for c in df_revenue.columns
    if '#Returning_Customers' in c or '#Returning_Customer' in c
]


def _base_name(col_name):
    """Strip metric suffix to get the base firm ID portion of a column name."""
    return (
        str(col_name)
        .replace('#New_Customers', '')
        .replace('#New_Customer', '')
        .replace('#Returning_Customers', '')
        .replace('#Returning_Customer', '')
        .lstrip('#')
        .strip('. ')
    )


# -- Drop firms with missing values -------------------------------------------
candidate_firms = sorted(
    {_base_name(c) for c in (new_customer_cols + returning_customer_cols)}
)
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
    cols_to_drop = [
        c for c in df_revenue.columns
        if c != 'Date' and _base_name(c) in firms_drop
    ]
    df_revenue = df_revenue.drop(columns=cols_to_drop)

new_customer_cols = [
    c for c in df_revenue.columns
    if '#New_Customers' in c or '#New_Customer' in c
]
returning_customer_cols = [
    c for c in df_revenue.columns
    if '#Returning_Customers' in c or '#Returning_Customer' in c
]

# -- Compute derived metrics (match v2) ---------------------------------------
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
        new_cols_data[q_t_col] = (
            (ret_s / total_prev_nonzero) / acq_rate.replace(0, np.nan)
        )
        sonr = new_s / total_nonzero
        new_cols_data[sonr_col] = sonr
        new_cols_data[nrlc_col] = acq_rate * sonr.shift(1)
        final_cols.extend([
            total_revenue_col, rrr_col, revenue_growth_col,
            acq_rate_col, q_t_col, sonr_col, nrlc_col
        ])

if new_cols_data:
    df_new = pd.DataFrame(new_cols_data, index=df_revenue.index)
    df_revenue = pd.concat([df_revenue, df_new], axis=1)

final_cols = [c for c in final_cols if c in df_revenue.columns]
df_revenue = df_revenue.loc[:, final_cols]

# -- Transform to panel (long) format -----------------------------------------
df_revenue['Date'] = pd.to_datetime(df_revenue['Date'], errors='coerce')
df_revenue = df_revenue.sort_values('Date').reset_index(drop=True)

metrics_suffixes = [
    '#Revenue_Growth', '#RRR', '#Acq_Rate', '#Q_t',
    '#Total_Revenue', '#SoNR', '#NRLC'
]
present_metric_cols = [
    c for c in df_revenue.columns
    if any(c.endswith(s) for s in metrics_suffixes)
]

df_wide = df_revenue.copy()
df_wide['Date'] = pd.to_datetime(df_wide['Date'], errors='coerce')
metric_cols = [c for c in df_wide.columns if c in present_metric_cols]
df_long = df_wide.melt(
    id_vars='Date', value_vars=metric_cols,
    var_name='col', value_name='value'
)
df_long['firm'] = df_long['col'].apply(
    lambda c: c.split('#')[0] if '#' in c else c
)
df_long['metric'] = df_long['col'].apply(
    lambda c: c.split('#', 1)[1] if '#' in c else ''
)

df_panel = (
    df_long
    .pivot_table(
        index=['Date', 'firm'], columns='metric',
        values='value', aggfunc='first'
    )
    .reset_index()
)
df_panel.columns = [
    c if isinstance(c, str) else c[1] for c in df_panel.columns
]
for col in ['Revenue_Growth', 'RRR', 'Acq_Rate', 'Q_t', 'Total_Revenue', 'SoNR', 'NRLC']:
    if col in df_panel.columns:
        df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce')

df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')
df_panel = df_panel.sort_values(['firm', 'Date']).reset_index(drop=True)
df_panel['period_year'] = df_panel['Date'].dt.year.astype(float)
df_panel = df_panel[df_panel['period_year'] <= ANALYSIS_END_YEAR].copy()


# Attach industry labels
def attach_industry_labels(panel_df, industry_df):
    """Merge GICS sub-industry names onto the panel using normalised IDs."""
    out = panel_df.copy()
    ind = industry_df[['ID', 'gics_sub_industry_name']].copy()
    ind['id_norm'] = ind['ID'].apply(_norm_id)
    ind = (
        ind.dropna(subset=['id_norm', 'gics_sub_industry_name'])
        .drop_duplicates('id_norm')
    )
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

# -- Trim outliers (match v2: 1st/98th percentiles) ---------------------------
for metric in ['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR', 'NRLC']:
    if metric not in df_panel.columns:
        continue
    x = pd.to_numeric(df_panel[metric], errors='coerce')
    lower = x.quantile(0.01)
    upper = x.quantile(0.98)
    df_panel.loc[x < lower, metric] = np.nan
    df_panel.loc[x > upper, metric] = np.nan

n_firms_final = df_panel['firm'].nunique()
print(f"\nPanel loaded: {len(df_panel)} rows, {n_firms_final} firms")
print(f"Years: {sorted(df_panel['period_year'].dropna().unique().astype(int))}")


# =============================================================================
# NEW OUTPUT 1: tab_sample_construction
# LaTeX table showing the sample construction funnel step by step.
# =============================================================================
print(f"\n{'='*70}")
print("NEW ANALYSIS 1: SAMPLE CONSTRUCTION TABLE")
print(f"{'='*70}")

# Derive counts already captured during loading above
n_bloomberg_universe = n_raw_universe
n_after_gics = n_after_gics_map
n_after_subind = n_after_subind_filter
n_final = n_firms_final

print(f"  Bloomberg Second Measure universe:          {n_bloomberg_universe}")
print(f"  After requiring GICS sub-industry mapping:  {n_after_gics}")
print(f"  After requiring >=2 firms per sub-industry: {n_after_subind}")
print(f"  After requiring complete revenue data:       {n_final}")

sample_table_latex = (
    "\\begin{tabular}{lc}\n"
    "\\toprule\n"
    "Filter step & Firms \\\\\n"
    "\\midrule\n"
    f"Bloomberg Second Measure universe & {n_bloomberg_universe:,} \\\\\n"
    f"After requiring GICS sub-industry mapping & {n_after_gics:,} \\\\\n"
    f"After requiring $\\geq$2 firms per GICS sub-industry & {n_after_subind:,} \\\\\n"
    f"After requiring complete revenue data (2018--2025) & {n_final:,} \\\\\n"
    "\\midrule\n"
    f"\\textbf{{Final sample}} & \\textbf{{{n_final:,}}} \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

sample_table_path = os.path.join(LATEX_TABLE_DIR, 'tab_sample_construction.tex')
with open(sample_table_path, 'w', encoding='utf-8') as f:
    f.write(sample_table_latex)
print(f"  LaTeX table saved: {sample_table_path}")


# =============================================================================
# NEW OUTPUT 2: F_histograms_metrics
# 2x2 figure with histograms for RG, AR, RRR, SoNR (pooled, trimmed values).
# =============================================================================
print(f"\n{'='*70}")
print("NEW ANALYSIS 2: METRIC HISTOGRAMS (2x2 FIGURE)")
print(f"{'='*70}")

# Pooled, trimmed series (NaNs already removed by trimming step above)
rg_vals = df_panel['Revenue_Growth'].dropna()
ar_vals = df_panel['Acq_Rate'].dropna()
rrr_vals = df_panel['RRR'].dropna()
sonr_vals = df_panel['SoNR'].dropna()

# Fraction of RRR observations below 1 (for annotation)
pct_rrr_below_1 = (rrr_vals < 1).mean() * 100

fig_hist, axes = plt.subplots(2, 2, figsize=(12, 9))
ax_rg = axes[0, 0]
ax_ar = axes[0, 1]
ax_rrr = axes[1, 0]
ax_sonr = axes[1, 1]


def _draw_histogram(ax, data, title, xlabel, bins=40, vline_val=None, vline_label=None):
    """Draw a single histogram panel with median/mean vertical lines."""
    ax.hist(data, bins=bins, color=PALETTE['primary'], alpha=0.7, edgecolor='white')
    median_val = data.median()
    mean_val = data.mean()
    ax.axvline(median_val, color=PALETTE['median_line'], linewidth=1.8,
               linestyle='-', label=f'Median = {median_val:.3f}')
    ax.axvline(mean_val, color=PALETTE['mean_line'], linewidth=1.8,
               linestyle='--', label=f'Mean = {mean_val:.3f}')
    if vline_val is not None:
        ax.axvline(vline_val, color='black', linewidth=1.4,
                   linestyle=':', label=vline_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)


# RG panel
_draw_histogram(
    ax_rg, rg_vals,
    title='Revenue Growth (RG)',
    xlabel='RG (pooled, trimmed)'
)

# AR panel
_draw_histogram(
    ax_ar, ar_vals,
    title='Acquisition Rate (AR)',
    xlabel='AR (pooled, trimmed)'
)

# RRR panel -- vertical line at RRR = 1, annotation on fraction below 1
_draw_histogram(
    ax_rrr, rrr_vals,
    title='Revenue Retention Rate (RRR)',
    xlabel='RRR (pooled, trimmed)',
    vline_val=1.0,
    vline_label='RRR = 1'
)
ax_rrr.text(
    0.97, 0.95,
    f'{pct_rrr_below_1:.0f}% of obs. have RRR < 1',
    transform=ax_rrr.transAxes, fontsize=8.5,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow', alpha=0.85)
)

# SoNR panel -- vertical line at SoNR = 0.5
_draw_histogram(
    ax_sonr, sonr_vals,
    title='Share of New Revenue (SoNR)',
    xlabel='SoNR (pooled, trimmed)',
    vline_val=0.5,
    vline_label='SoNR = 0.5'
)

plt.suptitle(
    f'Distributions of Revenue Growth Metrics  |  N={len(df_panel):,} obs, pooled 2018-2025',
    fontsize=12, y=1.01
)
plt.tight_layout()
save_fig(fig_hist, 'F_histograms_metrics')
plt.close(fig_hist)


# =============================================================================
# NEW OUTPUT 3: tab_uncertainty_regression
# Cross-sectional OLS: within-firm Var(RG) on mean AR, mean RRR, mean SoNR,
# and the mean of the firm-year product SoNR*AR (interaction term).
# =============================================================================
print(f"\n{'='*70}")
print("NEW ANALYSIS 3: UNCERTAINTY REGRESSION")
print(f"{'='*70}")

# Sort panel before computing lagged SoNR (used in the SoNR*AR interaction)
df_sorted = df_panel.sort_values(['firm', 'Date']).copy()

# Compute the firm-year interaction: SoNR_{t-1} * AR_t (acquisition contribution)
df_sorted['SoNR_lag1'] = df_sorted.groupby('firm')['SoNR'].shift(1)
df_sorted['sonr_ar_product'] = df_sorted['SoNR_lag1'] * df_sorted['Acq_Rate']

# Build cross-sectional dataset -- one row per firm
firm_var_rg = (
    df_sorted.groupby('firm')['Revenue_Growth']
    .apply(lambda s: s.dropna().var(ddof=1) if s.dropna().shape[0] >= 2 else np.nan)
    .rename('var_rg_i')
)
firm_mean_ar = df_sorted.groupby('firm')['Acq_Rate'].mean().rename('mean_ar_i')
firm_mean_rrr = df_sorted.groupby('firm')['RRR'].mean().rename('mean_rrr_i')
firm_mean_sonr = df_sorted.groupby('firm')['SoNR'].mean().rename('mean_sonr_i')
firm_mean_sonr_ar = (
    df_sorted.groupby('firm')['sonr_ar_product']
    .mean()
    .rename('sonr_ar_i')
)

# Merge all firm-level aggregates into one cross-sectional frame
df_unc = (
    pd.concat(
        [firm_var_rg, firm_mean_ar, firm_mean_rrr, firm_mean_sonr, firm_mean_sonr_ar],
        axis=1
    )
    .dropna()
    .reset_index()
)
print(f"  Cross-sectional sample for uncertainty regression: {len(df_unc)} firms")

# OLS: var_rg_i ~ mean_ar_i + mean_rrr_i + mean_sonr_i + sonr_ar_i
X_unc = sm.add_constant(df_unc[['mean_ar_i', 'mean_rrr_i', 'mean_sonr_i', 'sonr_ar_i']])
y_unc = df_unc['var_rg_i']
res_unc = sm.OLS(y_unc, X_unc).fit(cov_type='HC1')

print(f"  N = {int(res_unc.nobs)}, R2 = {res_unc.rsquared:.4f}")
for var in X_unc.columns:
    coef = res_unc.params[var]
    tstat = res_unc.tvalues[var]
    pval = res_unc.pvalues[var]
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
    print(f"    {var:15s}: coef={coef:+.6f}, t={tstat:+.3f}, p={pval:.3f} {stars}")


def _stars(pval):
    """Return LaTeX significance star string for a p-value."""
    if pval < 0.01:
        return '$^{***}$'
    elif pval < 0.05:
        return '$^{**}$'
    elif pval < 0.10:
        return '$^{*}$'
    return ''


def _fmt_coef(res, varname):
    """Format coefficient with significance stars for LaTeX."""
    coef = res.params[varname]
    pval = res.pvalues[varname]
    return f"{coef:.4f}{_stars(pval)}"


def _fmt_se(res, varname):
    """Format standard error in parentheses for LaTeX."""
    se = res.bse[varname]
    return f"({se:.4f})"


# Build LaTeX table rows for the uncertainty regression
var_map_unc = [
    ('mean_ar_i',    'Mean $\\AR$'),
    ('mean_rrr_i',   'Mean $\\RRR$'),
    ('mean_sonr_i',  'Mean $\\SoNR$'),
    ('sonr_ar_i',    'Mean $\\SoNR \\times \\AR$'),
    ('const',        'Constant'),
]

unc_rows = []
for varname, label in var_map_unc:
    unc_rows.append(f"  {label} & {_fmt_coef(res_unc, varname)} \\\\")
    unc_rows.append(f"   & {_fmt_se(res_unc, varname)} \\\\")

unc_table_latex = (
    "\\begin{tabular}{lc}\n"
    "\\toprule\n"
    " & $\\mathrm{Var}(\\RG)_i$ \\\\\n"
    "\\midrule\n"
    + "\n".join(unc_rows)
    + "\n\\midrule\n"
    f"  $N$ & {int(res_unc.nobs)} \\\\\n"
    f"  $R^2$ & {res_unc.rsquared:.4f} \\\\\n"
    "\\bottomrule\n"
    "\\multicolumn{2}{l}{\\textit{Note: OLS with HC1 robust standard errors. "
    "SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}}\n"
    "\\end{tabular}"
)

unc_table_path = os.path.join(LATEX_TABLE_DIR, 'tab_uncertainty_regression.tex')
with open(unc_table_path, 'w', encoding='utf-8') as f:
    f.write(unc_table_latex)
print(f"  LaTeX table saved: {unc_table_path}")


# =============================================================================
# NEW OUTPUT 4: tab_downside_risk_maxdrawdown
# Two OLS specifications:
#   Spec A: worst_year_rg_i  ~ mean_ar_i + mean_rrr_i + sonr_ar_i
#   Spec B: max_drawdown_i   ~ mean_ar_i + mean_rrr_i + sonr_ar_i
#
# worst_year_rg:   min annual RG for the firm (worst single year)
# max_drawdown:    largest cumulative decline = min sum of a contiguous run
#                  of consecutive negative RG values (more negative = worse)
# =============================================================================
print(f"\n{'='*70}")
print("NEW ANALYSIS 4: DOWNSIDE RISK -- MAX DRAWDOWN REGRESSION")
print(f"{'='*70}")

# -- Worst single-year RG (Spec A dependent variable) -------------------------
firm_worst_year = (
    df_sorted.groupby('firm')['Revenue_Growth']
    .min()
    .rename('worst_year_rg_i')
)

# -- True max drawdown: sum of the worst contiguous run of negative RG years --
# Definition: find the contiguous subsequence of annual RG values that are all
# negative and sum to the most negative total; report that sum.
# If a firm has no year with negative RG, max_drawdown = 0.

def _max_contiguous_drawdown(rg_array):
    """
    Return the most negative sum of any contiguous run of negative RG values.
    Result is <= 0. Returns 0.0 if no negative RG years exist.
    """
    rg_sorted = rg_array[~np.isnan(rg_array)]
    if len(rg_sorted) == 0:
        return np.nan
    min_sum = 0.0          # worst (most negative) cumulative sum found
    current_sum = 0.0      # running sum of current negative run
    for val in rg_sorted:
        if val < 0:
            current_sum += val
            if current_sum < min_sum:
                min_sum = current_sum
        else:
            current_sum = 0.0   # reset run when a non-negative year breaks the streak
    return min_sum


firm_max_drawdown = (
    df_sorted.sort_values(['firm', 'Date'])
    .groupby('firm')['Revenue_Growth']
    .apply(lambda s: _max_contiguous_drawdown(s.values))
    .rename('max_drawdown_i')
)

# -- Mean predictors (same as uncertainty regression) -------------------------
# Reuse firm_mean_ar, firm_mean_rrr, firm_mean_sonr_ar from Analysis 3
df_ds = (
    pd.concat(
        [firm_worst_year, firm_max_drawdown, firm_mean_ar, firm_mean_rrr, firm_mean_sonr_ar],
        axis=1
    )
    .reset_index()
)

# Require at least 3 RG observations per firm for reliability
firm_n_obs = (
    df_sorted.groupby('firm')['Revenue_Growth']
    .apply(lambda s: s.dropna().shape[0])
    .rename('n_obs')
)
df_ds = df_ds.merge(firm_n_obs.reset_index(), on='firm', how='left')
df_ds = df_ds[df_ds['n_obs'] >= 3].copy()
print(f"  Cross-sectional sample (>=3 obs): {len(df_ds)} firms")

# -- Spec A: worst_year_rg_i ~ mean_ar_i + mean_rrr_i + sonr_ar_i ------------
X_ds_cols = ['mean_ar_i', 'mean_rrr_i', 'sonr_ar_i']
df_specA = df_ds[['worst_year_rg_i'] + X_ds_cols].dropna()
X_specA = sm.add_constant(df_specA[X_ds_cols])
res_specA = sm.OLS(df_specA['worst_year_rg_i'], X_specA).fit(cov_type='HC1')

print(f"\n  Spec A (worst_year_rg): N={int(res_specA.nobs)}, R2={res_specA.rsquared:.4f}")
for var in X_specA.columns:
    coef = res_specA.params[var]
    pval = res_specA.pvalues[var]
    stars_str = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
    print(f"    {var:12s}: coef={coef:+.4f}, p={pval:.3f} {stars_str}")

# -- Spec B: max_drawdown_i ~ mean_ar_i + mean_rrr_i + sonr_ar_i -------------
df_specB = df_ds[['max_drawdown_i'] + X_ds_cols].dropna()
X_specB = sm.add_constant(df_specB[X_ds_cols])
res_specB = sm.OLS(df_specB['max_drawdown_i'], X_specB).fit(cov_type='HC1')

print(f"\n  Spec B (max_drawdown): N={int(res_specB.nobs)}, R2={res_specB.rsquared:.4f}")
for var in X_specB.columns:
    coef = res_specB.params[var]
    pval = res_specB.pvalues[var]
    stars_str = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
    print(f"    {var:12s}: coef={coef:+.4f}, p={pval:.3f} {stars_str}")

# -- Build two-column LaTeX regression table ----------------------------------
var_map_ds = [
    ('mean_ar_i',  '$\\overline{\\AR}_i$'),
    ('mean_rrr_i', '$\\overline{\\RRR}_i$'),
    ('sonr_ar_i',  '$\\overline{\\SoNR \\times \\AR}_i$'),
    ('const',      'Constant'),
]

ds_rows = []
for varname, label in var_map_ds:
    # Spec A cell
    if varname in res_specA.params.index:
        cell_a_coef = _fmt_coef(res_specA, varname)
        cell_a_se = _fmt_se(res_specA, varname)
    else:
        cell_a_coef = ''
        cell_a_se = ''
    # Spec B cell
    if varname in res_specB.params.index:
        cell_b_coef = _fmt_coef(res_specB, varname)
        cell_b_se = _fmt_se(res_specB, varname)
    else:
        cell_b_coef = ''
        cell_b_se = ''
    ds_rows.append(f"  {label} & {cell_a_coef} & {cell_b_coef} \\\\")
    ds_rows.append(f"   & {cell_a_se} & {cell_b_se} \\\\")

n_row_ds = (
    f"  $N$ & {int(res_specA.nobs)} & {int(res_specB.nobs)} \\\\"
)
r2_row_ds = (
    f"  $R^2$ & {res_specA.rsquared:.4f} & {res_specB.rsquared:.4f} \\\\"
)

ds_table_latex = (
    "\\begin{tabular}{lcc}\n"
    "\\toprule\n"
    " & (A) Worst-Year $\\RG_i$ & (B) Max Drawdown$_i$ \\\\\n"
    "\\midrule\n"
    + "\n".join(ds_rows)
    + "\n\\midrule\n"
    + n_row_ds + "\n"
    + r2_row_ds + "\n"
    + "\\bottomrule\n"
    "\\multicolumn{3}{l}{\\textit{Note: OLS with HC1 robust standard errors. "
    "SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}} \\\\\n"
    "\\multicolumn{3}{l}{\\textit{Max drawdown = most negative cumulative sum "
    "of a contiguous run of negative annual RG values.}}\n"
    "\\end{tabular}"
)

ds_table_path = os.path.join(LATEX_TABLE_DIR, 'tab_downside_risk_maxdrawdown.tex')
with open(ds_table_path, 'w', encoding='utf-8') as f:
    f.write(ds_table_latex)
print(f"\n  LaTeX table saved: {ds_table_path}")


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*70}")
print("ALL V3 ANALYSES COMPLETED")
print(f"{'='*70}")
print(f"Figures saved to:      {FIG_DIR}")
print(f"LaTeX tables saved to: {LATEX_TABLE_DIR}")
print()
print("Outputs produced:")
print(f"  [TABLE] tab_sample_construction.tex")
print(f"  [FIGURE] F_histograms_metrics.pdf / .png")
print(f"  [TABLE] tab_uncertainty_regression.tex")
print(f"  [TABLE] tab_downside_risk_maxdrawdown.tex")
