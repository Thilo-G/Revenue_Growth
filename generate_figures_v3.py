"""
generate_figures_v3.py
Publication-quality figure generation for "How Firms Grow With Their Customers"
(Kraft & Skiera, Goethe University Frankfurt)

Generates all 11 main figures + 6 selected appendix figures to figures_v3/.
Copies PDFs to RG_LaTeX/figures/.

Run:  py generate_figures_v3.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from scipy import stats as scipy_stats
import os
import shutil
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')

DATA_FILE1 = os.path.join(
    PROJECT_ROOT, 'RG_Data',
    '2026-02-18-Firm-Year-Revenue-Python.xlsx'
)
DATA_FILE2 = os.path.join(
    PROJECT_ROOT, 'RG_Data',
    'Firms-Industry-2025-11-21-Python.xlsx'
)

FIGURES_V3_DIR = os.path.join(SCRIPT_DIR, 'figures_v3')
LATEX_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'RG_LaTeX', 'figures')

os.makedirs(FIGURES_V3_DIR, exist_ok=True)
os.makedirs(LATEX_FIGURES_DIR, exist_ok=True)

ANALYSIS_END_YEAR = 2025
MIN_FIRMS_PER_INDUSTRY = 10

# Sample note used on every figure (added to caption in LaTeX, not in figure)
SAMPLE_NOTE = "Sample: 500 US firms, 2017-2025, n=3,920 firm-year observations, 16 GICS industries."

# ── Publication style ─────────────────────────────────────────────────────────
# Colorblind-friendly palette (Okabe-Ito, widely used in academic publishing)
CB_BLUE   = '#0072B2'   # acquisition / primary series
CB_ORANGE = '#E69F00'   # secondary series
CB_GREEN  = '#009E73'   # covariance / third series
CB_RED    = '#D55E00'   # median line
CB_PURPLE = '#CC79A7'   # fourth series
CB_GREY   = '#999999'   # neutral / insignificant
CB_LBLUE  = '#56B4E9'   # light blue

PALETTE = {
    'primary':     CB_BLUE,
    'acq':         CB_RED,
    'ret':         CB_BLUE,
    'cov':         CB_PURPLE,
    'median_line': CB_RED,
    'mean_line':   CB_GREEN,
    'secondary':   CB_ORANGE,
    'neutral':     CB_GREY,
}

# Base rcParams — 10pt at journal column width (7 in full, 3.5 in narrow)
BASE_RC = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.color': '#cccccc',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}
plt.rcParams.update(BASE_RC)


def save_fig(fig, name: str) -> None:
    """Save PDF + PNG to figures_v3/ then copy PDF to RG_LaTeX/figures/."""
    pdf_path = os.path.join(FIGURES_V3_DIR, f'{name}.pdf')
    png_path = os.path.join(FIGURES_V3_DIR, f'{name}.png')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    # Copy PDF to LaTeX figures directory
    dest_pdf = os.path.join(LATEX_FIGURES_DIR, f'{name}.pdf')
    shutil.copy2(pdf_path, dest_pdf)
    print(f"  Saved: {name}.pdf  (+ PNG, + copy to RG_LaTeX/figures/)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0: DATA LOADING (mirrors revenue_growth_presentation_v4.py pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _norm_id(x) -> str:
    s = str(x).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _base_name(col_name: str) -> str:
    return (
        str(col_name)
        .replace('#New_Customers', '').replace('#New_Customer', '')
        .replace('#Returning_Customers', '').replace('#Returning_Customer', '')
        .lstrip('#').strip('. ')
    )


def load_data():
    print("Loading data...")
    df_revenue = pd.read_excel(DATA_FILE1, header=None, engine='openpyxl')
    df_industry = pd.read_excel(DATA_FILE2, header=0, engine='openpyxl')
    df_industry.columns = df_industry.columns.str.strip()

    # Industry filtering: use 6-digit GICS industry name with >=10 firms
    df_industry['gics_industry_name'] = (
        df_industry['gics_industry_name']
        .astype(str).str.strip()
        .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    )
    df_industry = df_industry.dropna(subset=['gics_industry_name']).copy()
    ind_counts = df_industry.groupby('gics_industry_name')['ID'].nunique()
    keep_inds = ind_counts[ind_counts >= MIN_FIRMS_PER_INDUSTRY].index
    df_industry = df_industry[df_industry['gics_industry_name'].isin(keep_inds)].copy()

    valid_ids_norm = set(_norm_id(x) for x in df_industry["ID"].dropna())

    # Filter revenue columns
    max_header_row = min(2, len(df_revenue.index))
    cols_to_keep = [0]
    for col_idx in df_revenue.columns:
        if col_idx == 0:
            continue
        candidates = [_norm_id(df_revenue.iloc[r, col_idx])
                      for r in range(max_header_row)
                      if pd.notna(df_revenue.iloc[r, col_idx])]
        if any(h in valid_ids_norm for h in candidates):
            cols_to_keep.append(col_idx)

    if len(cols_to_keep) == 1:
        cols_to_keep = [0] + [
            c for c in df_revenue.columns[1:]
            if any(pd.notna(df_revenue.iloc[r, c]) and str(df_revenue.iloc[r, c]).strip() != ""
                   for r in range(max_header_row))
        ]

    df_revenue = df_revenue.iloc[:, cols_to_keep].copy()

    # Header processing
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

    # Identify New/Returning columns
    new_customer_cols = [c for c in df_revenue.columns if '#New_Customers' in c or '#New_Customer' in c]
    returning_customer_cols = [c for c in df_revenue.columns if '#Returning_Customers' in c or '#Returning_Customer' in c]

    # Drop firms with missing values
    candidate_firms = sorted({_base_name(c) for c in (new_customer_cols + returning_customer_cols)})
    firms_keep = []
    for f in candidate_firms:
        new_col = next((c for c in new_customer_cols if _base_name(c) == f), None)
        ret_col = next((c for c in returning_customer_cols if _base_name(c) == f), None)
        if new_col is None or ret_col is None:
            continue
        new_num = pd.to_numeric(df_revenue[new_col], errors='coerce')
        ret_num = pd.to_numeric(df_revenue[ret_col], errors='coerce')
        new_raw = df_revenue[new_col].astype(str).str.strip().str.lower()
        ret_raw = df_revenue[ret_col].astype(str).str.strip().str.lower()
        if not (new_raw.eq('missing value').any() or new_num.isna().any()
                or ret_raw.eq('missing value').any() or ret_num.isna().any()):
            firms_keep.append(f)

    firms_drop = set(candidate_firms) - set(firms_keep)
    if firms_drop:
        cols_drop = [c for c in df_revenue.columns if c != 'Date' and _base_name(c) in firms_drop]
        df_revenue = df_revenue.drop(columns=cols_drop)

    new_customer_cols = [c for c in df_revenue.columns if '#New_Customers' in c or '#New_Customer' in c]
    returning_customer_cols = [c for c in df_revenue.columns if '#Returning_Customers' in c or '#Returning_Customer' in c]

    # Compute derived metrics
    new_map = {_base_name(col): col for col in new_customer_cols}
    new_cols_data = {}
    final_cols = []

    for col in list(df_revenue.columns):
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

            rrr_col = col.replace('#Returning_Customers', '#RRR')
            rg_col = col.replace('#Returning_Customers', '#Revenue_Growth')
            ar_col = new_col.replace('#New_Customers', '#Acq_Rate')
            qt_col = col.replace('#Returning_Customers', '#Q_t')
            sonr_col = new_col.replace('#New_Customers', '#SoNR')
            tot_col = new_col.replace('#New_Customers', '#Total_Revenue')

            new_cols_data[tot_col] = total
            new_cols_data[rrr_col] = ret_s / total_prev_nonzero
            new_cols_data[rg_col] = (total - total.shift(1)) / total_prev_nonzero
            acq_rate = new_s / new_s.shift(1)
            new_cols_data[ar_col] = acq_rate
            new_cols_data[qt_col] = (ret_s / total_prev_nonzero) / acq_rate.replace(0, np.nan)
            sonr = new_s / total_nonzero
            new_cols_data[sonr_col] = sonr
            final_cols.extend([tot_col, rrr_col, rg_col, ar_col, qt_col, sonr_col])

    if new_cols_data:
        df_revenue = pd.concat([df_revenue, pd.DataFrame(new_cols_data, index=df_revenue.index)], axis=1)
    df_revenue = df_revenue.loc[:, [c for c in final_cols if c in df_revenue.columns]]

    # Transform to panel (long) format
    df_revenue['Date'] = pd.to_datetime(df_revenue['Date'], errors='coerce')
    df_revenue = df_revenue.sort_values('Date').reset_index(drop=True)

    metrics_suffixes = ['#Revenue_Growth', '#RRR', '#Acq_Rate', '#Q_t', '#Total_Revenue', '#SoNR']
    metric_cols = [c for c in df_revenue.columns if any(c.endswith(s) for s in metrics_suffixes)]

    df_long = df_revenue.melt(id_vars='Date', value_vars=metric_cols, var_name='col', value_name='value')
    df_long['firm'] = df_long['col'].apply(lambda c: c.split('#')[0] if '#' in c else c)
    df_long['metric'] = df_long['col'].apply(lambda c: c.split('#', 1)[1] if '#' in c else '')

    df_panel = (
        df_long.pivot_table(index=['Date', 'firm'], columns='metric', values='value', aggfunc='first')
        .reset_index()
    )
    df_panel.columns = [c if isinstance(c, str) else c[1] for c in df_panel.columns]

    for col in ['Revenue_Growth', 'RRR', 'Acq_Rate', 'Q_t', 'Total_Revenue', 'SoNR']:
        if col in df_panel.columns:
            df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce')

    df_panel['Date'] = pd.to_datetime(df_panel['Date'], errors='coerce')
    df_panel = df_panel.sort_values(['firm', 'Date']).reset_index(drop=True)
    df_panel['period_year'] = df_panel['Date'].dt.year.astype(float)
    df_panel = df_panel[df_panel['period_year'] <= ANALYSIS_END_YEAR].copy()

    # Attach industry labels
    ind = df_industry[['ID', 'gics_industry_name']].copy()
    ind['id_norm'] = ind['ID'].apply(_norm_id)
    ind = ind.dropna(subset=['id_norm', 'gics_industry_name']).drop_duplicates('id_norm')
    id_to_industry = ind.set_index('id_norm')['gics_industry_name']
    firm_raw = df_panel['firm'].astype(str)
    df_panel['industry'] = firm_raw.apply(_norm_id).map(id_to_industry)
    df_panel['industry'] = df_panel['industry'].fillna(
        firm_raw.str.split('.').str[0].apply(_norm_id).map(id_to_industry)
    )
    df_panel['industry'] = df_panel['industry'].fillna(
        firm_raw.str.split('.').str[-1].apply(_norm_id).map(id_to_industry)
    )

    df_panel = df_panel.dropna(subset=['industry']).copy()

    # Re-apply industry threshold
    ind_firm_counts = df_panel.groupby('industry')['firm'].nunique()
    keep_industries = ind_firm_counts[ind_firm_counts >= MIN_FIRMS_PER_INDUSTRY].index
    df_panel = df_panel[df_panel['industry'].isin(keep_industries)].copy()

    # Winsorize at 1st/99th percentiles
    for metric in ['Acq_Rate', 'RRR', 'Revenue_Growth', 'SoNR']:
        x = pd.to_numeric(df_panel[metric], errors='coerce')
        df_panel.loc[x < x.quantile(0.01), metric] = np.nan
        df_panel.loc[x > x.quantile(0.99), metric] = np.nan

    n_firms = df_panel['firm'].nunique()
    n_industries = df_panel['industry'].nunique()
    n_obs = len(df_panel[['Revenue_Growth', 'Acq_Rate', 'RRR', 'SoNR']].dropna(how='all'))
    print(f"Panel: {n_obs:,} obs, {n_firms} firms, {n_industries} industries")
    print(f"Years: {sorted(df_panel['period_year'].dropna().unique().astype(int))}")
    return df_panel


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_panel = load_data()

# Precompute lagged SoNR for decomposition figures
df_panel = df_panel.sort_values(['firm', 'Date']).copy()
df_panel['SoNR_lag1'] = df_panel.groupby('firm')['SoNR'].shift(1)
df_panel['acq_contribution'] = df_panel['SoNR_lag1'] * df_panel['Acq_Rate']


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE HELPER: formatted percent axis
# ══════════════════════════════════════════════════════════════════════════════

def _pct_formatter(x, _):
    return f'{x:.0f}%'


# ══════════════════════════════════════════════════════════════════════════════
# F1: Decomposition Stacked Bar by Year
# (SoNR_{t-1}×AR + RRR contributions to median 1+RG, by year)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- F1: Decomposition stacked bar ---")

yearly = (
    df_panel[df_panel['period_year'] >= 2019]
    .groupby('period_year')
    .agg(
        acq_med=('acq_contribution', 'median'),
        ret_med=('RRR', 'median'),
        rg_med=('Revenue_Growth', 'median'),
    )
    .dropna()
)
years_f1 = yearly.index.astype(int)

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
x_pos = np.arange(len(years_f1))
width = 0.6

ax1.bar(x_pos, yearly['acq_med'], width,
        label=r'Acquisition contribution ($\mathrm{SoNR}_{t-1} \times \mathrm{AR}_t$)',
        color=PALETTE['acq'], alpha=0.85)
ax1.bar(x_pos, yearly['ret_med'], width,
        bottom=yearly['acq_med'],
        label='Retention contribution ($\mathrm{RRR}_t$)',
        color=PALETTE['ret'], alpha=0.85)
ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
            label='Break-even (1+RG = 1)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(years_f1)
ax1.set_xlabel('Year')
ax1.set_ylabel('Contribution to (1 + Revenue Growth), median')
ax1.set_title(
    'Annual Decomposition of Revenue Growth into Acquisition\n'
    'and Retention Contributions, 500 US Firms, 2019-2025'
)
ax1.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
save_fig(fig1, 'F1_decomposition_stacked_bar')
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# F2: 2x2 Distributions (RG, AR, RRR, SoNR)
# ══════════════════════════════════════════════════════════════════════════════
print("--- F2: 2x2 distributions ---")

fig2, axes2 = plt.subplots(2, 2, figsize=(7, 5.5))

hist_configs = [
    ('Revenue_Growth', 'Revenue Growth (RG, ratio)', axes2[0, 0], 'Firm-year pooled RG (ratio)'),
    ('SoNR',          'Share of New Revenue (SoNR, fraction)', axes2[0, 1], 'Firm-level median SoNR (fraction)'),
    ('Acq_Rate',      'Acquisition Rate (AR, ratio)', axes2[1, 0], 'Firm-year pooled AR (ratio)'),
    ('RRR',           'Revenue Retention Rate (RRR, ratio)', axes2[1, 1], 'Firm-year pooled RRR (ratio)'),
]

for metric, ax_title, ax, xlabel in hist_configs:
    if metric == 'SoNR':
        data = df_panel.groupby('firm')[metric].median().dropna()
    else:
        data = df_panel[metric].dropna()

    median_val = data.median()
    mean_val = data.mean()
    ax.hist(data, bins=45, color=PALETTE['primary'], alpha=0.75,
            edgecolor='white', linewidth=0.4)
    ax.axvline(median_val, color=PALETTE['median_line'], linestyle='-',
               linewidth=1.8, label=f'Median: {median_val:.3f}')
    ax.axvline(mean_val, color=PALETTE['mean_line'], linestyle='--',
               linewidth=1.8, label=f'Mean: {mean_val:.3f}')
    ax.set_title(ax_title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Number of observations', fontsize=9)
    ax.legend(fontsize=8)

fig2.suptitle(
    'Cross-Sectional Distributions of Revenue Growth Metrics,\n'
    '500 US Firms, 2017-2025',
    fontsize=11, y=1.01
)
plt.tight_layout()
save_fig(fig2, 'F2_distributions_2x2')
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# F3: Boxplots of RG by year
# ══════════════════════════════════════════════════════════════════════════════
print("--- F3: Boxplots RG by year ---")

rg_by_year = [
    df_panel.loc[df_panel['period_year'] == yr, 'Revenue_Growth'].dropna().values
    for yr in sorted(df_panel['period_year'].dropna().unique().astype(int))
    if yr >= 2018
]
year_labels_f3 = [
    int(yr) for yr in sorted(df_panel['period_year'].dropna().unique().astype(int))
    if yr >= 2018
]

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
bp = ax3.boxplot(rg_by_year, labels=year_labels_f3,
                 patch_artist=True, notch=False,
                 medianprops=dict(color=PALETTE['median_line'], linewidth=2),
                 boxprops=dict(facecolor=CB_LBLUE, alpha=0.6, color=CB_BLUE),
                 whiskerprops=dict(color=CB_BLUE),
                 capprops=dict(color=CB_BLUE),
                 flierprops=dict(marker='o', markerfacecolor=CB_GREY,
                                 markersize=2, alpha=0.4, linestyle='none'))
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_xlabel('Year')
ax3.set_ylabel('Revenue growth (ratio, annual firm-year)')
ax3.set_title(
    'Annual Distribution of Revenue Growth Across 500 US Firms, 2018-2025\n'
    '(Boxplots: IQR and whiskers at 1.5 x IQR; median shown in red)'
)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.tight_layout()
save_fig(fig3, 'F3_boxplots_rg_by_year')
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# F4: Industry-Adjusted AR vs RRR Scatter with OLS line
# ══════════════════════════════════════════════════════════════════════════════
print("--- F4: Adjusted scatter AR vs RRR ---")

for metric in ['Acq_Rate', 'RRR']:
    ind_med = df_panel.groupby(['industry', 'period_year'])[metric].transform('median')
    df_panel[f'adj_{metric}'] = df_panel[metric] - ind_med

firm_adj = df_panel.groupby('firm').agg(
    adj_AR=('adj_Acq_Rate', 'median'),
    adj_RRR=('adj_RRR', 'median'),
).dropna()

adj_corr = firm_adj['adj_AR'].corr(firm_adj['adj_RRR'])

# OLS line
X_adj = sm.add_constant(firm_adj['adj_AR'].values)
res_adj = sm.OLS(firm_adj['adj_RRR'].values, X_adj).fit()

fig4, ax4 = plt.subplots(figsize=(5, 4.5))
ax4.scatter(firm_adj['adj_AR'], firm_adj['adj_RRR'],
            alpha=0.35, s=18, color=PALETTE['primary'],
            label=f'Firm median (n = {len(firm_adj):,})')
xr = np.linspace(firm_adj['adj_AR'].min(), firm_adj['adj_AR'].max(), 200)
ax4.plot(xr, res_adj.params[0] + res_adj.params[1] * xr,
         color=PALETTE['median_line'], linewidth=2, label='OLS fit')
ax4.axhline(0, color=CB_GREY, linewidth=0.6)
ax4.axvline(0, color=CB_GREY, linewidth=0.6)
ax4.set_xlabel('Industry-adjusted Acquisition Rate (AR, deviation from industry-year median)')
ax4.set_ylabel('Industry-adjusted Revenue Retention Rate (RRR, deviation from industry-year median)')
ax4.set_title(
    f'Firm-Level AR and RRR After Removing Industry-Year Fixed Effects\n'
    f'Pearson r = {adj_corr:.3f}, 500 US Firms, 2017-2025'
)
ax4.legend(fontsize=8)
plt.tight_layout()
save_fig(fig4, 'F4_adj_scatter_ols')
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
# F5: Histogram of within-firm AR-RRR correlations
# ══════════════════════════════════════════════════════════════════════════════
print("--- F5: Within-firm AR-RRR correlation histogram ---")

firm_corrs = (
    df_panel.dropna(subset=['Acq_Rate', 'RRR'])
    .groupby('firm')
    .apply(lambda g: g['Acq_Rate'].corr(g['RRR']) if len(g) >= 3 else np.nan)
    .dropna()
)

pct_positive = (firm_corrs > 0).mean() * 100
median_corr = firm_corrs.median()
mean_corr = firm_corrs.mean()

fig5, ax5 = plt.subplots(figsize=(5, 4))
ax5.hist(firm_corrs, bins=30, color=PALETTE['primary'], alpha=0.8,
         edgecolor='white', linewidth=0.4)
ax5.axvline(median_corr, color=PALETTE['median_line'], linewidth=2,
            linestyle='-', label=f'Median: {median_corr:.3f}')
ax5.axvline(mean_corr, color=PALETTE['mean_line'], linewidth=2,
            linestyle='--', label=f'Mean: {mean_corr:.3f}')
ax5.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
ax5.set_xlabel('Within-firm Pearson correlation between AR and RRR (time-series)')
ax5.set_ylabel('Number of firms')
ax5.set_title(
    f'Distribution of Within-Firm AR-RRR Correlations Across 500 US Firms\n'
    f'{pct_positive:.1f}% of firms show positive correlation'
)
ax5.legend(fontsize=8)
plt.tight_layout()
save_fig(fig5, 'F5_firm_adj_correlation_hist')
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# F6: Long-run convergence classification
# ══════════════════════════════════════════════════════════════════════════════
print("--- F6: Long-run convergence classification ---")

if 'Q_t' in df_panel.columns:
    q_df = df_panel[['firm', 'Q_t']].dropna(subset=['Q_t']).copy()
    q_df['Q_t'] = pd.to_numeric(q_df['Q_t'], errors='coerce')
    q_df = q_df.dropna()
    by_firm = q_df.groupby('firm')['Q_t']
    n_total_q = q_df['firm'].nunique()
    n_acq = by_firm.apply(lambda s: (s < 1).all()).sum()
    n_ret = by_firm.apply(lambda s: (s > 1).all()).sum()
    n_mixed = n_total_q - n_acq - n_ret
else:
    # Fallback: use median AR vs median RRR comparison
    lr_medians = df_panel.groupby('firm')[['Acq_Rate', 'RRR']].median().dropna()
    n_total_q = len(lr_medians)
    n_acq = (lr_medians['Acq_Rate'] > lr_medians['RRR']).sum()
    n_ret = (lr_medians['Acq_Rate'] < lr_medians['RRR']).sum()
    n_mixed = n_total_q - n_acq - n_ret

fig6, ax6 = plt.subplots(figsize=(5, 4))
categories_f6 = [
    f'Acquisition-driven\n(AR > RRR always)\n{n_acq} firms ({n_acq/n_total_q:.1%})',
    f'Retention-driven\n(RRR > AR always)\n{n_ret} firms ({n_ret/n_total_q:.1%})',
    f'Mixed or unclear\n{n_mixed} firms ({n_mixed/n_total_q:.1%})',
]
values_f6 = [n_acq, n_ret, n_mixed]
colors_f6 = [PALETTE['acq'], PALETTE['ret'], CB_GREY]

bars6 = ax6.barh(categories_f6, values_f6, color=colors_f6, alpha=0.85,
                 edgecolor='white', height=0.6)
for bar, val in zip(bars6, values_f6):
    ax6.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
             str(val), va='center', ha='left', fontsize=9)

ax6.set_xlabel('Number of firms')
ax6.set_ylabel('')
ax6.set_title(
    'Long-Run Growth Strategy Classification of 500 US Firms\n'
    '(Based on time-series dominance of AR vs. RRR, 2017-2025)'
)
ax6.set_xlim(0, max(values_f6) * 1.18)
ax6.invert_yaxis()
plt.tight_layout()
save_fig(fig6, 'F6_longrun_classification')
plt.close(fig6)


# ══════════════════════════════════════════════════════════════════════════════
# F7: Industry lollipop — median RG by GICS industry
# ══════════════════════════════════════════════════════════════════════════════
print("--- F7: Industry lollipop RG ---")

ind_medians_f7 = (
    df_panel.groupby('industry')['Revenue_Growth'].median()
    .dropna()
    .sort_values()
)

fig7, ax7 = plt.subplots(figsize=(7, max(4, len(ind_medians_f7) * 0.38)))
y_pos = np.arange(len(ind_medians_f7))
colors_f7 = [PALETTE['ret'] if v >= 0 else PALETTE['acq'] for v in ind_medians_f7]

ax7.hlines(y_pos, 0, ind_medians_f7.values, colors=CB_GREY, linewidth=1.5, alpha=0.6)
ax7.scatter(ind_medians_f7.values, y_pos, s=60, c=colors_f7, zorder=3, alpha=0.9)
ax7.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax7.axvline(ind_medians_f7.median(), color=PALETTE['median_line'], linewidth=1.5,
            linestyle=':', alpha=0.7, label=f'Median across industries: {ind_medians_f7.median():.3f}')

ax7.set_yticks(y_pos)
ax7.set_yticklabels(ind_medians_f7.index, fontsize=8)
ax7.set_xlabel('Median annual revenue growth (ratio)')
ax7.set_title(
    'Median Revenue Growth by GICS Industry, 500 US Firms, 2017-2025\n'
    '(Blue = positive median; red = negative median)'
)
ax7.legend(fontsize=8)
plt.tight_layout()
save_fig(fig7, 'F7_lollipop_industries_rg')
plt.close(fig7)


# ══════════════════════════════════════════════════════════════════════════════
# F8: Quadrant boxplot — RG by AR x RRR quadrant
# ══════════════════════════════════════════════════════════════════════════════
print("--- F8: Quadrant boxplot ---")

firm_medians_f8 = df_panel.groupby('firm')[['Acq_Rate', 'RRR', 'Revenue_Growth']].median().dropna()
ar_cut = firm_medians_f8['Acq_Rate'].median()
rrr_cut = firm_medians_f8['RRR'].median()

firm_medians_f8['AR_high'] = firm_medians_f8['Acq_Rate'] >= ar_cut
firm_medians_f8['RRR_high'] = firm_medians_f8['RRR'] >= rrr_cut

quadrant_labels_f8 = ['HH\n(High AR, High RRR)', 'HL\n(High AR, Low RRR)',
                       'LH\n(Low AR, High RRR)', 'LL\n(Low AR, Low RRR)']
quadrant_data_f8 = []
for ar_h, rrr_h in [(True, True), (True, False), (False, True), (False, False)]:
    mask = (firm_medians_f8['AR_high'] == ar_h) & (firm_medians_f8['RRR_high'] == rrr_h)
    quadrant_data_f8.append(firm_medians_f8.loc[mask, 'Revenue_Growth'].dropna().values)

colors_f8 = [PALETTE['primary'], PALETTE['acq'], PALETTE['ret'], CB_GREY]

fig8, ax8 = plt.subplots(figsize=(6, 4.5))
bp8 = ax8.boxplot(quadrant_data_f8, labels=quadrant_labels_f8,
                  patch_artist=True, notch=False,
                  medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp8['boxes'], colors_f8):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Annotate with N per quadrant
for i, data in enumerate(quadrant_data_f8):
    ax8.text(i + 1, ax8.get_ylim()[0] * 0.98,
             f'n={len(data)}', ha='center', va='top', fontsize=8)

ax8.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax8.set_ylabel('Firm-median revenue growth (ratio)')
ax8.set_xlabel('Quadrant (firm classified by median AR and RRR vs. cross-firm medians)')
ax8.set_title(
    'Revenue Growth Distribution by AR-RRR Quadrant,\n'
    '500 US Firms, 2017-2025 (Firm-level medians)'
)
plt.tight_layout()
save_fig(fig8, 'F8_quadrant_boxplot')
plt.close(fig8)


# ══════════════════════════════════════════════════════════════════════════════
# F9: Per-SD contribution bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("--- F9: Per-SD contribution ---")

sonr_median = df_panel['SoNR'].median()
ar_sd = df_panel['Acq_Rate'].std()
rrr_sd = df_panel['RRR'].std()

# Identity: dRG/dAR = SoNR_{t-1}; dRG/dRRR = 1
delta_rg_ar = sonr_median * ar_sd   # scaled contribution from AR
delta_rg_rrr = rrr_sd               # contribution from RRR

fig9, ax9 = plt.subplots(figsize=(4.5, 4))
bar_labels_f9 = [
    f'Acquisition Rate (AR)\nSD = {ar_sd:.3f}, Marginal weight = SoNR = {sonr_median:.3f}',
    f'Revenue Retention Rate (RRR)\nSD = {rrr_sd:.3f}, Marginal weight = 1.000',
]
bar_vals_f9 = [delta_rg_ar * 100, delta_rg_rrr * 100]
bar_colors_f9 = [PALETTE['acq'], PALETTE['ret']]

bars9 = ax9.bar(bar_labels_f9, bar_vals_f9, color=bar_colors_f9,
                alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars9, bar_vals_f9):
    ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             f'+{val:.1f} pp', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax9.set_ylabel('Change in revenue growth (percentage points)')
ax9.set_ylim(0, max(bar_vals_f9) * 1.3)
ax9.set_title(
    'Effect of a One-Standard-Deviation Increase in AR or RRR\n'
    'on Revenue Growth, 500 US Firms, 2017-2025'
)
plt.tight_layout()
save_fig(fig9, 'F9_sd_contribution')
plt.close(fig9)


# ══════════════════════════════════════════════════════════════════════════════
# F10: Downside risk scatter (mean RRR vs worst-year RG)
# ══════════════════════════════════════════════════════════════════════════════
print("--- F10: Downside risk scatter ---")

firm_risk = df_panel.groupby('firm').agg(
    min_RG=('Revenue_Growth', 'min'),
    mean_RRR=('RRR', 'mean'),
    mean_AR=('Acq_Rate', 'mean'),
    n_obs=('Revenue_Growth', 'count'),
).reset_index().dropna()
firm_risk = firm_risk[firm_risk['n_obs'] >= 3].copy()

X10 = sm.add_constant(firm_risk[['mean_RRR', 'mean_AR']])
res10 = sm.OLS(firm_risk['min_RG'], X10).fit(cov_type='HC1')

rrr_range_f10 = np.linspace(firm_risk['mean_RRR'].min(), firm_risk['mean_RRR'].max(), 200)
mean_ar_val = firm_risk['mean_AR'].mean()
ols_line_f10 = (res10.params['const']
                + res10.params['mean_RRR'] * rrr_range_f10
                + res10.params['mean_AR'] * mean_ar_val)

fig10, ax10 = plt.subplots(figsize=(5.5, 4.5))
ax10.scatter(firm_risk['mean_RRR'], firm_risk['min_RG'],
             alpha=0.35, s=18, color=PALETTE['ret'],
             label=f'Firm (n = {len(firm_risk):,})')
ax10.plot(rrr_range_f10, ols_line_f10,
          color=PALETTE['median_line'], linewidth=2,
          label=f'OLS line (holding mean AR = {mean_ar_val:.2f})')
ax10.axhline(0, color=CB_GREY, linewidth=0.8, linestyle='--', alpha=0.6)

beta_rrr = res10.params['mean_RRR']
pval_rrr = res10.pvalues['mean_RRR']
ax10.set_xlabel('Mean Revenue Retention Rate (RRR, firm time-series average)')
ax10.set_ylabel('Worst-year Revenue Growth (min annual RG per firm)')
ax10.set_title(
    f'Downside Protection: Mean RRR and Worst-Year Revenue Growth\n'
    f'OLS: beta(RRR) = {beta_rrr:+.3f}, p = {pval_rrr:.3f}, N = {len(firm_risk):,} firms'
)
ax10.legend(fontsize=8)
plt.tight_layout()
save_fig(fig10, 'F10_downside_risk_scatter')
plt.close(fig10)


# ══════════════════════════════════════════════════════════════════════════════
# F_histograms_metrics: 2x2 histograms with RRR=1 and SoNR=0.5 reference lines
# ══════════════════════════════════════════════════════════════════════════════
print("--- F_histograms_metrics ---")

rg_vals = df_panel['Revenue_Growth'].dropna()
ar_vals = df_panel['Acq_Rate'].dropna()
rrr_vals = df_panel['RRR'].dropna()
sonr_vals = df_panel['SoNR'].dropna()
pct_rrr_below_1 = (rrr_vals < 1).mean() * 100

fig_hist, axes_hist = plt.subplots(2, 2, figsize=(7, 5.5))

def _draw_hist(ax, data, title, xlabel, vline_val=None, vline_label=None):
    median_val = data.median()
    mean_val = data.mean()
    ax.hist(data, bins=40, color=PALETTE['primary'], alpha=0.75,
            edgecolor='white', linewidth=0.4)
    ax.axvline(median_val, color=PALETTE['median_line'], linewidth=1.8,
               linestyle='-', label=f'Median = {median_val:.3f}')
    ax.axvline(mean_val, color=PALETTE['mean_line'], linewidth=1.8,
               linestyle='--', label=f'Mean = {mean_val:.3f}')
    if vline_val is not None:
        ax.axvline(vline_val, color='black', linewidth=1.2,
                   linestyle=':', label=vline_label)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.legend(fontsize=8)

_draw_hist(axes_hist[0, 0], rg_vals,
           'Revenue Growth (RG)', 'RG (pooled firm-year, ratio)')
_draw_hist(axes_hist[0, 1], ar_vals,
           'Acquisition Rate (AR)', 'AR (pooled firm-year, ratio)')
_draw_hist(axes_hist[1, 0], rrr_vals,
           'Revenue Retention Rate (RRR)', 'RRR (pooled firm-year, ratio)',
           vline_val=1.0, vline_label='RRR = 1')
axes_hist[1, 0].text(0.97, 0.95,
                      f'{pct_rrr_below_1:.0f}% of obs. have RRR < 1',
                      transform=axes_hist[1, 0].transAxes, fontsize=8,
                      va='top', ha='right',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
_draw_hist(axes_hist[1, 1], sonr_vals,
           'Share of New Revenue (SoNR)', 'SoNR (pooled firm-year, fraction)',
           vline_val=0.5, vline_label='SoNR = 0.5')

fig_hist.suptitle(
    'Distributions of Revenue Growth Metrics Across 500 US Firms, 2017-2025\n'
    '(Pooled firm-year observations, winsorized at 1st/99th percentile)',
    fontsize=11, y=1.02
)
plt.tight_layout()
save_fig(fig_hist, 'F_histograms_metrics')
plt.close(fig_hist)


# ══════════════════════════════════════════════════════════════════════════════
# APPENDIX FIGURES (selected 6)
# ══════════════════════════════════════════════════════════════════════════════

# ── IA_F1: Variance decomposition stacked bar (Acquisition, RRR, Covariance)
# Justification: The main text states variance shares (RRR 37%, SoNR×AR 34%,
# Cov 29%) but no main figure shows the decomposition explicitly.
print("--- IA_F1: Variance decomposition stacked bar ---")

firm_decomp_rows = []
for firm_id, grp in df_panel.groupby('firm'):
    fd = grp[['acq_contribution', 'RRR', 'Revenue_Growth']].dropna()
    if len(fd) < 3:
        continue
    v_rg = fd['Revenue_Growth'].var(ddof=1)
    if v_rg <= 0:
        continue
    v_a = fd['acq_contribution'].var(ddof=1)
    v_r = fd['RRR'].var(ddof=1)
    cov_ar = fd['acq_contribution'].cov(fd['RRR'])
    firm_decomp_rows.append({
        'share_A': v_a / v_rg,
        'share_RRR': v_r / v_rg,
        'share_cov': 2 * cov_ar / v_rg,
    })

df_decomp = pd.DataFrame(firm_decomp_rows)
med_a = df_decomp['share_A'].median()
med_r = df_decomp['share_RRR'].median()
med_c = df_decomp['share_cov'].median()

# Pooled-within estimate
df_demeaned = df_panel.copy()
for col in ['acq_contribution', 'RRR', 'Revenue_Growth']:
    df_demeaned[col] = df_demeaned.groupby('firm')[col].transform(lambda x: x - x.mean())
pooled = df_demeaned[['acq_contribution', 'RRR', 'Revenue_Growth']].dropna()
pv_rg = pooled['Revenue_Growth'].var(ddof=1)
ps_a = pooled['acq_contribution'].var(ddof=1) / pv_rg
ps_r = pooled['RRR'].var(ddof=1) / pv_rg
ps_c = 2 * pooled['acq_contribution'].cov(pooled['RRR']) / pv_rg

fig_ia1, ax_ia1 = plt.subplots(figsize=(5, 4))
bar_labels_ia1 = ['Pooled-Within', 'Firm-Level Median']
acq_vals_ia1 = [ps_a * 100, med_a * 100]
ret_vals_ia1 = [ps_r * 100, med_r * 100]
cov_vals_ia1 = [ps_c * 100, med_c * 100]
x_ia1 = np.arange(len(bar_labels_ia1))
width_ia1 = 0.5

ax_ia1.bar(x_ia1, acq_vals_ia1, width_ia1,
           label=r'Acquisition ($\mathrm{SoNR}_{t-1} \times \mathrm{AR}_t$)',
           color=PALETTE['acq'], alpha=0.85)
ax_ia1.bar(x_ia1, ret_vals_ia1, width_ia1, bottom=acq_vals_ia1,
           label='Retention (RRR)', color=PALETTE['ret'], alpha=0.85)
cov_bottom = [a + r for a, r in zip(acq_vals_ia1, ret_vals_ia1)]
ax_ia1.bar(x_ia1, cov_vals_ia1, width_ia1, bottom=cov_bottom,
           label='Covariance (2 x cov)', color=PALETTE['cov'], alpha=0.85)

ax_ia1.set_xticks(x_ia1)
ax_ia1.set_xticklabels(bar_labels_ia1)
ax_ia1.set_ylabel('Share of within-firm variance in revenue growth (%)')
ax_ia1.set_title(
    'Within-Firm Revenue Growth Variance Decomposed into\n'
    'Acquisition, Retention, and Covariance Components, 500 US Firms'
)
ax_ia1.legend(loc='upper right', fontsize=8)
ax_ia1.set_ylim(0, 115)
plt.tight_layout()
save_fig(fig_ia1, 'IA_F1_variance_decomposition')
plt.close(fig_ia1)


# ── IA_F2: Rumelt variance decomposition grouped bar (AR vs RRR)
# Justification: Core identification result (firm effects dominate for RRR,
# residual for AR) referenced in H5; no main figure visualises this directly.
print("--- IA_F2: Rumelt grouped bar ---")

# Use published statistics from numbers_pack (do not re-run bootstrap here)
rumelt_components = ['Industry', 'Firm', 'Year', 'Ind x Year', 'Residual']
ar_pcts  = [0.0, 4.5, 19.1, 0.0, 76.6]   # from numbers_pack
rrr_pcts = [22.0, 47.3, 6.6, 0.0, 24.1]  # from numbers_pack

x_ia2 = np.arange(len(rumelt_components))
width_ia2 = 0.35

fig_ia2, ax_ia2 = plt.subplots(figsize=(7, 4.5))
bars_ar  = ax_ia2.bar(x_ia2 - width_ia2/2, ar_pcts, width_ia2,
                      label='Acquisition Rate (AR)', color=PALETTE['acq'], alpha=0.85)
bars_rrr = ax_ia2.bar(x_ia2 + width_ia2/2, rrr_pcts, width_ia2,
                      label='Revenue Retention Rate (RRR)', color=PALETTE['ret'], alpha=0.85)
ax_ia2.set_xticks(x_ia2)
ax_ia2.set_xticklabels(rumelt_components)
ax_ia2.set_ylabel('Share of total variance (%)')
ax_ia2.set_xlabel('Variance component (Henderson Type III decomposition)')
ax_ia2.set_title(
    'Rumelt Variance Decomposition: Sources of Variation in AR and RRR\n'
    'Across 500 US Firms, 2017-2025 (Henderson Type III)'
)
ax_ia2.legend(fontsize=9)
for bar in list(bars_ar) + list(bars_rrr):
    h = bar.get_height()
    if h > 2:
        ax_ia2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
save_fig(fig_ia2, 'IA_F2_rumelt_decomposition')
plt.close(fig_ia2)


# ── IA_F3: Year-by-year boxplots of AR
# Justification: Allows readers to verify that AR trends (COVID spike, recovery)
# are symmetric to the RG boxplots shown in F3, supporting the decomposition claim.
print("--- IA_F3: Boxplots AR by year ---")

ar_by_year = [
    df_panel.loc[df_panel['period_year'] == yr, 'Acq_Rate'].dropna().values
    for yr in sorted(df_panel['period_year'].dropna().unique().astype(int))
    if yr >= 2018
]
year_labels_ia3 = [
    int(yr) for yr in sorted(df_panel['period_year'].dropna().unique().astype(int))
    if yr >= 2018
]

fig_ia3, ax_ia3 = plt.subplots(figsize=(7, 4))
ax_ia3.boxplot(ar_by_year, labels=year_labels_ia3,
               patch_artist=True, notch=False,
               medianprops=dict(color=PALETTE['median_line'], linewidth=2),
               boxprops=dict(facecolor=CB_LBLUE, alpha=0.6, color=CB_BLUE),
               whiskerprops=dict(color=CB_BLUE), capprops=dict(color=CB_BLUE),
               flierprops=dict(marker='o', markerfacecolor=CB_GREY,
                               markersize=2, alpha=0.4, linestyle='none'))
ax_ia3.axhline(1, color=CB_GREY, linewidth=0.8, linestyle='--', alpha=0.5,
               label='AR = 1 (no change in new-customer revenue)')
ax_ia3.set_xlabel('Year')
ax_ia3.set_ylabel('Acquisition rate (AR, ratio: new revenue_t / new revenue_{t-1})')
ax_ia3.set_title(
    'Annual Distribution of Acquisition Rate (AR) Across 500 US Firms, 2018-2025\n'
    '(Boxplots: IQR and whiskers at 1.5 x IQR; median shown in red)'
)
ax_ia3.legend(fontsize=8)
plt.tight_layout()
save_fig(fig_ia3, 'IA_F3_boxplots_AR_year')
plt.close(fig_ia3)


# ── IA_F4: Year-by-year boxplots of RRR
# Justification: Complements F3 (RG boxplots) and IA_F3 (AR boxplots) to give
# a complete time-series view of all three metrics for diagnostics.
print("--- IA_F4: Boxplots RRR by year ---")

rrr_by_year = [
    df_panel.loc[df_panel['period_year'] == yr, 'RRR'].dropna().values
    for yr in sorted(df_panel['period_year'].dropna().unique().astype(int))
    if yr >= 2018
]

fig_ia4, ax_ia4 = plt.subplots(figsize=(7, 4))
ax_ia4.boxplot(rrr_by_year, labels=year_labels_ia3,
               patch_artist=True, notch=False,
               medianprops=dict(color=PALETTE['median_line'], linewidth=2),
               boxprops=dict(facecolor=CB_LBLUE, alpha=0.6, color=CB_BLUE),
               whiskerprops=dict(color=CB_BLUE), capprops=dict(color=CB_BLUE),
               flierprops=dict(marker='o', markerfacecolor=CB_GREY,
                               markersize=2, alpha=0.4, linestyle='none'))
ax_ia4.axhline(1, color=CB_GREY, linewidth=0.8, linestyle='--', alpha=0.5,
               label='RRR = 1 (full revenue retention from existing customers)')
ax_ia4.set_xlabel('Year')
ax_ia4.set_ylabel('Revenue retention rate (RRR, ratio: returning revenue_t / total revenue_{t-1})')
ax_ia4.set_title(
    'Annual Distribution of Revenue Retention Rate (RRR) Across 500 US Firms, 2018-2025\n'
    '(Boxplots: IQR and whiskers at 1.5 x IQR; median shown in red)'
)
ax_ia4.legend(fontsize=8)
plt.tight_layout()
save_fig(fig_ia4, 'IA_F4_boxplots_RRR_year')
plt.close(fig_ia4)


# ── IA_F5: Industry-level lollipops for AR and RRR (side by side)
# Justification: Referees will ask whether industry differences in AR and RRR
# parallel the industry differences in RG (F7); this directly answers that.
print("--- IA_F5: Industry lollipops AR and RRR ---")

ind_med_ar = df_panel.groupby('industry')['Acq_Rate'].median().dropna().sort_values()
ind_med_rrr = df_panel.groupby('industry')['RRR'].median().dropna().sort_values()

fig_ia5, (ax_ar5, ax_rrr5) = plt.subplots(1, 2, figsize=(10, max(4, len(ind_med_ar) * 0.38)))

for ax, ind_med, metric_label, color in [
    (ax_ar5, ind_med_ar, 'Acquisition Rate (AR)', PALETTE['acq']),
    (ax_rrr5, ind_med_rrr, 'Revenue Retention Rate (RRR)', PALETTE['ret']),
]:
    y_pos = np.arange(len(ind_med))
    ax.hlines(y_pos, 0, ind_med.values, colors=CB_GREY, linewidth=1.5, alpha=0.6)
    ax.scatter(ind_med.values, y_pos, s=55, c=color, zorder=3, alpha=0.9)
    ax.axvline(ind_med.median(), color=PALETTE['median_line'], linewidth=1.5,
               linestyle=':', alpha=0.7,
               label=f'Median: {ind_med.median():.3f}')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ind_med.index, fontsize=7.5)
    ax.set_xlabel(f'Median {metric_label} (ratio)')
    ax.set_title(f'Median {metric_label}\nby GICS Industry')
    ax.legend(fontsize=8)

fig_ia5.suptitle(
    'Industry-Level Medians of AR and RRR, 500 US Firms, 2017-2025',
    fontsize=11, y=1.01
)
plt.tight_layout()
save_fig(fig_ia5, 'IA_F5_lollipop_AR_RRR')
plt.close(fig_ia5)


# ── IA_F6: Persistence scatter (2019-2021 vs 2022-2024 firm means for AR and RRR)
# Justification: Directly visualises the between-period stability claim (H5) that
# firm-level RRR is more stable than AR across time; a referee benchmark.
print("--- IA_F6: Persistence scatter ---")

half1 = df_panel[df_panel['period_year'].isin([2019, 2020, 2021])]
half2 = df_panel[df_panel['period_year'].isin([2022, 2023, 2024])]

fig_ia6, (ax_ar6, ax_rrr6) = plt.subplots(1, 2, figsize=(9, 4.5))

for ax, metric, label, color in [
    (ax_ar6, 'Acq_Rate', 'Acquisition Rate (AR)', PALETTE['acq']),
    (ax_rrr6, 'RRR', 'Revenue Retention Rate (RRR)', PALETTE['ret']),
]:
    m1 = half1.groupby('firm')[metric].mean()
    m2 = half2.groupby('firm')[metric].mean()
    common = m1.index.intersection(m2.index)
    bp_corr = m1.loc[common].corr(m2.loc[common])

    ax.scatter(m1.loc[common], m2.loc[common],
               alpha=0.35, s=15, color=color)
    lim_min = min(m1.loc[common].min(), m2.loc[common].min())
    lim_max = max(m1.loc[common].max(), m2.loc[common].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', alpha=0.35, linewidth=1.2, label='45-degree line')
    ax.set_xlabel(f'Mean {label}, 2019-2021 (ratio)')
    ax.set_ylabel(f'Mean {label}, 2022-2024 (ratio)')
    ax.set_title(f'{label}\nBetween-period Pearson r = {bp_corr:.3f}')
    ax.legend(fontsize=8)

fig_ia6.suptitle(
    'Persistence of AR and RRR Across Periods (2019-2021 vs. 2022-2024),\n'
    '500 US Firms, 2017-2025 (Firm-level time-series means)',
    fontsize=11, y=1.02
)
plt.tight_layout()
save_fig(fig_ia6, 'IA_F6_persistence_scatter')
plt.close(fig_ia6)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION: list all generated files
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERIFICATION: Files generated in figures_v3/")
print("=" * 70)

generated = sorted(f for f in os.listdir(FIGURES_V3_DIR) if f.endswith('.pdf'))
for fname in generated:
    fpath = os.path.join(FIGURES_V3_DIR, fname)
    fsize = os.path.getsize(fpath)
    print(f"  {fname}  ({fsize:,} bytes)")

print(f"\nTotal PDFs: {len(generated)}")

print("\n" + "=" * 70)
print("Files copied to RG_LaTeX/figures/")
print("=" * 70)
latex_pdfs = sorted(f for f in os.listdir(LATEX_FIGURES_DIR) if f.endswith('.pdf'))
for fname in latex_pdfs:
    print(f"  {fname}")

print("\nDONE.")
