"""
permeability models: naive vs. sub-volume-aware validation

usage:
    python permeability_models.py data.xlsx            # full run
    python permeability_models.py data.xlsx --quick    # tiny grids, for a fast test

needs: pandas, openpyxl, numpy, scikit-learn, matplotlib, xgboost (falls back to sklearn's
HistGradientBoostingRegressor if xgboost is not installed)

validation modes
  naive          raw data (duplicates kept), random 80/20 split
  dedup_random   exact duplicate rows removed, random 80/20 split
  grouped        duplicates removed, split so that all rows of a sub-volume
                 (rock_type + casename) stay on the same side
tuning: GridSearchCV, 3-fold CV on the training part only
(GroupKFold for grouped mode, KFold otherwise). Metrics are on log10(k). Five repeated splits,
mean +/- SD reported.
"""
import sys, os, json, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, KFold, GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
try:
    from xgboost import XGBRegressor; HAVE_XGB = True
except Exception:
    HAVE_XGB = False
BOOST = 'XGBoost' if HAVE_XGB else 'HistGradientBoosting (XGBoost not installed)'
QUICK = '--quick' in sys.argv
N_SPLITS = 2 if QUICK else 5
JOBS = -1
OUT = 'extended_out'; os.makedirs(OUT, exist_ok=True)

# ---------------- data ----------------
raw = pd.read_excel(sys.argv[1])
raw.columns = ['casename', 'porosity', 'eff_porosity', 'rock_type', 'AR', 'DOA', 'k']
def prep(df):
    df = df.copy()
    df['y'] = np.log10(df.k)                 # target: log10 permeability
    df['lpor'] = np.log10(df.porosity)       # log10 porosity
    df['leff'] = np.log10(df.eff_porosity)
    df['group'] = df.rock_type.astype(str) + '_' + df.casename.astype(str)   # one sub-volume
    return df
raw = prep(raw)
dedup = raw.drop_duplicates(subset=['casename', 'porosity', 'eff_porosity', 'rock_type', 'AR', 'DOA', 'k']).reset_index(drop=True)
print(f'rows: {len(raw)} raw -> {len(dedup)} after removing exact duplicate rows')
print(f'rock type 3: {int((raw.rock_type==3).sum())} raw rows -> {int((dedup.rock_type==3).sum())} unique rows, '
      f'{dedup[dedup.rock_type==3].group.nunique()} sub-volumes')
out_clean = sys.argv[1].rsplit('.', 1)[0] + '_deduplicated.csv'
dedup[['rock_type', 'casename', 'group', 'porosity', 'eff_porosity', 'AR', 'DOA', 'k']].rename(columns={'group': 'subvolume_id'}).to_csv(out_clean, index=False)
print('saved', out_clean)

# ---------------- models ----------------
def models():
    m = {'Linear Regression': (LinearRegression(), {}),
         'Decision Tree': (DecisionTreeRegressor(random_state=42),
                           {'max_depth': [2, 3, 4, 6, 8], 'min_samples_leaf': [5, 20, 50]}),
         'Random Forest': (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=JOBS),
                           {'max_depth': [4, 6, 8], 'min_samples_leaf': [1, 5, 20]})}
    if HAVE_XGB:
        m[BOOST] = (XGBRegressor(random_state=42, n_jobs=JOBS),
                    {'max_depth': [3, 5, 8], 'learning_rate': [.03, .05, .1],
                     'min_child_weight': [5, 10], 'n_estimators': [200, 400]})
    else:
        m[BOOST] = (HistGradientBoostingRegressor(random_state=42),
                    {'max_depth': [3, 5, 8], 'learning_rate': [.03, .05, .1],
                     'min_samples_leaf': [5, 10], 'max_iter': [200, 400]})
    if QUICK:   # shrink grids for a smoke test
        for k, (est, g) in list(m.items()):
            m[k] = (est, {p: v[:1] for p, v in g.items()} if g else g)
    return m

PERSPLIT = []
def run(df, feats, mode, label, save_plots=False):
    grouped = mode == 'grouped'
    splitter = (GroupShuffleSplit(N_SPLITS, test_size=.2, random_state=42).split(df, groups=df.group) if grouped
                else ShuffleSplit(N_SPLITS, test_size=.2, random_state=42).split(df))
    res = {m: [] for m in models()}; best = {m: [] for m in models()}
    for i, (a, b) in enumerate(splitter):
        tr, te = df.iloc[a], df.iloc[b]
        for name, (est, grid) in models().items():
            if grid:
                cv = GroupKFold(3) if grouped else KFold(3, shuffle=True, random_state=42)
                gs = GridSearchCV(est, grid, cv=cv, scoring='r2', n_jobs=1)
                gs.fit(tr[feats], tr.y, **({'groups': tr.group} if grouped else {}))
                model = gs.best_estimator_; best[name].append(gs.best_params_)
            else:
                model = est.fit(tr[feats], tr.y)
            p = model.predict(te[feats])
            res[name].append((r2_score(te.y, p), mean_squared_error(te.y, p) ** .5, mean_absolute_error(te.y, p)))
            PERSPLIT.append({'feature_set': label, 'validation': mode, 'model': name, 'split': i, 'R2': res[name][-1][0], 'RMSE': res[name][-1][1], 'MAE': res[name][-1][2]})
            if save_plots and i == 0:
                safe = name.split(' ')[0].replace('(', '')
                if name == BOOST:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.scatter(te.y, p, s=6, alpha=.4)
                    lim = [min(te.y.min(), p.min()), max(te.y.max(), p.max())]
                    ax.plot(lim, lim, 'k--', lw=1); ax.set_xlabel('Simulated log10(k)'); ax.set_ylabel('Predicted log10(k)')
                    ax.set_title(f'{BOOST} - {label} - {mode}', fontsize=8); fig.tight_layout()
                    fig.savefig(f'{OUT}/pred_vs_sim_{label}_{mode}.png', dpi=200); plt.close(fig)
                if hasattr(model, 'feature_importances_'):
                    fig, ax = plt.subplots(figsize=(5, 3))
                    imp = model.feature_importances_; ax.barh(feats, imp); ax.set_xlabel('Importance (normalized)')
                    ax.set_title(f'{name} importance - {label} - {mode}', fontsize=8); fig.tight_layout()
                    fig.savefig(f'{OUT}/importance_{safe}_{label}_{mode}.png', dpi=200); plt.close(fig)
                    pd.DataFrame({'feature': feats, 'importance': imp}).to_csv(f'{OUT}/importance_{safe}_{label}_{mode}.csv', index=False)
        print(f'  {label} / {mode}: split {i + 1}/{N_SPLITS} done', flush=True)
    summary = {}
    for m, v in res.items():
        v = np.array(v)
        summary[m] = {'R2': (v[:, 0].mean(), v[:, 0].std()), 'RMSE': (v[:, 1].mean(), v[:, 1].std()),
                      'MAE': (v[:, 2].mean(), v[:, 2].std()), 'best_params_each_split': best[m]}
    return summary

T3raw = raw[raw.rock_type == 3].reset_index(drop=True)
T3 = dedup[dedup.rock_type == 3].reset_index(drop=True)
experiments = [   # (label, mode, dataframe, features)
    ('all_types_porosity',            'naive',        raw,   ['lpor']),
    ('all_types_porosity',            'dedup_random', dedup, ['lpor']),
    ('all_types_porosity',            'grouped',      dedup, ['lpor']),
    ('type3_porosity',                'naive',        T3raw, ['lpor']),
    ('type3_porosity',                'dedup_random', T3,    ['lpor']),
    ('type3_porosity',                'grouped',      T3,    ['lpor']),
    ('type3_por_AR_DOA',              'naive',        T3raw, ['lpor', 'AR', 'DOA']),
    ('type3_por_AR_DOA',              'dedup_random', T3,    ['lpor', 'AR', 'DOA']),
    ('type3_por_AR_DOA',              'grouped',      T3,    ['lpor', 'AR', 'DOA']),
    ('type3_por_effpor_AR_DOA',       'grouped',      T3,    ['lpor', 'leff', 'AR', 'DOA']),
]
results = {}
def dump():
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1, default=float)
    rows = []
    for key, sm in results.items():
        label, mode = key.split('|')
        for m, v in sm.items():
            rows.append({'feature_set': label, 'validation': mode, 'model': m,
                         'R2': f"{v['R2'][0]:.3f} +/- {v['R2'][1]:.3f}", 'RMSE': f"{v['RMSE'][0]:.3f} +/- {v['RMSE'][1]:.3f}",
                         'MAE': f"{v['MAE'][0]:.3f} +/- {v['MAE'][1]:.3f}"})
    pd.DataFrame(rows).to_csv(f'{OUT}/results_table.csv', index=False)
    pd.DataFrame(PERSPLIT).to_csv(f'{OUT}/per_split_metrics.csv', index=False)
    return pd.DataFrame(rows)
for label, mode, df, feats in experiments:
    results[f'{label}|{mode}'] = run(df, feats, mode, label, save_plots=(label == 'type3_por_AR_DOA'))
    tab = dump()      # partial results are usable if you have to stop early
print('\n', tab.to_string(index=False))
print(f'\nall outputs are in ./{OUT}/ (results_table.csv, per_split_metrics.csv, results.json, figures).')
if not HAVE_XGB: print('note: xgboost was not installed; the boosting rows used sklearn HistGradientBoosting.')
