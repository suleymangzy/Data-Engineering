# ═══════════════════════════════════════════════════════════════════
#  IMPORT SECTION
# ═══════════════════════════════════════════════════════════════════
import warnings
import os
import numpy as np
import pandas as pd
import logging
import re
from contextlib import contextmanager, redirect_stdout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sklearn imports
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, 
    mean_absolute_percentage_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ML Kütüphaneleri
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
#  COMPATIBILITY PATCHES
# ═══════════════════════════════════════════════════════════════════
for _attr, _type in [('float', float), ('int', int), ('bool', bool)]:
    if not hasattr(np, _attr): setattr(np, _attr, _type)

import sklearn.base
try:
    from sklearn.utils.validation import validate_data as _skl_validate
    if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
        sklearn.base.BaseEstimator._validate_data = lambda self, *a, **kw: _skl_validate(self, *a, **kw)
except ImportError: pass

try:
    import gplearn.genetic as _gp
    from sklearn.utils.validation import check_array as _check_array
    def _gplearn_validate_data(self, X, y=None, **kwargs):
        X = _check_array(X, dtype='numeric')
        self.n_features_in_ = X.shape[1]
        if y is not None:
            y = _check_array(y, ensure_2d=False, dtype='numeric')
            return X, y
        return X
    _gp.BaseSymbolic._validate_data = _gplearn_validate_data
except (ImportError, AttributeError): pass

import evolutionary_forest.forest as _ef_mod
_ef_mod.consistency_check = lambda learner: None

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════
np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

SATURATED_INPUTS = ['T(girdi)']
SATURATED_OUTPUTS = ['P(çıktı)', 'v sıvı (çıktı)', 'v buhar (çıktı)', 'h sıvı (çıktı)', 'h buhar (çıktı)', 's sıvı (çıktı)', 's buhar (çıktı)']
SUPERHEATED_INPUTS = ['T (girdi)', 'P (girdi)']
SUPERHEATED_OUTPUTS = ['v (çıktı)', 'h (çıktı)', 's (çıktı)']

def get_regressors():
    """ Hızlı ve güçlü regresörler. GP ve EF tahmin listesinden çıkarılarak donma sorunu çözülmüştür. """
    return {
        'AdaBoost':  AdaBoostRegressor(random_state=42),
        'CatBoost':  CatBoostRegressor(random_state=42, verbose=0),
        'DART':      lgb.LGBMRegressor(boosting_type='dart', random_state=42, verbose=-1, n_jobs=-1),
        'ET':        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'GBDT':      GradientBoostingRegressor(random_state=42),
        'KNN':       KNeighborsRegressor(n_jobs=-1),
        'LightGBM':  lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
        'RF':        RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost':   xgb.XGBRegressor(random_state=42, verbosity=0),
    }


# ═══════════════════════════════════════════════════════════════════
#  FORMULA FORMATTING
# ═══════════════════════════════════════════════════════════════════
def format_math_expr(expr: str) -> str:
    expr = str(expr).strip()
    expr = re.sub(r'(?i)ARG(\d+)', r'x\1', expr)
    expr = re.sub(r'\bX(\d+)\b', r'x\1', expr)
    expr = expr.replace('"', '').replace("'", "")
    expr = re.sub(r'\b(x\d+)\b', r'"\1"', expr)
    safe_dict = {
        'Add': lambda a, b: f"({a} + {b})", 'add': lambda a, b: f"({a} + {b})",
        'Sub': lambda a, b: f"({a} - {b})", 'sub': lambda a, b: f"({a} - {b})",
        'Mul': lambda a, b: f"({a} * {b})", 'mul': lambda a, b: f"({a} * {b})",
        'Div': lambda a, b: f"({a} / {b})", 'div': lambda a, b: f"({a} / {b})",
        'AQ':  lambda a, b: f"({a} / {b})",
        'Sin': lambda a: f"sin({a})", 'sin': lambda a: f"sin({a})",
        'Cos': lambda a: f"cos({a})", 'cos': lambda a: f"cos({a})",
        'Exp': lambda a: f"exp({a})", 'exp': lambda a: f"exp({a})",
        'Log': lambda a: f"log({a})", 'log': lambda a: f"log({a})",
        'Abs': lambda a: f"abs({a})", 'abs': lambda a: f"abs({a})",
        'Neg': lambda a: f"(-{a})", 'neg': lambda a: f"(-{a})",
        'Inv': lambda a: f"(1 / {a})", 'inv': lambda a: f"(1 / {a})",
        'Max': lambda a, b: f"max({a}, {b})", 'max': lambda a, b: f"max({a}, {b})",
        'Min': lambda a, b: f"min({a}, {b})", 'min': lambda a, b: f"min({a}, {b})"
    }
    try:
        formatted_expr = eval(expr, {"__builtins__": {}}, safe_dict)
        if isinstance(formatted_expr, (list, tuple)): return str(formatted_expr[0])
        return str(formatted_expr)
    except Exception:
        return expr.replace('"', '').split('|')[0].strip()


# ═══════════════════════════════════════════════════════════════════
#  SCENARIOS & DOMAIN KNOWLEDGE FEATURES
# ═══════════════════════════════════════════════════════════════════
# Ana senaryolar: yalnizca HAM girdilerden turetilen evrimsel ozniteliklerin katkisini olcer.
SCENARIOS = ['Base', 'STGP', 'EF', 'HYBRID']
# Ayri karsilastirma (SADECE doymus faz): ham girdi + 1/T + ln(T). Kazancin evrimsel
# ozniteliklerden mi, yalnizca termodinamik donusumlerden mi geldigini ayirt etmek icindir.
DOMAIN_SCENARIO = 'Domain_Knowledge'
ALL_SCENARIOS = SCENARIOS + [DOMAIN_SCENARIO]
DOMAIN_FEATURE_NAMES = ['1/T', 'ln(T)']
KELVIN_OFFSET = 273.15


def is_saturated_input(input_cols):
    return len(input_cols) == 1 and input_cols[0] == 'T(girdi)'


def scenarios_for(input_cols):
    """Doymus fazda 5 senaryo (Alan Bilgisi dahil), kizgin fazda yalnizca 4 ana senaryo."""
    return ALL_SCENARIOS if is_saturated_input(input_cols) else SCENARIOS


def domain_knowledge_features(X_raw):
    """Tek girdili (T, °C) doymus faz icin 1/T ve ln(T) (Kelvin) uretir."""
    T_K = X_raw[:, 0] + KELVIN_OFFSET
    return np.column_stack([1.0 / T_K, np.log(T_K)])


# ═══════════════════════════════════════════════════════════════════
#  FEATURE CONSTRUCTION (STGP / EF) - FOLD BAZLI, SIZINTISIZ
# ═══════════════════════════════════════════════════════════════════
@contextmanager
def _quiet():
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        yield


def _clean(arr):
    return np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=1e12, neginf=-1e12)


def _fit_stgp(X_train, y_train, n_best_features=10):
    """STGP'yi SADECE egitim verisiyle fit eder. Donus: (model, kullanilan_sutun_sayisi, formuller)."""
    try:
        model = SymbolicTransformer(n_jobs=1, random_state=42)
        with _quiet():
            model.fit(X_train, y_train)
            X_out = _clean(model.transform(X_train))
        n_cols = min(n_best_features, X_out.shape[1])
        forms = {}
        if hasattr(model, '_best_programs'):
            for idx, prog in enumerate(model._best_programs[:n_cols]):
                forms[f'STGP_{idx:02d}'] = format_math_expr(str(prog))
        return model, n_cols, forms
    except Exception as e:
        logger.warning(f"STGP fit/transform basarisiz oldu, bu oznitelik grubu atlaniyor: {e}")
        return None, 0, {}


def _fit_ef(X_train, y_train, n_best_features=10):
    """EF'yi SADECE egitim verisiyle fit eder. Donus: (model, kullanilan_sutun_sayisi, formuller)."""
    try:
        model = EvolutionaryForestRegressor(random_state=42, basic_primitives="default", verbose=False, n_process=1)
        with _quiet():
            model.fit(X_train, y_train)
            X_out = _clean(model.transform(X_train))
        n_cols = min(n_best_features, X_out.shape[1])
        forms = {}
        hof = getattr(model, '_best_hof', getattr(model, 'hof', None))
        if hof is not None:
            for idx, prog in enumerate(hof[:n_cols]):
                forms[f'EF_{idx:02d}'] = format_math_expr(str(prog))
        return model, n_cols, forms
    except Exception as e:
        logger.warning(f"EF fit/transform basarisiz oldu, bu oznitelik grubu atlaniyor: {e}")
        return None, 0, {}


def _transform_features(model, X_scaled, n_cols):
    if model is None or n_cols == 0:
        return np.empty((X_scaled.shape[0], 0))
    with _quiet():
        return _clean(model.transform(X_scaled))[:, :n_cols]


class ScenarioFeatureBuilder:
    """Bir katin egitim verisiyle fit edilir; ayni donusumu test setine ve sentetik izgaraya uygular.
    STGP/EF hicbir zaman test hedefini gormez."""

    def __init__(self, input_cols, n_best_features=10):
        self.input_cols = list(input_cols)
        self.n_best_features = n_best_features
        self.scenarios = scenarios_for(self.input_cols)

    def fit(self, X_train, y_train):
        self.scaler_ = StandardScaler().fit(X_train)
        X_scaled = self.scaler_.transform(X_train)
        self.stgp_model_, self.n_stgp_, self.stgp_forms_ = _fit_stgp(X_scaled, y_train, self.n_best_features)
        self.ef_model_, self.n_ef_, self.ef_forms_ = _fit_ef(X_scaled, y_train, self.n_best_features)
        return self

    def build_matrices(self, X_raw):
        """Her senaryo icin (olceklenmemis) ozellik matrisini dondurur."""
        X_scaled = self.scaler_.transform(X_raw)
        X_stgp = _transform_features(self.stgp_model_, X_scaled, self.n_stgp_)
        X_ef = _transform_features(self.ef_model_, X_scaled, self.n_ef_)
        mats = {
            'Base': X_raw,
            'STGP': np.hstack([X_raw, X_stgp]),
            'EF': np.hstack([X_raw, X_ef]),
            'HYBRID': np.hstack([X_raw, X_stgp, X_ef]),
        }
        if DOMAIN_SCENARIO in self.scenarios:
            mats[DOMAIN_SCENARIO] = np.hstack([X_raw, domain_knowledge_features(X_raw)])
        return mats

    def feature_info(self, scenario):
        names = list(self.input_cols)
        types = ['Orijinal'] * len(names)
        if scenario in ('STGP', 'HYBRID'):
            names += [f'STGP_{i:02d}' for i in range(self.n_stgp_)]
            types += ['STGP'] * self.n_stgp_
        if scenario in ('EF', 'HYBRID'):
            names += [f'EF_{i:02d}' for i in range(self.n_ef_)]
            types += ['EF'] * self.n_ef_
        if scenario == DOMAIN_SCENARIO:
            names += DOMAIN_FEATURE_NAMES
            types += ['Alan_Bilgisi'] * len(DOMAIN_FEATURE_NAMES)
        return names, types


# ═══════════════════════════════════════════════════════════════════
#  THERMODYNAMIC CONSISTENCY (MONOTONICITY) TEST ON SYNTHETIC GRID
# ═══════════════════════════════════════════════════════════════════
def build_synthetic_grid(input_cols, vary_col='T', min_val=-30, max_val=50, step=0.1,
                         fixed_col=None, fixed_val=None):
    """Ham girdi sutunlarindan olusan sentetik izgara dondurur: (grid_vals, X_grid_raw)."""
    grid_vals = np.arange(min_val, max_val, step)
    t_col = 'T (girdi)' if 'T (girdi)' in input_cols else 'T(girdi)'
    p_col = 'P (girdi)' if 'P (girdi)' in input_cols else None

    grid_data = {}
    if vary_col == 'T':
        grid_data[t_col] = grid_vals
        if fixed_col == 'P' and p_col: grid_data[p_col] = fixed_val
    elif vary_col == 'P':
        grid_data[p_col] = grid_vals
        if fixed_col == 'T': grid_data[t_col] = fixed_val

    grid_df = pd.DataFrame(grid_data)[list(input_cols)]
    return grid_vals, grid_df.values.astype(np.float64)


def get_grid_specs(input_cols):
    if len(input_cols) == 1 and 'T(girdi)' in input_cols:
        return [dict(vary_col='T', min_val=-30, max_val=50, step=0.1)]
    return [
        dict(vary_col='T', min_val=20, max_val=100, step=0.1, fixed_col='P', fixed_val=1.5),   # izobarik
        dict(vary_col='P', min_val=0.5, max_val=3.0, step=0.01, fixed_col='T', fixed_val=40.0),  # izotermal
    ]


def is_decreasing_expected(vary_col, target_name):
    if vary_col == 'T' and target_name in ['v buhar (çıktı)', 's buhar (çıktı)']: return True
    if vary_col == 'P' and target_name in ['v (çıktı)', 's (çıktı)']: return True
    return False


def evaluate_grid_violations(fitted_pipes, builder, input_cols, target_name, fold):
    """Fold modelleriyle sentetik izgarada monotonluk ihlallerini sayar (senaryo x algoritma)."""
    rows = []
    for spec in get_grid_specs(input_cols):
        grid_vals, X_grid_raw = build_synthetic_grid(input_cols, **spec)
        matrices = builder.build_matrices(X_grid_raw)
        decreasing = is_decreasing_expected(spec['vary_col'], target_name)
        fixed_col, fixed_val = spec.get('fixed_col'), spec.get('fixed_val')
        test_str = (f"Sabit {fixed_col}={fixed_val}, {spec['vary_col']} degisiyor" if fixed_col
                    else f"{spec['vary_col']} degisiyor (Doymus)")

        for (scenario, algo_name), pipe in fitted_pipes.items():
            try:
                preds = pipe.predict(matrices[scenario])
                derivatives = np.diff(preds) / np.diff(grid_vals)
                violations = int((derivatives > 0).sum() if decreasing else (derivatives < 0).sum())
                rows.append({
                    'Target': target_name, 'Algoritma': algo_name, 'Scenario': scenario, 'Fold': fold,
                    'Test_Tipi': test_str, 'Fiziksel_Ihlal_Sayisi': violations,
                    'Ihlal_Yuzdesi_(%)': violations / len(derivatives) * 100,
                })
            except Exception as e:
                logger.warning(f"{algo_name}/{scenario} icin sentetik izgara testi basarisiz: {e}")
    return rows


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════
def _order_scenarios(df):
    df = df.copy()
    df['Scenario'] = pd.Categorical(df['Scenario'], categories=ALL_SCENARIOS, ordered=True)
    return df


def summarize_scenarios(df_met, df_uyum):
    """Senaryo bazinda ortalama Test R2, Test MAPE ve monotonluk ihlal yuzdesi (RMSE hedefler arasi birim farki nedeniyle yok)."""
    m = df_met.groupby('Scenario', observed=True)[['Test_R2', 'Test_MAPE']].mean()
    v = df_uyum.groupby('Scenario', observed=True)['Ihlal_Yuzdesi_(%)'].mean().rename('Ort_Ihlal_Yuzdesi_(%)')
    return m.join(v)


def build_domain_knowledge_comparison(df_met, df_uyum):
    """Mevcut senaryolari (doymus fazda Domain_Knowledge dahil) hedef x algoritma bazinda yan yana koyar."""
    met = df_met.pivot_table(index=['Target', 'Algorithm'], columns='Scenario',
                             values=['Test_R2', 'Test_RMSE'], observed=True)
    met.columns = [f'{metric}_{scen}' for metric, scen in met.columns]

    viol = (df_uyum.groupby(['Target', 'Algoritma', 'Scenario'], observed=True)['Ihlal_Yuzdesi_(%)']
            .mean().unstack('Scenario'))
    viol.columns = [f'Ihlal_{scen}' for scen in viol.columns]
    viol.index.names = ['Target', 'Algorithm']

    cmp = met.join(viol).reset_index()
    ordered = ['Target', 'Algorithm'] + [f'{p}_{s}' for p in ('Test_R2', 'Test_RMSE', 'Ihlal') for s in ALL_SCENARIOS]
    cmp = cmp[[c for c in ordered if c in cmp.columns]]

    pairs = [(DOMAIN_SCENARIO, 'Base'), ('STGP', 'Base'), ('EF', 'Base'), ('HYBRID', 'Base'), ('HYBRID', DOMAIN_SCENARIO)]
    for a, b in pairs:
        if f'Test_R2_{a}' in cmp and f'Test_R2_{b}' in cmp:
            cmp[f'Delta_R2_{a}_vs_{b}'] = cmp[f'Test_R2_{a}'] - cmp[f'Test_R2_{b}']
        if f'Ihlal_{a}' in cmp and f'Ihlal_{b}' in cmp:
            cmp[f'Delta_Ihlal_{a}_vs_{b}'] = cmp[f'Ihlal_{a}'] - cmp[f'Ihlal_{b}']
    return cmp


# ═══════════════════════════════════════════════════════════════════
#  UNIFIED MASTER LOOP
# ═══════════════════════════════════════════════════════════════════
def run_unified_analysis(df, input_cols, outputs_list, dataset_name,
                         n_splits=5, n_best_features=10, algorithms=None, output_dir="Results"):
    """Tum analiz TEK bir K-Fold dongusunde yapilir. STGP, EF, olceklendiriciler ve regresorler her katta
    YALNIZCA o katin egitim verisiyle fit edilir; test verisi ve sentetik izgara sadece transform/predict gorur.

    Senaryolar: Base, STGP, EF, HYBRID. Doymus fazda ayrica Domain_Knowledge (1/T, ln(T)) karsilastirmasi yapilir.
    """
    print(f"\n{'▓'*60}")
    print(f"  {dataset_name.upper()} - SIZINTISIZ K-FOLD VE TERMODİNAMİK UYUM ANALİZİ")
    print(f"{'▓'*60}")

    os.makedirs(output_dir, exist_ok=True)
    epsilon = 1e-10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    regressors = get_regressors()
    if algorithms is not None:
        regressors = {k: v for k, v in regressors.items() if k in algorithms}

    X_raw = df[input_cols].values.astype(np.float64)
    all_metrics, all_importances, all_uyum, all_formulas = [], [], [], []

    # -------------------------------------------------------------------------
    # FAZ 1: FOLD DONGUSU (oznitelik uretimi + 5 senaryo egitimi + importance + monotonluk)
    # -------------------------------------------------------------------------
    print("\n[FAZ 1/3] K-Fold dongusu: STGP-EF her katta yalnizca egitim verisiyle fit ediliyor...")
    for target_col in outputs_list:
        print(f"  -> Hedef: {target_col}")
        y = df[target_col].values.astype(np.float64)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_raw), start=1):
            X_tr, X_te, y_tr, y_te = X_raw[train_idx], X_raw[test_idx], y[train_idx], y[test_idx]

            builder = ScenarioFeatureBuilder(input_cols, n_best_features).fit(X_tr, y_tr)
            mats_tr, mats_te = builder.build_matrices(X_tr), builder.build_matrices(X_te)

            for key, form in {**builder.stgp_forms_, **builder.ef_forms_}.items():
                all_formulas.append({'Target': target_col, 'Fold': fold, 'Oznitelik': key,
                                     'Algoritma': 'STGP' if key.startswith('STGP') else 'EF', 'Formul': form})

            fitted_pipes = {}
            for scenario in builder.scenarios:
                f_names, f_types = builder.feature_info(scenario)
                for algo_name, template in regressors.items():
                    pipe = Pipeline([('scaler', StandardScaler()), ('model', clone(template))])
                    with _quiet():
                        pipe.fit(mats_tr[scenario], y_tr)
                        y_tr_pred = pipe.predict(mats_tr[scenario])
                        y_te_pred = pipe.predict(mats_te[scenario])
                    fitted_pipes[(scenario, algo_name)] = pipe

                    all_metrics.append({
                        'Target': target_col, 'Algorithm': algo_name, 'Scenario': scenario, 'Fold': fold,
                        'Train_R2': r2_score(y_tr, y_tr_pred), 'Test_R2': r2_score(y_te, y_te_pred),
                        'Train_RMSE': np.sqrt(mean_squared_error(y_tr, y_tr_pred)), 'Test_RMSE': np.sqrt(mean_squared_error(y_te, y_te_pred)),
                        'Train_MAE': mean_absolute_error(y_tr, y_tr_pred), 'Test_MAE': mean_absolute_error(y_te, y_te_pred),
                        'Train_MAPE': mean_absolute_percentage_error(y_tr + epsilon, y_tr_pred + epsilon),
                        'Test_MAPE': mean_absolute_percentage_error(y_te + epsilon, y_te_pred + epsilon),
                    })

                    if scenario != 'Base':
                        with _quiet():
                            pi = permutation_importance(pipe, mats_te[scenario], y_te, n_repeats=5, random_state=42, n_jobs=-1)
                        for i, fname in enumerate(f_names):
                            all_importances.append({
                                'Target': target_col, 'Scenario': scenario, 'Algorithm': algo_name, 'Feature': fname,
                                'Type': f_types[i], 'Importance_Mean': pi.importances_mean[i],
                                'Importance_Std': pi.importances_std[i], 'Fold': fold,
                            })

            all_uyum.extend(evaluate_grid_violations(fitted_pipes, builder, input_cols, target_col, fold))

    # -------------------------------------------------------------------------
    # FAZ 2: FOLD SONUCLARININ ORTALANMASI
    # -------------------------------------------------------------------------
    print("\n[FAZ 2/3] Fold sonuclari ortalaniyor...")
    sort_cols = ['Target', 'Algorithm', 'Scenario']
    df_met = (pd.DataFrame(all_metrics).groupby(sort_cols).mean(numeric_only=True)
              .drop(columns=['Fold']).reset_index())
    df_met = _order_scenarios(df_met).sort_values(sort_cols).reset_index(drop=True)

    df_imp = (pd.DataFrame(all_importances).groupby(['Target', 'Scenario', 'Algorithm', 'Feature', 'Type'])
              .mean(numeric_only=True).drop(columns=['Fold']).reset_index())
    df_imp = _order_scenarios(df_imp).sort_values(
        by=['Target', 'Scenario', 'Algorithm', 'Importance_Mean'], ascending=[True, True, True, False]).reset_index(drop=True)

    df_uyum = (pd.DataFrame(all_uyum).groupby(['Target', 'Algoritma', 'Scenario', 'Test_Tipi'])
               .mean(numeric_only=True).drop(columns=['Fold']).reset_index())
    df_uyum = _order_scenarios(df_uyum).sort_values(['Target', 'Algoritma', 'Scenario', 'Test_Tipi']).reset_index(drop=True)

    df_domain = build_domain_knowledge_comparison(df_met, df_uyum)

    # -------------------------------------------------------------------------
    # FAZ 3: EXCEL CIKTISI
    # -------------------------------------------------------------------------
    print("\n[FAZ 3/3] Tum sonuclar Excel dosyasina kaydediliyor...")
    excel_path = os.path.join(output_dir, f"{dataset_name}_Nihai_Sonuclar.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_met.to_excel(writer, sheet_name='Performans_Ortalamalari', index=False)
        df_imp.to_excel(writer, sheet_name='Oznitelik_Bireysel_Katki', index=False)
        df_uyum.to_excel(writer, sheet_name='Termodinamik_Uyum', index=False)
        if DOMAIN_SCENARIO in df_met['Scenario'].astype(str).unique():
            df_domain.to_excel(writer, sheet_name='Alan_Bilgisi_Karsilastirma', index=False)
        pd.DataFrame(all_formulas).to_excel(writer, sheet_name='Uretilen_Formuller', index=False)

    print(f"-> Çıktı Dosyası Başarıyla Oluşturuldu: {excel_path}")
    return df_met, df_imp, df_uyum
