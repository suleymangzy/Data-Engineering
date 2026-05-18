# ═══════════════════════════════════════════════════════════════════
#  IMPORT SECTION
# ═══════════════════════════════════════════════════════════════════
import warnings
import os
import numpy as np
import pandas as pd
import logging
import re
import gc # RAM yönetimi için eklendi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import shap
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# Sklearn imports
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, 
    mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

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

# ── NumPy compatibility patch (numpy >= 2.0) ──────────────────────
for _attr, _type in [('float', float), ('int', int), ('bool', bool)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

# ── sklearn compatibility patch (sklearn >= 1.6) ────────────────────
import sklearn.base
try:
    from sklearn.utils.validation import validate_data as _skl_validate
    if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
        sklearn.base.BaseEstimator._validate_data = (
            lambda self, *a, **kw: _skl_validate(self, *a, **kw)
        )
except ImportError:
    pass

# ── gplearn compatibility patch (sklearn 1.6+) ──────────────────────
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
except (ImportError, AttributeError):
    pass

# ── evolutionary_forest compatibility patch ──────────────────────────
import evolutionary_forest.forest as _ef_mod
_ef_mod.consistency_check = lambda learner: None

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION SETTINGS
# ═══════════════════════════════════════════════════════════════════
np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
SCENARIO_ORDER = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']

SATURATED_INPUTS = ['T(girdi)']
SATURATED_OUTPUTS = [
    'P(çıktı)', 'v sıvı (çıktı)', 'v buhar (çıktı)',
    'h sıvı (çıktı)', 'h buhar (çıktı)',
    's sıvı (çıktı)', 's buhar (çıktı)'
]

SUPERHEATED_INPUTS = ['T (girdi)', 'P (girdi)']
SUPERHEATED_OUTPUTS = ['v (çıktı)', 'h (çıktı)', 's (çıktı)']

# ═══════════════════════════════════════════════════════════════════
#  REGRESSOR DICTIONARY
# ═══════════════════════════════════════════════════════════════════
def get_regressors():
    """Returns regression algorithms to be used (KNN added from paper)."""
    return {
        'AdaBoost':  AdaBoostRegressor(n_estimators=200, random_state=42),
        'CatBoost':  CatBoostRegressor(n_estimators=200, random_state=42, verbose=0),
        'DART':      lgb.LGBMRegressor(boosting_type='dart', n_estimators=200,
                                       random_state=42, verbose=-1),
        'EF':        EvolutionaryForestRegressor(
                         n_gen=20, n_pop=200, basic_primitives='optimal',
                         verbose=False, random_state=42, n_process=1),
        'ET':        ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'GBDT':      GradientBoostingRegressor(n_estimators=200, random_state=42),
        'GP':        SymbolicRegressor(
                         generations=20, population_size=1000,
                         function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg'],
                         parsimony_coefficient=0.005, max_samples=0.9,
                         verbose=0, random_state=42, n_jobs=1),
        'KNN':       KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'LightGBM':  lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
        'RF':        RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost':   xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
    }


# ═══════════════════════════════════════════════════════════════════
#  DATA PREPARATION - TRAIN/TEST SPLIT AND SCALING
# ═══════════════════════════════════════════════════════════════════
def prepare_data(df, input_cols, target_col, test_size=0.3, random_state=42):
    """Splits data into train/test sets and applies StandardScaler normalization."""
    X = df[input_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y


# ═══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING - SMOGN (Data Augmentation)
# ═══════════════════════════════════════════════════════════════════
def apply_smogn(X_train, y_train, k=5, rare_threshold_low=15, rare_threshold_high=85, noise_level=0.01):
    """
    Termodinamik veriler (özellikle düşük varyanslı doymuş fazlar) için 
    Çeşmeli (2025) teorisine dayanılarak sıfırdan yazılmış SMOGN algoritması.
    Kütüphane kaynaklı 'relevance function' (all points are 1) çökmelerini engeller.
    """
    print("Özel SMOGN Algoritması başlatılıyor (Custom Build)...")
    
    X = np.array(X_train)
    y = np.array(y_train)
    
    lower_bound = np.percentile(y, rare_threshold_low)
    upper_bound = np.percentile(y, rare_threshold_high)
    
    minority_idx = np.where((y <= lower_bound) | (y >= upper_bound))[0]
    majority_idx = np.where((y > lower_bound) & (y < upper_bound))[0]
    
    if len(minority_idx) < k or len(majority_idx) == 0:
        print("Uyarı: Veri varyansı sentetik üretime uygun değil, orijinal veriye dönülüyor.")
        return X_train, y_train

    X_min, y_min = X[minority_idx], y[minority_idx]
    X_maj, y_maj = X[majority_idx], y[majority_idx]
    
    synthetic_X = []
    synthetic_y = []
    
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X_min)
    distances, indices = nn.kneighbors(X_min)
    
    for i in range(len(X_min)):
        nn_idx = indices[i, np.random.randint(1, k+1)]
        step = np.random.rand()
        
        new_X = X_min[i] + step * (X_min[nn_idx] - X_min[i])
        new_y = y_min[i] + step * (y_min[nn_idx] - y_min[i])
        
        new_y += np.random.normal(0, noise_level * np.std(y))
        
        synthetic_X.append(new_X)
        synthetic_y.append(new_y)
        
    synthetic_X = np.array(synthetic_X)
    synthetic_y = np.array(synthetic_y)
    
    drop_count = int(len(majority_idx) * 0.20)
    keep_indices = np.random.choice(len(X_maj), len(X_maj) - drop_count, replace=False)
    
    X_maj_balanced = X_maj[keep_indices]
    y_maj_balanced = y_maj[keep_indices]
    
    X_final = np.vstack((X_min, synthetic_X, X_maj_balanced))
    y_final = np.concatenate((y_min, synthetic_y, y_maj_balanced))
    
    print(f"Özel SMOGN Başarılı! Orijinal: {len(X)} -> Artırılmış/Dengelenmiş: {len(X_final)}")
    
    return X_final, y_final

# ═══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING - STGP-EF (Feature Engineering)
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
        if isinstance(formatted_expr, (list, tuple)):
            return " | ".join(str(x) for x in formatted_expr)
        return str(formatted_expr)
    except Exception:
        return expr.replace('"', '')

def extract_symbolic_transformer_formulas(stgp_model, n_features: int = 10) -> dict:
    formulas = {}
    try:
        if hasattr(stgp_model, '_best_programs'):
            programs = stgp_model._best_programs
            n_to_show = min(n_features, len(programs))
            logger.info(f"\n{'─' * 78}")
            logger.info(f"SYMBOLIC REGRESSION (STGP) - Generated Features ({n_to_show} of {len(programs)})")
            logger.info(f"{'─' * 78}")
            for idx in range(n_to_show):
                formula = format_math_expr(str(programs[idx]))
                formulas[f'STGP_{idx}'] = formula
                logger.info(f"  STGP_{idx:02d}: {formula}")
    except Exception as e:
        logger.warning(f"Error extracting STGP formulas: {e}")
    return formulas

def extract_ef_formulas(ef_model, n_features: int = 10) -> dict:
    formulas = {}
    try:
        if hasattr(ef_model, '_best_hof') or hasattr(ef_model, 'hof'):
            hof = getattr(ef_model, '_best_hof', getattr(ef_model, 'hof', None))
            if hof is not None:
                n_to_show = min(n_features, len(hof))
                logger.info(f"\n{'─' * 78}")
                logger.info(f"EVOLUTIONARY FOREST (EF) - Generated Features ({n_to_show} of {len(hof)})")
                logger.info(f"{'─' * 78}")
                for idx in range(n_to_show):
                    formula = format_math_expr(str(hof[idx]))
                    formulas[f'EF_{idx}'] = formula
                    logger.info(f"  EF_{idx:02d}: {formula}")
    except Exception as e:
        logger.warning(f"Error extracting EF formulas: {e}")
    return formulas

def apply_stgp_ef(X_train, y_train, X_test, n_best_features=10):
    """
    STGP ve EF algoritmalarını kullanarak hibrit özellik inşası yapar.
    Terminali kirleten kütüphane çıktıları (population_evaluation vb.) susturulmuştur.
    """
    logger.info("Hibrit Özellik İnşası (STGP-EF) Başlatılıyor...")

    # Stage 1: STGP (Sembolik Transformer)
    X_train_stgp = np.empty((X_train.shape[0], 0))
    X_test_stgp  = np.empty((X_test.shape[0], 0))
    try:
        stgp_model = SymbolicTransformer(n_jobs=1, random_state=42)
        # İstenmeyen kütüphane çıktılarını susturmak için stdout yönlendirmesi
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            stgp_model.fit(X_train, y_train)
            X_train_stgp = np.nan_to_num(stgp_model.transform(X_train))
            X_test_stgp  = np.nan_to_num(stgp_model.transform(X_test))
        extract_symbolic_transformer_formulas(stgp_model, n_features=n_best_features)
    except Exception as e:
        logger.error(f"STGP başarısız oldu: {e}")
    finally:
        try: del stgp_model
        except: pass
        gc.collect()

    n_stgp_selected = min(n_best_features, X_train_stgp.shape[1])
    if n_stgp_selected > 0:
        X_train_stgp = X_train_stgp[:, :n_stgp_selected]
        X_test_stgp  = X_test_stgp[:, :n_stgp_selected]

    # Stage 2: EF (Evrimsel Orman)
    X_train_ef = np.empty((X_train.shape[0], 0))
    X_test_ef  = np.empty((X_test.shape[0], 0))
    try:
        ef_model = EvolutionaryForestRegressor(random_state=42, basic_primitives="default", verbose=False)
        # population_evaluation printlerini susturmak için stdout yönlendirmesi
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            ef_model.fit(X_train, y_train)
            X_train_ef = ef_model.transform(X_train)
            X_test_ef  = ef_model.transform(X_test)
        extract_ef_formulas(ef_model, n_features=n_best_features)
    except Exception as e:
        logger.error(f"EF başarısız oldu: {e}")
    finally:
        try: del ef_model
        except: pass
        gc.collect()

    n_ef_selected = min(n_best_features, X_train_ef.shape[1])
    if n_ef_selected > 0:
        X_train_ef = X_train_ef[:, :n_ef_selected]
        X_test_ef  = X_test_ef[:, :n_ef_selected]

    # Stage 3: Verilerin Birleştirilmesi
    X_train_constructed = np.hstack((X_train_stgp, X_train_ef)) if X_train_stgp.size and X_train_ef.size else np.empty((X_train.shape[0], 0))
    X_test_constructed  = np.hstack((X_test_stgp, X_test_ef)) if X_test_stgp.size and X_test_ef.size else np.empty((X_test.shape[0], 0))

    X_train_hybrid = np.hstack((X_train, X_train_constructed)) if X_train_constructed.size else X_train
    X_test_hybrid  = np.hstack((X_test, X_test_constructed)) if X_test_constructed.size else X_test

    return X_train_hybrid, X_test_hybrid


# ═══════════════════════════════════════════════════════════════════
#  REGRESSOR EVALUATION
# ═══════════════════════════════════════════════════════════════════
def evaluate_regressors(X_train, y_train, X_test, y_test):
    regressors = get_regressors()
    results = {}
    nan_template = {'Train_R2': np.nan, 'Test_R2': np.nan, 'Train_RMSE': np.nan, 'Test_RMSE': np.nan, 'Train_MAPE': np.nan, 'Test_MAPE': np.nan}

    for name, model in regressors.items():
        try:
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
            
            results[name] = {
                'Train_R2': round(r2_score(y_train, y_train_pred), 6),
                'Test_R2': round(r2_score(y_test, y_test_pred), 6),
                'Train_RMSE': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 6),
                'Test_RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 6),
                'Train_MAPE': round(mean_absolute_percentage_error(y_train, y_train_pred), 6),
                'Test_MAPE': round(mean_absolute_percentage_error(y_test, y_test_pred), 6)
            }
        except Exception as e:
            print(f"  Error ({name}): {e}")
            results[name] = nan_template.copy()

    return results

# ═══════════════════════════════════════════════════════════════════
#  SCENARIO ANALYSIS 
# ═══════════════════════════════════════════════════════════════════
def run_all_scenarios(df, input_cols, target_col):
    print(f"\n{'='*60}")
    print(f"  Target: {target_col}  |  Input: {input_cols}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test, _, _ = prepare_data(df, input_cols, target_col)
    all_results = {}
    empty_metrics = {k: np.nan for k in ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAPE', 'Test_MAPE']}

    # 1) Base
    print("  [1/4] Base training...")
    all_results['Base'] = evaluate_regressors(X_train, y_train, X_test, y_test)
    nan_results = {name: empty_metrics.copy() for name in all_results['Base']}

    # 2) SMOGN
    print("  [2/4] SMOGN training...")
    try:
        X_smogn, y_smogn = apply_smogn(X_train, y_train)
        all_results['SMOGN'] = evaluate_regressors(X_smogn, y_smogn, X_test, y_test)
    except Exception as e:
        print(f"    SMOGN error: {e}")
        all_results['SMOGN'] = nan_results

    # 3) STGP-EF
    print("  [3/4] STGP-EF training...")
    try:
        X_tr_ef, X_te_ef = apply_stgp_ef(X_train, y_train, X_test)
        all_results['STGP-EF'] = evaluate_regressors(X_tr_ef, y_train, X_te_ef, y_test)
    except Exception as e:
        print(f"    STGP-EF error: {e}")
        all_results['STGP-EF'] = nan_results

    # 4) SMOGN + STGP-EF
    print("  [4/4] SMOGN + STGP-EF training...")
    try:
        if 'X_smogn' not in locals() or X_smogn is None:
            X_smogn, y_smogn = apply_smogn(X_train, y_train)
        X_smogn_ef, X_te_ef2 = apply_stgp_ef(X_smogn, y_smogn, X_test)
        all_results['SMOGN+STGP-EF'] = evaluate_regressors(X_smogn_ef, y_smogn, X_te_ef2, y_test)
    except Exception as e:
        print(f"    SMOGN+STGP-EF error: {e}")
        all_results['SMOGN+STGP-EF'] = nan_results

    print("  Completed.\n")
    return all_results

# ── HATA ÇÖZÜMÜ: Fonksiyon İsimleri Notebook ile Uyumlulaştırıldı ──
def run_saturated_analysis(df):
    """Doymuş Buhar Analizi"""
    print("\n" + "▓" * 60)
    print("  DOYMUŞ BUHAR ANALİZİ")
    print("▓" * 60)
    all_target_results = {}
    for target in SATURATED_OUTPUTS:
        all_target_results[target] = run_all_scenarios(df, SATURATED_INPUTS, target)
    return all_target_results

def run_superheated_analysis(df):
    """Kızgın Buhar Analizi"""
    print("\n" + "▓" * 60)
    print("  KIZGIN BUHAR ANALİZİ")
    print("▓" * 60)
    all_target_results = {}
    for target in SUPERHEATED_OUTPUTS:
        all_target_results[target] = run_all_scenarios(df, SUPERHEATED_INPUTS, target)
    return all_target_results

# ═══════════════════════════════════════════════════════════════════
#  RESULTS PROCESSING AND SAVING
# ═══════════════════════════════════════════════════════════════════
def build_results_table(target_results):
    rows = []
    for target, scenarios in target_results.items():
        for scenario, scores in scenarios.items():
            for algo, metrics in scores.items():
                row = {'Target': target, 'Scenario': scenario, 'Algorithm': algo}
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)

def show_best_results(results_df):
    idx = results_df.groupby('Target')['Test_R2'].idxmax()
    cols = ['Target', 'Scenario', 'Algorithm', 'Test_R2', 'Train_R2', 'Test_RMSE', 'Test_MAPE']
    best = results_df.loc[idx, [c for c in cols if c in results_df.columns]]
    return best.reset_index(drop=True)

def compare_scenarios(results_df):
    pivot = results_df.pivot_table(index=['Target', 'Algorithm'], columns='Scenario', values='Test_R2')
    return pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])

def target_summary(results_df, target_col):
    sub = results_df[results_df['Target'] == target_col]
    pivot = sub.pivot_table(index='Algorithm', columns='Scenario', values='Test_R2')
    return pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])

def save_wide_results(df_long, path):
    metrics = ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAPE', 'Test_MAPE']
    index_cols = [c for c in ['Dataset', 'Target', 'Algorithm'] if c in df_long.columns]
    
    pivot = df_long.pivot_table(index=index_cols, columns='Senaryo', values=metrics)
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    
    ordered_cols = [f"{s}_{m}" for s in SCENARIO_ORDER for m in metrics if f"{s}_{m}" in pivot.columns]
    pivot = pivot.reindex(columns=ordered_cols)

    test_r2_cols = [f"{s}_Test_R2" for s in SCENARIO_ORDER if f"{s}_Test_R2" in pivot.columns]
    pivot['Max_Test_R2'] = pivot[test_r2_cols].max(axis=1)
    pivot['Max_Scenario'] = pivot[test_r2_cols].idxmax(axis=1).str.replace('_Test_R2', '')

    wide = pivot.reset_index().sort_values(index_cols).reset_index(drop=True)
    wide.to_csv(path, index=False, float_format='%.6f')
    
    save_comparison_summary(wide, path)
    return wide

def save_comparison_summary(wide_df, path):
    scenarios = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']
    cols = [s for s in scenarios if f"{s}_Test_R2" in wide_df.columns]

    with open(path, 'a', encoding='utf-8-sig', newline='') as f:
        f.write('\nSCENARIO BASED COMPARISON\nScenario,Win_Count,Win_Rate_%,Scenario_Mean_Test_R2,Winner_Mean_Max_Test_R2\n')
        for s in cols:
            col_name = f"{s}_Test_R2"
            win_count = int((wide_df['Max_Scenario'] == s).sum())
            win_rate  = round(win_count / len(wide_df) * 100, 2)
            mean_s    = round(wide_df[col_name].mean(), 6)
            mask      = wide_df['Max_Scenario'] == s
            mean_max  = round(wide_df.loc[mask, 'Max_Test_R2'].mean(), 6) if win_count > 0 else ''
            f.write(f'{s},{win_count},{win_rate},{mean_s},{mean_max}\n')

        f.write('\nALGORITHM BASED COMPARISON\n')
        header = ('Algorithm,' + ','.join([f"Mean_{s}_Test_R2" for s in cols]) +
                  ',Mean_Max_Test_R2,Best_Scenario,' + ','.join([f"Win_Count_{s}" for s in cols]) + '\n')
        f.write(header)
        
        for alg in sorted(wide_df['Algorithm'].unique()):
            grp = wide_df[wide_df['Algorithm'] == alg]
            means = [round(grp[f"{s}_Test_R2"].mean(), 6) for s in cols]
            modes = grp['Max_Scenario'].mode()
            best = modes[0] if not modes.empty else "Unknown"
            wins = grp['Max_Scenario'].value_counts().to_dict()
            wins_vals = [wins.get(s, 0) for s in cols]
            row = [alg] + means + [round(grp['Max_Test_R2'].mean(), 6), best] + wins_vals
            f.write(','.join(str(x) for x in row) + '\n')

# ═══════════════════════════════════════════════════════════════════
#  SHAP EXPLAINABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def calculate_shap_values(model, X_data):
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X_data)

def plot_global_feature_importance(shap_values, X_data, save_path=None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.title("Küresel Özellik Önemi (Global Feature Importance)", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_global_summary(shap_values, X_data, save_path=None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, show=False)
    plt.title("SHAP Özet Grafiği (SHAP Summary Plot)", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_local_waterfall(explainer, shap_values, X_data, instance_index=0, save_path=None):
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, shap.Explanation):
        shap.plots.waterfall(shap_values[instance_index], show=False)
    else:
        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[instance_index], X_data.iloc[instance_index], show=False)
    plt.title(f"Yerel Açıklanabilirlik - İndeks: {instance_index}", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_thermodynamic_dependence(shap_values, X_data, feature_name, interaction_feature="auto", save_path=None):
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature_name, shap_values, X_data, interaction_index=interaction_feature, show=False)
    plt.title(f"Termodinamik Bağımlılık Analizi: {feature_name}", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()