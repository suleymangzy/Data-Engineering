# ═══════════════════════════════════════════════════════════════════
#  IMPORT SECTION
# ═══════════════════════════════════════════════════════════════════
import warnings
import os
import numpy as np
import pandas as pd
import logging
import re
import gc 

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
    mean_absolute_percentage_error, mean_absolute_error
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
def prepare_data(df, input_cols, target_col, test_size=0.2, random_state=42):
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
#  FEATURE ENGINEERING - STGP-EF (FEATURE CONSTRUCTION)
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
            logger.info("SYMBOLIC TRANSFORMER (STGP) - Generated Features")
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
                logger.info("EVOLUTIONARY FOREST (EF) - Generated Features")
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
        ef_model = EvolutionaryForestRegressor(random_state=42, basic_primitives="default", verbose=False, n_process=1)
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
    nan_template = {'Train_R2': np.nan, 'Test_R2': np.nan, 'Train_RMSE': np.nan, 'Test_RMSE': np.nan, 'Train_MAE': np.nan, 'Test_MAE': np.nan, 'Train_MAPE': np.nan, 'Test_MAPE': np.nan}

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
                'Train_MAE': round(mean_absolute_error(y_train, y_train_pred), 6),
                'Test_MAE': round(mean_absolute_error(y_test, y_test_pred), 6),
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
def run_hybrid_scenarios(df, input_cols, target_col):
    print(f"\n{'='*60}")
    print(f"  Target: {target_col}  |  Input: {input_cols}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test, _, _ = prepare_data(df, input_cols, target_col)
    all_results = {}
    empty_metrics = {k: np.nan for k in ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAE', 'Test_MAE', 'Train_MAPE', 'Test_MAPE']}

    # 1) Base
    print("  [1/2] Base training...")
    all_results['Base'] = evaluate_regressors(X_train, y_train, X_test, y_test)
    nan_results = {name: empty_metrics.copy() for name in all_results['Base']}

    # 2) STGP-EF
    print("  [2/2] STGP-EF training...")
    try:
        X_tr_ef, X_te_ef = apply_stgp_ef(X_train, y_train, X_test)
        all_results['STGP-EF'] = evaluate_regressors(X_tr_ef, y_train, X_te_ef, y_test)
    except Exception as e:
        print(f"    STGP-EF error: {e}")
        all_results['STGP-EF'] = nan_results

    print("  Completed.\n")
    return all_results

def run_all_scenarios(df, input_cols, target_col):
    print(f"\n{'='*60}")
    print(f"  Target: {target_col}  |  Input: {input_cols}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test, _, _ = prepare_data(df, input_cols, target_col)
    all_results = {}
    empty_metrics = {k: np.nan for k in ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAE', 'Test_MAE', 'Train_MAPE', 'Test_MAPE']}

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

# ═══════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTIONS FOR SATURATED AND SUPERHEATED SCENARIOS
# ═══════════════════════════════════════════════════════════════════
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
#  MAIN ANALYSIS FUNCTIONS FOR HYBRID SCENARIOS (STGP-EF ONLY)
# ═══════════════════════════════════════════════════════════════════
def run_saturated_hybrid_analysis(df):
    """Doymuş Buhar Analizi"""
    print("\n" + "▓" * 60)
    print("  DOYMUŞ BUHAR ANALİZİ")
    print("▓" * 60)
    all_target_results = {}
    for target in SATURATED_OUTPUTS:
        all_target_results[target] = run_hybrid_scenarios(df, SATURATED_INPUTS, target)
    return all_target_results

def run_superheated_hybrid_analysis(df):
    """Kızgın Buhar Analizi"""
    print("\n" + "▓" * 60)
    print("  KIZGIN BUHAR ANALİZİ")
    print("▓" * 60)
    all_target_results = {}
    for target in SUPERHEATED_OUTPUTS:
        all_target_results[target] = run_hybrid_scenarios(df, SUPERHEATED_INPUTS, target)
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
    cols = ['Target', 'Scenario', 'Algorithm', 'Test_R2','Test_RMSE', 'Test_MAE', 'Test_MAPE']
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
    metrics = ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAE', 'Test_MAE', 'Train_MAPE', 'Test_MAPE']
    index_cols = [c for c in ['Dataset', 'Veri Seti', 'Target', 'Algorithm'] if c in df_long.columns]
    
    pivot = df_long.pivot_table(index=index_cols, columns='Scenario', values=metrics)
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
#  STGP‑EF SCENARIO-BASED GAIN ANALYSIS (BASE vs STGP-EF)
# ═══════════════════════════════════════════════════════════════════

def analyze_stgp_ef_gains(results_df):
    """
    Base ve STGP-EF senaryoları için:
      - Her (Target, Algorithm) çifti bazında Test_R2 kazanımı (STGP-EF - Base)
      - Senaryo düzeyinde toplam kazanım sayısı ve ortalama katkı
      - Algoritma ve hedef (Target) düzeyinde ortalama katkı
    analiz edilir.

    Beklenti:
      results_df, build_results_table(...) çıktısı gibi uzun formatta olsun
      ve en azından şu sütunları içersin:
        ['Target', 'Scenario', 'Algorithm', 'Test_R2'].
    """

    # Sadece Base ve STGP-EF senaryoları dikkate alınır
    sub = results_df[results_df['Scenario'].isin(['Base', 'STGP-EF'])].copy()
    if sub.empty:
        raise ValueError("results_df içerisinde 'Base' ve 'STGP-EF' senaryoları bulunamadı.")

    # (Target, Algorithm) düzeyinde geniş tablo: Base_Test_R2, STGP-EF_Test_R2
    pivot = sub.pivot_table(
        index=['Target', 'Algorithm'],
        columns='Scenario',
        values='Test_R2'
    )

    # Sütun isimlerini daha açık hale getirelim
    pivot = pivot.rename(columns={'Base': 'Base_Test_R2', 'STGP-EF': 'STGPEF_Test_R2'})

    # Kazanım (delta R2)
    pivot['Delta_Test_R2'] = pivot['STGPEF_Test_R2'] - pivot['Base_Test_R2']
    pivot['Gain_Flag'] = pivot['Delta_Test_R2'] > 0

    # Genel senaryo özeti
    total_cases = len(pivot)
    total_gains = int(pivot['Gain_Flag'].sum())
    gain_rate = total_gains / total_cases if total_cases > 0 else np.nan
    mean_delta = pivot['Delta_Test_R2'].mean()

    scenario_summary = {
        'total_cases': total_cases,
        'total_gains': total_gains,
        'gain_rate': gain_rate,
        'mean_delta_Test_R2': mean_delta
    }

    # Algoritma bazlı özet
    algo_summary = (
        pivot
        .reset_index()
        .groupby('Algorithm')['Delta_Test_R2']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'Mean_Delta_Test_R2', 'count': 'N'})
        .reset_index()
    )

    # Hedef (çıktı) bazlı özet
    target_summary_df = (
        pivot
        .reset_index()
        .groupby('Target')['Delta_Test_R2']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'Mean_Delta_Test_R2', 'count': 'N'})
        .reset_index()
    )

    return {
        'pivot': pivot.reset_index(),          # Her Target-Algorithm için Base vs STGP-EF karşılaştırması
        'scenario_summary': scenario_summary,  # Genel kazanım istatistikleri
        'algo_summary': algo_summary,          # Algoritma bazlı ortalama katkılar
        'target_summary': target_summary_df    # Hedef bazlı ortalama katkılar
    }


# ═══════════════════════════════════════════════════════════════════
#  HYBRID FEATURE CONTRIBUTION ANALYSIS (BASE vs HYBRID + IMPORTANCE)
# ═══════════════════════════════════════════════════════════════════

from sklearn.inspection import permutation_importance

def analyze_hybrid_feature_contributions(
    df,
    input_cols,
    target_col,
    base_model_name='XGBoost',
    n_repeats=10,
    random_state=42
):
    """
    Belirli bir hedef değişken için, seçili bir regresyon algoritması kullanılarak:
      - Sadece orijinal girdilerle (Base) elde edilen performans
      - STGP-EF ile genişletilmiş hibrit öznitelik uzayında elde edilen performans
      - Hibrit modeldeki tüm özniteliklerin (base + hybrid) permutation importance değerleri
    hesaplanır.

    Çıktı:
      {
        'meta': {...},
        'metrics': pd.DataFrame,      # Base vs Hybrid metrik karşılaştırması
        'importances': pd.DataFrame   # Tüm öznitelikler için önem istatistikleri
      }
    """

    # 1) Base veri setini hazırla
    X_train, X_test, y_train, y_test, _, _ = prepare_data(df, input_cols, target_col)

    regressors = get_regressors()
    if base_model_name not in regressors:
        raise ValueError(f"{base_model_name} get_regressors sözlüğünde tanımlı değil.")

    # 2) Base model eğitimi ve metrikler
    base_model = regressors[base_model_name]
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        base_model.fit(X_train, y_train)
        y_train_pred_base = base_model.predict(X_train)
        y_test_pred_base  = base_model.predict(X_test)

    base_metrics = {
        'Train_R2':   r2_score(y_train, y_train_pred_base),
        'Test_R2':    r2_score(y_test,  y_test_pred_base),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred_base)),
        'Test_RMSE':  np.sqrt(mean_squared_error(y_test,  y_test_pred_base)),
        'Train_MAE':  mean_absolute_error(y_train, y_train_pred_base),
        'Test_MAE':   mean_absolute_error(y_test,  y_test_pred_base),
    }

    # 3) STGP-EF ile hibrit öznitelik uzayını oluştur
    X_train_hybrid, X_test_hybrid = apply_stgp_ef(X_train, y_train, X_test)
    n_base = X_train.shape[1]
    n_hybrid = X_train_hybrid.shape[1]
    n_new = n_hybrid - n_base

    # Eğer yeni öznitelik üretilmemişse, base çıktısını tekrar ver
    if n_new <= 0:
        metrics_df = pd.DataFrame([
            {'Scenario': 'Base',   **base_metrics},
            {'Scenario': 'Hybrid', **base_metrics}
        ])
        imp_df = pd.DataFrame(columns=['feature', 'type', 'importance_mean', 'importance_std'])
        return {
            'meta': {
                'target': target_col,
                'input_cols': input_cols,
                'base_model': base_model_name,
                'n_base_features': n_base,
                'n_new_features': 0
            },
            'metrics': metrics_df,
            'importances': imp_df
        }

    # 4) Hibrit model eğitimi ve metrikler
    hybrid_model = regressors[base_model_name]
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        hybrid_model.fit(X_train_hybrid, y_train)
        y_train_pred_hybrid = hybrid_model.predict(X_train_hybrid)
        y_test_pred_hybrid  = hybrid_model.predict(X_test_hybrid)

    hybrid_metrics = {
        'Train_R2':   r2_score(y_train, y_train_pred_hybrid),
        'Test_R2':    r2_score(y_test,  y_test_pred_hybrid),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred_hybrid)),
        'Test_RMSE':  np.sqrt(mean_squared_error(y_test,  y_test_pred_hybrid)),
        'Train_MAE':  mean_absolute_error(y_train, y_train_pred_hybrid),
        'Test_MAE':   mean_absolute_error(y_test,  y_test_pred_hybrid),
    }

    # 5) Permutation importance (hibrit model için, tüm öznitelikler)
    pi = permutation_importance(
        hybrid_model,
        X_test_hybrid,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    feature_names = [f"x{i}" for i in range(n_base)] + [f"h{i}" for i in range(n_new)]
    feature_types = ['base'] * n_base + ['hybrid'] * n_new

    imp_df = pd.DataFrame({
        'feature': feature_names,
        'type': feature_types,
        'importance_mean': pi.importances_mean,
        'importance_std':  pi.importances_std
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

    # 6) Base vs Hybrid metrik tablosu
    metrics_df = pd.DataFrame([
        {'Scenario': 'Base',   **base_metrics},
        {'Scenario': 'Hybrid', **hybrid_metrics}
    ])

    return {
        'meta': {
            'target': target_col,
            'input_cols': input_cols,
            'base_model': base_model_name,
            'n_base_features': n_base,
            'n_new_features': n_new
        },
        'metrics': metrics_df,
        'importances': imp_df
    }


def summarize_hybrid_contributions_for_all_targets(
    df,
    inputs_list,
    outputs_list,
    base_model_name='XGBoost'
):
    """
    Belirtilen tüm hedef değişkenler için:
      - Base vs Hybrid metriklerini
      - Hibrit öznitelik önemlerini
    birleştirerek toplu özet üretir.

    Çıktı:
      metrics_all      : pd.DataFrame (Target x Scenario bazında metrikler)
      importances_all  : pd.DataFrame (Target x feature bazında importance)
    """
    metrics_rows = []
    imp_rows = []

    for target in outputs_list:
        res = analyze_hybrid_feature_contributions(
            df, inputs_list, target, base_model_name=base_model_name
        )

        m = res['metrics'].copy()
        m['Target'] = target
        m['Base_Model'] = base_model_name
        metrics_rows.append(m)

        imp = res['importances'].copy()
        imp['Target'] = target
        imp['Base_Model'] = base_model_name
        imp_rows.append(imp)

    metrics_all = pd.concat(metrics_rows, ignore_index=True)
    importances_all = pd.concat(imp_rows, ignore_index=True)

    return metrics_all, importances_all


# ═══════════════════════════════════════════════════════════════════
#  THERMODYNAMIC CONSISTENCY CHECKS
# ═══════════════════════════════════════════════════════════════════

def check_saturated_pointwise_constraints(df):
    """
    Doymuş faz için satır bazlı eşitsizlik kontrolleri.

    Varsa kontrol edilen sütun çiftleri:
      - v buhar (çıktı)  > v sıvı (çıktı)
      - h buhar (çıktı)  > h sıvı (çıktı)
      - s buhar (çıktı)  > s sıvı (çıktı)

    Çıktı:
      summary    : {kural_adı: {n_total, n_violations, violation_rate}}
      violations : ihlalli satırların birleştirilmiş DataFrame'i
    """
    rules = []

    if {'v buhar (çıktı)', 'v sıvı (çıktı)'}.issubset(df.columns):
        rules.append(('v_g > v_f', lambda r: r['v buhar (çıktı)'] > r['v sıvı (çıktı)']))

    if {'h buhar (çıktı)', 'h sıvı (çıktı)'}.issubset(df.columns):
        rules.append(('h_g > h_f', lambda r: r['h buhar (çıktı)'] > r['h sıvı (çıktı)']))

    if {'s buhar (çıktı)', 's sıvı (çıktı)'}.issubset(df.columns):
        rules.append(('s_g > s_f', lambda r: r['s buhar (çıktı)'] > r['s sıvı (çıktı)']))

    violations_list = []
    summary = {}

    for name, rule in rules:
        mask_ok = df.apply(rule, axis=1)
        n_total = len(df)
        n_ok = mask_ok.sum()
        n_bad = n_total - n_ok
        frac_bad = n_bad / n_total if n_total > 0 else np.nan

        summary[name] = {
            'n_total': n_total,
            'n_violations': int(n_bad),
            'violation_rate': frac_bad
        }

        if n_bad > 0:
            viol_df = df.loc[~mask_ok].copy()
            viol_df['violated_rule'] = name
            violations_list.append(viol_df)

    violations = pd.concat(violations_list, ignore_index=True) if violations_list else pd.DataFrame()
    return summary, violations


def check_saturated_monotonicity(df, T_col='T(girdi)', P_col='P(çıktı)'):
    """
    Doymuş faz için örnek monotonluk kontrolleri:
      - P(T): sıcaklık arttıkça doymuş basınç artar.
      - İsteğe bağlı olarak: h_sıvı(T), h_buhar(T), s_sıvı(T), s_buhar(T) için artanlık.

    Çıktı:
      summary : {büyüklük_adı: {n_points, n_violations, violation_rate}}
    """
    summary = {}
    if T_col not in df.columns:
        return summary

    df_sorted = df.sort_values(T_col)
    dT = np.diff(df_sorted[T_col].values)

    def _monotone_check(y):
        if len(y) < 2:
            return {'n_points': len(y), 'n_violations': np.nan, 'violation_rate': np.nan}
        dy = np.diff(y)
        mask = dT > 0
        dy = dy[mask]
        n = len(dy)
        if n == 0:
            return {'n_points': len(y), 'n_violations': np.nan, 'violation_rate': np.nan}
        n_bad = (dy < 0).sum()
        return {'n_points': len(y), 'n_violations': int(n_bad), 'violation_rate': n_bad / n}

    # P(T) monoton artan olmalı
    if P_col in df_sorted.columns:
        summary['P_vs_T_increasing'] = _monotone_check(df_sorted[P_col].values)

    for col in ['h sıvı (çıktı)', 'h buhar (çıktı)', 's sıvı (çıktı)', 's buhar (çıktı)']:
        if col in df_sorted.columns:
            key = f"{col}_vs_T_increasing"
            summary[key] = _monotone_check(df_sorted[col].values)

    return summary


def check_superheated_monotonicity(df, T_col='T (girdi)', P_col='P (girdi)'):
    """
    Kızgın faz (superheated) için tipik monotonluk kontrolleri:

      - Sabit P boyunca:
          v (çıktı)(T), h (çıktı)(T), s (çıktı)(T) artan olmalıdır.
      - Sabit T boyunca:
          v (çıktı)(P) genel olarak azalan olmalıdır (aynı sıcaklıkta basınç artarken özgül hacim azalır).

    Çıktı:
      summary : {kontrol_adı: ortalama_ihlâl_oranı}
    """
    summary = {}

    if not {T_col, P_col}.issubset(df.columns):
        return summary

    # Sabit P için T'ye göre artış (v, h, s)
    for prop in ['v (çıktı)', 'h (çıktı)', 's (çıktı)']:
        if prop not in df.columns:
            continue

        rates = []
        for p_val, sub in df.groupby(P_col):
            sub = sub.sort_values(T_col)
            if len(sub) < 3:
                continue

            dT = np.diff(sub[T_col].values)
            dy = np.diff(sub[prop].values)
            mask = dT > 0
            dy = dy[mask]
            if len(dy) == 0:
                continue

            n_bad = (dy < 0).sum()  # T artarken büyüklük azalıyorsa ihlal
            rates.append(n_bad / len(dy))

        key = f"{prop}_vs_T_at_constP_increasing"
        summary[key] = np.mean(rates) if rates else np.nan

    # Sabit T için P'ye göre azalış (v)
    prop = 'v (çıktı)'
    if prop in df.columns:
        rates = []
        for T_val, sub in df.groupby(T_col):
            sub = sub.sort_values(P_col)
            if len(sub) < 3:
                continue

            dP = np.diff(sub[P_col].values)
            dv = np.diff(sub[prop].values)
            mask = dP > 0
            dv = dv[mask]
            if len(dv) == 0:
                continue

            n_bad = (dv > 0).sum()  # P artarken v artıyorsa ihlal
            rates.append(n_bad / len(dv))

        key = f"{prop}_vs_P_at_constT_decreasing"
        summary[key] = np.mean(rates) if rates else np.nan

    return summary
