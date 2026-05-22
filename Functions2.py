# ═══════════════════════════════════════════════════════════════════
#  IMPORT SECTION
# ═══════════════════════════════════════════════════════════════════
import warnings
import os
import numpy as np
import pandas as pd
import logging
import re
from contextlib import redirect_stdout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sklearn imports
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
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
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════
def enrich_input_space(df, input_cols):
    df_enriched = df.copy()
    new_cols = list(input_cols)
    if 'T(girdi)' in input_cols and len(input_cols) == 1:
        T_K = df_enriched['T(girdi)'] + 273.15 
        df_enriched['1/T'] = 1.0 / T_K
        df_enriched['ln(T)'] = np.log(T_K)
        new_cols.extend(['1/T', 'ln(T)'])
    return df_enriched, new_cols

def prepare_data_fold(X, y, train_idx, test_idx):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    return X_train, X_test, y_train, y_test

# ═══════════════════════════════════════════════════════════════════
#  FEATURE CONSTRUCTION (STGP-EF)
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

def apply_stgp_ef_full(X, y, n_best_features=10):
    stgp_forms, ef_forms = {}, {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    stgp_model, ef_model = None, None
    
    # STGP
    X_stgp = np.empty((X_scaled.shape[0], 0))
    try:
        stgp_model = SymbolicTransformer(n_jobs=1, random_state=42)
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            stgp_model.fit(X_scaled, y)
            X_stgp = np.nan_to_num(stgp_model.transform(X_scaled))
            if hasattr(stgp_model, '_best_programs'):
                for idx, prog in enumerate(stgp_model._best_programs[:n_best_features]):
                    stgp_forms[f'STGP_{idx:02d}'] = format_math_expr(str(prog))
    except Exception as e: pass
    
    n_stgp = min(n_best_features, X_stgp.shape[1])
    X_stgp = X_stgp[:, :n_stgp] if n_stgp > 0 else X_stgp

    # EF
    X_ef = np.empty((X_scaled.shape[0], 0))
    try:
        ef_model = EvolutionaryForestRegressor(random_state=42, basic_primitives="default", verbose=False, n_process=1)
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            ef_model.fit(X_scaled, y)
            X_ef = ef_model.transform(X_scaled)
            hof = getattr(ef_model, '_best_hof', getattr(ef_model, 'hof', None))
            if hof is not None:
                for idx, prog in enumerate(hof[:n_best_features]):
                    ef_forms[f'EF_{idx:02d}'] = format_math_expr(str(prog))
    except Exception as e: pass

    n_ef = min(n_best_features, X_ef.shape[1])
    X_ef = X_ef[:, :n_ef] if n_ef > 0 else X_ef

    X_constructed = np.hstack((X_stgp, X_ef)) if X_stgp.size and X_ef.size else np.empty((X.shape[0], 0))
    X_hybrid = np.hstack((X, X_constructed)) if X_constructed.size else X

    return X_hybrid, stgp_forms, ef_forms, stgp_model, ef_model, scaler

# ═══════════════════════════════════════════════════════════════════
#  THERMODYNAMIC GRID TEST (UNIVERSAL PIML)
# ═══════════════════════════════════════════════════════════════════
def check_thermodynamics_on_synthetic_grid(models_dict, input_cols, target_name, 
                                           vary_col='T', min_val=-30, max_val=50, step=0.1, 
                                           fixed_col=None, fixed_val=None,
                                           stgp_model=None, ef_model=None, base_scaler=None, n_best=10):
    
    grid_vals = np.arange(min_val, max_val, step)
    grid_data = {}
    
    t_col_name = 'T (girdi)' if 'T (girdi)' in input_cols else 'T(girdi)'
    p_col_name = 'P (girdi)' if 'P (girdi)' in input_cols else None
    
    if vary_col == 'T':
        grid_data[t_col_name] = grid_vals
        if fixed_col == 'P' and p_col_name: grid_data[p_col_name] = fixed_val
    elif vary_col == 'P':
        grid_data[p_col_name] = grid_vals
        if fixed_col == 'T' and t_col_name: grid_data[t_col_name] = fixed_val
            
    grid_df = pd.DataFrame(grid_data)
    grid_df = grid_df[input_cols] # Sütun sırasını eşitle
    
    grid_enr, enr_cols = enrich_input_space(grid_df, input_cols)
    X_grid_base = grid_enr[enr_cols].values
    
    # STGP-EF modelleri verilmişse (Hybrid Senaryo), sentetik matrisi zenginleştir
    if stgp_model is not None or ef_model is not None:
        X_scaled = base_scaler.transform(X_grid_base) if base_scaler else X_grid_base
        X_stgp = np.empty((X_scaled.shape[0], 0))
        if stgp_model:
            try:
                X_stgp = np.nan_to_num(stgp_model.transform(X_scaled))
                n_s = min(n_best, X_stgp.shape[1])
                X_stgp = X_stgp[:, :n_s] if n_s > 0 else X_stgp
            except: pass
        X_ef = np.empty((X_scaled.shape[0], 0))
        if ef_model:
            try:
                X_ef = ef_model.transform(X_scaled)
                n_e = min(n_best, X_ef.shape[1])
                X_ef = X_ef[:, :n_e] if n_e > 0 else X_ef
            except: pass
        X_const = np.hstack((X_stgp, X_ef)) if X_stgp.size and X_ef.size else np.empty((X_grid_base.shape[0], 0))
        X_grid_final = np.hstack((X_grid_base, X_const)) if X_const.size else X_grid_base
    else:
        X_grid_final = X_grid_base # Base Senaryo

    # FİZİKSEL KURAL MOTORU (Kızgın ve Doymuş Fazlara Dinamik Tepki Verir)
    is_decreasing = False
    if vary_col == 'T' and target_name in ['v buhar (çıktı)', 's buhar (çıktı)']: is_decreasing = True
    elif vary_col == 'P' and target_name in ['v (çıktı)', 's (çıktı)']: is_decreasing = True
    
    results = []
    for algo_name, model in models_dict.items():
        try:
            preds = model.predict(X_grid_final)
            derivatives = np.diff(preds) / np.diff(grid_vals)
            
            if is_decreasing: violations = (derivatives > 0).sum()
            else: violations = (derivatives < 0).sum()
                
            violation_rate = (violations / len(derivatives)) * 100
            test_str = f"Sabit {fixed_col}={fixed_val}, {vary_col} degisiyor" if fixed_col else f"{vary_col} degisiyor (Doymus)"
            
            results.append({
                'Target': target_name, 'Algoritma': algo_name, 'Test_Tipi': test_str,
                'Fiziksel_Ihlal_Sayisi': violations, 'Ihlal_Yuzdesi_(%)': violation_rate
            })
        except: pass
    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════════
#  UNIFIED MASTER LOOP (FAZ 1 - 5)
# ═══════════════════════════════════════════════════════════════════
def run_unified_analysis(df, input_cols, outputs_list, dataset_name):
    print(f"\n{'▓'*60}")
    print(f"  {dataset_name.upper()} - KESİN SIRALI K-FOLD VE PIML ANALİZİ")
    print(f"{'▓'*60}")
    
    os.makedirs("Results", exist_ok=True)
    epsilon = 1e-10
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    regressors = get_regressors()
    
    master_datasets = {}
    dict_formulas = []
    all_metrics = []
    all_importances = []
    
    # -------------------------------------------------------------------------
    # FAZ 1: HEDEF BAZLI VERİ SETLERİNİN HAZIRLANMASI (STGP-EF)
    # -------------------------------------------------------------------------
    print("\n[FAZ 1/5] Her hedef değişken için ayrı ayrı hibrit veri setleri üretiliyor...")
    df_enr, enr_cols = enrich_input_space(df, input_cols)
    X_base = df[input_cols].values.astype(np.float64) 
    X_enr = df_enr[enr_cols].values.astype(np.float64) 
    
    for target_col in outputs_list:
        print(f"  -> Hedef ({target_col}) için STGP-EF çalıştırılıyor...")
        y = df_enr[target_col].values.astype(np.float64)
        X_hyb, stgp_forms, ef_forms, stgp_model, ef_model, base_scaler = apply_stgp_ef_full(X_enr, y, n_best_features=10)
        
        n_base_hyb = len(enr_cols)
        n_hyb_feat = X_hyb.shape[1] - n_base_hyb
        hyb_names = (list(stgp_forms.keys()) + list(ef_forms.keys()))[:n_hyb_feat]
        feature_names = enr_cols + hyb_names
        feature_types = ['Orijinal'] * n_base_hyb + ['Hibrit'] * n_hyb_feat
        
        df_hyb_dataset = pd.DataFrame(X_hyb, columns=feature_names)
        df_hyb_dataset['Target'] = y
        
        master_datasets[target_col] = {
            'y': y, 'X_hyb': X_hyb, 'feature_names': feature_names, 'feature_types': feature_types,
            'df_export': df_hyb_dataset, 'stgp_model': stgp_model, 'ef_model': ef_model, 'base_scaler': base_scaler
        }
        
        for k, v in stgp_forms.items(): dict_formulas.append({'Target': target_col, 'Oznitelik': k, 'Algoritma': 'STGP', 'Formul': v})
        for k, v in ef_forms.items(): dict_formulas.append({'Target': target_col, 'Oznitelik': k, 'Algoritma': 'EF', 'Formul': v})

    # -------------------------------------------------------------------------
    # FAZ 2: BASE SENARYO EĞİTİMLERİ (TÜM HEDEFLER)
    # -------------------------------------------------------------------------
    print("\n[FAZ 2/5] Orijinal (Base) veri setleri ile modeller K-Fold ile eğitiliyor...")
    for target_col in outputs_list:
        y = master_datasets[target_col]['y']
        for algo_name, model in regressors.items():
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_base)):
                X_tr, X_te, y_tr, y_te = prepare_data_fold(X_base, y, train_idx, test_idx)
                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    model.fit(X_tr, y_tr)
                    y_tr_pred = model.predict(X_tr)
                    y_te_pred = model.predict(X_te)
                
                all_metrics.append({
                    'Target': target_col, 'Algorithm': algo_name, 'Scenario': 'Base', 'Fold': fold+1,
                    'Train_R2': r2_score(y_tr, y_tr_pred), 'Test_R2': r2_score(y_te, y_te_pred),
                    'Train_RMSE': np.sqrt(mean_squared_error(y_tr, y_tr_pred)), 'Test_RMSE': np.sqrt(mean_squared_error(y_te, y_te_pred)),
                    'Train_MAE': mean_absolute_error(y_tr, y_tr_pred), 'Test_MAE': mean_absolute_error(y_te, y_te_pred),
                    'Train_MAPE': mean_absolute_percentage_error(y_tr+epsilon, y_tr_pred+epsilon), 'Test_MAPE': mean_absolute_percentage_error(y_te+epsilon, y_te_pred+epsilon)
                })

    # -------------------------------------------------------------------------
    # FAZ 3: STGP-EF SENARYO EĞİTİMLERİ VE ÖZNİTELİK KATKISI (TÜM HEDEFLER)
    # -------------------------------------------------------------------------
    print("\n[FAZ 3/5] STGP-EF (Hibrit) veri setleri ile modeller eğitiliyor ve katkılar hesaplanıyor...")
    for target_col in outputs_list:
        y = master_datasets[target_col]['y']
        X_hyb = master_datasets[target_col]['X_hyb']
        f_names = master_datasets[target_col]['feature_names']
        f_types = master_datasets[target_col]['feature_types']
        
        for algo_name, model in regressors.items():
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_hyb)):
                X_tr_h, X_te_h, y_tr_h, y_te_h = prepare_data_fold(X_hyb, y, train_idx, test_idx)
                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    model.fit(X_tr_h, y_tr_h)
                    y_tr_pred_h = model.predict(X_tr_h)
                    y_te_pred_h = model.predict(X_te_h)
                
                all_metrics.append({
                    'Target': target_col, 'Algorithm': algo_name, 'Scenario': 'Hybrid', 'Fold': fold+1,
                    'Train_R2': r2_score(y_tr_h, y_tr_pred_h), 'Test_R2': r2_score(y_te_h, y_te_pred_h),
                    'Train_RMSE': np.sqrt(mean_squared_error(y_tr_h, y_tr_pred_h)), 'Test_RMSE': np.sqrt(mean_squared_error(y_te_h, y_te_pred_h)),
                    'Train_MAE': mean_absolute_error(y_tr_h, y_tr_pred_h), 'Test_MAE': mean_absolute_error(y_te_h, y_te_pred_h),
                    'Train_MAPE': mean_absolute_percentage_error(y_tr_h+epsilon, y_tr_pred_h+epsilon), 'Test_MAPE': mean_absolute_percentage_error(y_te_h+epsilon, y_te_pred_h+epsilon)
                })
                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    pi = permutation_importance(model, X_te_h, y_te_h, n_repeats=5, random_state=42, n_jobs=-1)
                
                for i, fname in enumerate(f_names):
                    all_importances.append({
                        'Target': target_col, 'Algorithm': algo_name, 'Feature': fname, 'Type': f_types[i],
                        'Importance_Mean': pi.importances_mean[i], 'Importance_Std': pi.importances_std[i], 'Fold': fold+1
                    })

    # -------------------------------------------------------------------------
    # FAZ 4: TERMODİNAMİK UYUM (PIML - SENTETİK IZGARA) TESTİ
    # -------------------------------------------------------------------------
    print("\n[FAZ 4/5] Modellerin Termodinamik Yasalara Uyumu (Sentetik Izgara Testi) yapılıyor...")
    piml_results = []
    is_saturated = (len(input_cols) == 1 and 'T(girdi)' in input_cols)

    for target_col in outputs_list:
        data_info = master_datasets[target_col]
        y = data_info['y']
        X_hyb = data_info['X_hyb']
        
        trained_base, trained_hyb = {}, {}
        for algo_name, b_model in get_regressors().items():
            pipe_base = Pipeline([('scaler', StandardScaler()), ('model', b_model)])
            pipe_base.fit(X_base, y)
            trained_base[algo_name] = pipe_base
            
        for algo_name, h_model in get_regressors().items():
            pipe_hyb = Pipeline([('scaler', StandardScaler()), ('model', h_model)])
            pipe_hyb.fit(X_hyb, y)
            trained_hyb[algo_name] = pipe_hyb

        if is_saturated:
            res_b = check_thermodynamics_on_synthetic_grid(trained_base, input_cols, target_col, vary_col='T', min_val=-30, max_val=50, step=0.1)
            res_b['Scenario'] = 'Base'
            res_h = check_thermodynamics_on_synthetic_grid(trained_hyb, input_cols, target_col, vary_col='T', min_val=-30, max_val=50, step=0.1, stgp_model=data_info['stgp_model'], ef_model=data_info['ef_model'], base_scaler=data_info['base_scaler'])
            res_h['Scenario'] = 'Hybrid'
            piml_results.extend([res_b, res_h])
        else:
            # Kızgın Faz: Izobarik (Sabit P=1.5, T Değişir)
            res_b_t = check_thermodynamics_on_synthetic_grid(trained_base, input_cols, target_col, vary_col='T', min_val=20, max_val=100, step=0.1, fixed_col='P', fixed_val=1.5)
            res_b_t['Scenario'] = 'Base'
            res_h_t = check_thermodynamics_on_synthetic_grid(trained_hyb, input_cols, target_col, vary_col='T', min_val=20, max_val=100, step=0.1, fixed_col='P', fixed_val=1.5, stgp_model=data_info['stgp_model'], ef_model=data_info['ef_model'], base_scaler=data_info['base_scaler'])
            res_h_t['Scenario'] = 'Hybrid'
            # Kızgın Faz: Izotermal (Sabit T=40, P Değişir)
            res_b_p = check_thermodynamics_on_synthetic_grid(trained_base, input_cols, target_col, vary_col='P', min_val=0.5, max_val=3.0, step=0.01, fixed_col='T', fixed_val=40.0)
            res_b_p['Scenario'] = 'Base'
            res_h_p = check_thermodynamics_on_synthetic_grid(trained_hyb, input_cols, target_col, vary_col='P', min_val=0.5, max_val=3.0, step=0.01, fixed_col='T', fixed_val=40.0, stgp_model=data_info['stgp_model'], ef_model=data_info['ef_model'], base_scaler=data_info['base_scaler'])
            res_h_p['Scenario'] = 'Hybrid'
            piml_results.extend([res_b_t, res_h_t, res_b_p, res_h_p])

    # -------------------------------------------------------------------------
    # FAZ 5: ORTALAMALARIN HESAPLANMASI VE EXCEL ÇIKTISI
    # -------------------------------------------------------------------------
    print("\n[FAZ 5/5] Analiz tamamlandı. Tüm sonuçlar Excel dosyasına kaydediliyor...")
    
    df_met = pd.DataFrame(all_metrics).groupby(['Target', 'Algorithm', 'Scenario']).mean(numeric_only=True).drop(columns=['Fold']).reset_index()
    df_imp = pd.DataFrame(all_importances).groupby(['Target', 'Algorithm', 'Feature', 'Type']).mean(numeric_only=True).drop(columns=['Fold']).reset_index()
    df_imp = df_imp.sort_values(by=['Target', 'Algorithm', 'Importance_Mean'], ascending=[True, True, False])
    df_piml = pd.concat(piml_results, ignore_index=True)

    excel_path = f"Results/{dataset_name}_Nihai_Sonuclar.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_met.to_excel(writer, sheet_name='Performans_Ortalamalari', index=False)
        df_imp.to_excel(writer, sheet_name='Oznitelik_Bireysel_Katki', index=False)
        df_piml.to_excel(writer, sheet_name='Termodinamik_Uyum_PIML', index=False)
        pd.DataFrame(dict_formulas).to_excel(writer, sheet_name='Uretilen_Formuller', index=False)
        
        for t_name, data_info in master_datasets.items():
            safe_name = str(t_name).replace('(', '').replace(')', '').replace(' ', '_')[:25] 
            data_info['df_export'].to_excel(writer, sheet_name=f'Data_{safe_name}', index=False)

    print(f"-> Çıktı Dosyası Başarıyla Oluşturuldu: {excel_path}")
    return df_met, df_imp, df_piml