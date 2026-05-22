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

from contextlib import redirect_stdout

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
    # Doymuş faz için (Sadece T girdisi varsa) uzay genişletilir
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
    except Exception as e:
        logger.error(f"STGP başarısız: {e}")
    
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
    except Exception as e:
        logger.error(f"EF başarısız: {e}")

    n_ef = min(n_best_features, X_ef.shape[1])
    X_ef = X_ef[:, :n_ef] if n_ef > 0 else X_ef

    X_constructed = np.hstack((X_stgp, X_ef)) if X_stgp.size and X_ef.size else np.empty((X.shape[0], 0))
    X_hybrid = np.hstack((X, X_constructed)) if X_constructed.size else X

    return X_hybrid, stgp_forms, ef_forms


# ═══════════════════════════════════════════════════════════════════
#  UNIFIED MASTER LOOP (EXCEL EXPORT)
# ═══════════════════════════════════════════════════════════════════
def run_unified_analysis(df, input_cols, outputs_list, dataset_name):
    """
    KESİN SIRALI MİMARİ:
    Faz 1: Her hedef değişken için özel STGP-EF uygulanır, veri setleri oluşturulur ve hafızaya alınır.
    Faz 2: Base senaryo (sadece orijinal veri) tüm hedefler ve foldlar için eğitilir.
    Faz 3: STGP-EF senaryosu tüm hedefler ve foldlar için eğitilir, katkı (Importance) hesaplanır.
    Faz 4: Excel'e kaydetme.
    """
    print(f"\n{'▓'*60}")
    print(f"  {dataset_name.upper()} - KESİN SIRALI K-FOLD ANALİZİ VE EXCEL RAPORLAMA")
    print(f"{'▓'*60}")
    
    os.makedirs("Results", exist_ok=True)
    epsilon = 1e-10
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    regressors = get_regressors()
    
    # Tüm sonuçları ve veri setlerini tutacağımız sözlükler
    master_datasets = {}
    dict_formulas = []
    all_metrics = []
    all_importances = []
    
    # =========================================================================
    # FAZ 1: HEDEF BAZLI VERİ SETLERİNİN HAZIRLANMASI (PRE-GENERATION)
    # =========================================================================
    print("\n[FAZ 1/3] Her hedef değişken için ayrı ayrı hibrit veri setleri üretiliyor...")
    
    df_enr, enr_cols = enrich_input_space(df, input_cols)
    X_base = df[input_cols].values.astype(np.float64) # Sadece orijinal girdi (Base için)
    X_enr = df_enr[enr_cols].values.astype(np.float64) # Orijinal + Genişletilmiş girdi (STGP-EF'in kullanacağı)
    
    for target_col in outputs_list:
        print(f"  -> Hedef ({target_col}) için STGP-EF çalıştırılıyor...")
        y = df_enr[target_col].values.astype(np.float64)
        
        # Sadece bu hedefe özel STGP-EF özellik inşası
        X_hyb, stgp_forms, ef_forms = apply_stgp_ef_full(X_enr, y, n_best_features=10)
        
        n_base_hyb = len(enr_cols)
        n_hyb_feat = X_hyb.shape[1] - n_base_hyb
        hyb_names = list(stgp_forms.keys()) + list(ef_forms.keys())
        hyb_names = hyb_names[:n_hyb_feat]
        
        feature_names = enr_cols + hyb_names
        feature_types = ['Orijinal'] * n_base_hyb + ['Hibrit'] * n_hyb_feat
        
        # DataFrame olarak saklama (Excel çıktısı için)
        df_hyb_dataset = pd.DataFrame(X_hyb, columns=feature_names)
        df_hyb_dataset['Target'] = y
        
        # İlgili hedefin verilerini Master Sözlüğe kaydet
        master_datasets[target_col] = {
            'y': y,
            'X_hyb': X_hyb,
            'feature_names': feature_names,
            'feature_types': feature_types,
            'df_export': df_hyb_dataset
        }
        
        for k, v in stgp_forms.items(): dict_formulas.append({'Target': target_col, 'Oznitelik': k, 'Algoritma': 'STGP', 'Formul': v})
        for k, v in ef_forms.items(): dict_formulas.append({'Target': target_col, 'Oznitelik': k, 'Algoritma': 'EF', 'Formul': v})


    # =========================================================================
    # FAZ 2: BASE SENARYO EĞİTİMLERİ (TÜM HEDEFLER)
    # =========================================================================
    print("\n[FAZ 2/3] Orijinal (Base) veri setleri ile modeller eğitiliyor...")
    
    for target_col in outputs_list:
        y = master_datasets[target_col]['y']
        
        for algo_name, model in regressors.items():
                print(f"      -> {algo_name} eğitiliyor (Base)...") # YENİ EKLENEN SATIR
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
                

    # =========================================================================
    # FAZ 3: STGP-EF SENARYO EĞİTİMLERİ VE ÖZNİTELİK KATKISI (TÜM HEDEFLER)
    # =========================================================================
    print("\n[FAZ 3/3] STGP-EF (Hibrit) veri setleri ile modeller eğitiliyor ve katkılar hesaplanıyor...")
    
    for target_col in outputs_list:
        y = master_datasets[target_col]['y']
        X_hyb = master_datasets[target_col]['X_hyb']
        f_names = master_datasets[target_col]['feature_names']
        f_types = master_datasets[target_col]['feature_types']
        
        for algo_name, model in regressors.items():
            print(f"      -> {algo_name} eğitiliyor (STGP-EF & Importance)...") # YENİ EKLENEN SATIR
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_hyb)):
                X_tr_h, X_te_h, y_tr_h, y_te_h = prepare_data_fold(X_hyb, y, train_idx, test_idx)
                
                # Model Eğitimi ve Performans Metrikleri
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
                
                # Bireysel Öznitelik Katkısı (Permutation Importance)
                with open(os.devnull, 'w') as f, redirect_stdout(f):
                    pi = permutation_importance(model, X_te_h, y_te_h, n_repeats=5, random_state=42, n_jobs=-1)
                
                for i, fname in enumerate(f_names):
                    all_importances.append({
                        'Target': target_col, 'Algorithm': algo_name, 'Feature': fname, 'Type': f_types[i],
                        'Importance_Mean': pi.importances_mean[i], 'Importance_Std': pi.importances_std[i], 'Fold': fold+1
                    })

    # =========================================================================
    # FAZ 4: ORTALAMALARIN HESAPLANMASI VE EXCEL ÇIKTISI
    # =========================================================================
    print("\n[TAMAMLANDI] Sonuçlar toparlanıp Excel dosyasına yazılıyor...")
    
    df_met = pd.DataFrame(all_metrics)
    avg_met = df_met.groupby(['Target', 'Algorithm', 'Scenario']).mean(numeric_only=True).drop(columns=['Fold']).reset_index()
    
    df_imp = pd.DataFrame(all_importances)
    avg_imp = df_imp.groupby(['Target', 'Algorithm', 'Feature', 'Type']).mean(numeric_only=True).drop(columns=['Fold']).reset_index()
    avg_imp = avg_imp.sort_values(by=['Target', 'Algorithm', 'Importance_Mean'], ascending=[True, True, False])

    excel_path = f"Results/{dataset_name}_Nihai_Sonuclar.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        avg_met.to_excel(writer, sheet_name='Performans_Ortalamalari', index=False)
        avg_imp.to_excel(writer, sheet_name='Oznitelik_Bireysel_Katki', index=False)
        pd.DataFrame(dict_formulas).to_excel(writer, sheet_name='Uretilen_Formuller', index=False)
        
        # Her hedefe özel üretilen hibrit veri seti ayrı sekme olarak kaydedilir
        for t_name, data_info in master_datasets.items():
            safe_name = str(t_name).replace('(', '').replace(')', '').replace(' ', '_')[:25] # Excel sekme limiti güvenliği
            data_info['df_export'].to_excel(writer, sheet_name=f'Data_{safe_name}', index=False)

    print(f"-> Çıktı Dosyası: {excel_path}")
    
    # Notebook için primary çıktıyı döndür (Görselleştirme için listelerin ilk hedefini alır)
    primary_target = outputs_list[0]
    return avg_met, avg_imp, master_datasets[primary_target]['df_export']


# ═══════════════════════════════════════════════════════════════════
#  THERMODYNAMIC CONSISTENCY CHECKS
# ═══════════════════════════════════════════════════════════════════
def check_saturated_pointwise_constraints(df):
    rules = []
    if {'v buhar (çıktı)', 'v sıvı (çıktı)'}.issubset(df.columns): rules.append(('v_g > v_f', lambda r: r['v buhar (çıktı)'] > r['v sıvı (çıktı)']))
    if {'h buhar (çıktı)', 'h sıvı (çıktı)'}.issubset(df.columns): rules.append(('h_g > h_f', lambda r: r['h buhar (çıktı)'] > r['h sıvı (çıktı)']))
    if {'s buhar (çıktı)', 's sıvı (çıktı)'}.issubset(df.columns): rules.append(('s_g > s_f', lambda r: r['s buhar (çıktı)'] > r['s sıvı (çıktı)']))

    violations_list = []
    summary = {}
    for name, rule in rules:
        mask_ok = df.apply(rule, axis=1)
        n_total = len(df)
        n_bad = n_total - mask_ok.sum()
        summary[name] = {'n_total': n_total, 'n_violations': int(n_bad), 'violation_rate': n_bad / n_total if n_total > 0 else np.nan}
        if n_bad > 0:
            viol_df = df.loc[~mask_ok].copy()
            viol_df['violated_rule'] = name
            violations_list.append(viol_df)
    return summary, pd.concat(violations_list, ignore_index=True) if violations_list else pd.DataFrame()

def check_saturated_monotonicity(df, T_col='T(girdi)', P_col='P(çıktı)'):
    summary = {}
    if T_col not in df.columns: return summary
    df_sorted = df.sort_values(T_col)
    dT = np.diff(df_sorted[T_col].values)

    def _monotone_check(y):
        if len(y) < 2: return {'n_points': len(y), 'n_violations': np.nan, 'violation_rate': np.nan}
        dy = np.diff(y)[dT > 0]
        if len(dy) == 0: return {'n_points': len(y), 'n_violations': np.nan, 'violation_rate': np.nan}
        n_bad = (dy < 0).sum()
        return {'n_points': len(y), 'n_violations': int(n_bad), 'violation_rate': n_bad / len(dy)}

    if P_col in df_sorted.columns: summary['P_vs_T_increasing'] = _monotone_check(df_sorted[P_col].values)
    for col in ['h sıvı (çıktı)', 'h buhar (çıktı)', 's sıvı (çıktı)', 's buhar (çıktı)']:
        if col in df_sorted.columns: summary[f"{col}_vs_T_increasing"] = _monotone_check(df_sorted[col].values)
    return summary

def check_superheated_monotonicity(df, T_col='T (girdi)', P_col='P (girdi)'):
    summary = {}
    if not {T_col, P_col}.issubset(df.columns): return summary

    for prop in ['v (çıktı)', 'h (çıktı)', 's (çıktı)']:
        if prop not in df.columns: continue
        rates = []
        for p_val, sub in df.groupby(P_col):
            sub = sub.sort_values(T_col)
            if len(sub) < 3: continue
            dy = np.diff(sub[prop].values)[np.diff(sub[T_col].values) > 0]
            if len(dy) > 0: rates.append((dy < 0).sum() / len(dy))
        summary[f"{prop}_vs_T_at_constP_increasing"] = np.mean(rates) if rates else np.nan

    prop = 'v (çıktı)'
    if prop in df.columns:
        rates = []
        for T_val, sub in df.groupby(T_col):
            sub = sub.sort_values(P_col)
            if len(sub) < 3: continue
            dv = np.diff(sub[prop].values)[np.diff(sub[P_col].values) > 0]
            if len(dv) > 0: rates.append((dv > 0).sum() / len(dv))
        summary[f"{prop}_vs_P_at_constT_decreasing"] = np.mean(rates) if rates else np.nan
    return summary

def check_thermodynamics_on_synthetic_grid(models_dict, input_cols, min_T=-30, max_T=50, step=0.1, const_P=None):
    T_range = np.arange(min_T, max_T, step)
    grid_df = pd.DataFrame({'T (girdi)': T_range, 'P (girdi)': const_P}) if const_P is not None else pd.DataFrame({'T(girdi)': T_range})
        
    grid_enr, enr_cols = enrich_input_space(grid_df, input_cols)
    X_grid = grid_enr[enr_cols].values
    
    results = []
    for algo_name, model in models_dict.items():
        try:
            preds = model.predict(X_grid)
            derivatives = np.diff(preds) / np.diff(T_range)
            violations = (derivatives < 0).sum()
            results.append({
                'Algoritma': algo_name, 'Sentetik_Nokta_Sayisi': len(preds),
                'Fiziksel_Ihlal_Sayisi': violations, 'Ihlal_Yuzdesi_(%)': (violations / len(derivatives)) * 100
            })
        except Exception as e:
            logger.warning(f"{algo_name} grid testi başarısız: {e}")
            
    df_results = pd.DataFrame(results)
    return df_results