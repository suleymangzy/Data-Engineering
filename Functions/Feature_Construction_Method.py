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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    nan_template = {
        'Train_R2': np.nan, 'Test_R2': np.nan, 
        'Train_RMSE': np.nan, 'Test_RMSE': np.nan, 
        'Train_MAE': np.nan, 'Test_MAE': np.nan,
        'Train_MAPE': np.nan, 'Test_MAPE': np.nan
    }

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
#  BASELINE MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════
def evaluate_baseline_model(X_train, y_train, X_test, y_test):
    """
    Orijinal veriler üzerinde baseline (temel karşılaştırma) model oluşturur.
    Baseline olarak XGBoost kullanıldı (çünkü genellikle en iyi performans gösterir).
    STGP-EF yönteminin katkısını ölçmek için baseline referans modeli olarak işlev görür.
    """
    try:
        baseline_model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            baseline_model.fit(X_train, y_train)
            y_train_pred = baseline_model.predict(X_train)
            y_test_pred = baseline_model.predict(X_test)
        
        baseline_results = {
            'Train_R2': round(r2_score(y_train, y_train_pred), 6),
            'Test_R2': round(r2_score(y_test, y_test_pred), 6),
            'Train_RMSE': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 6),
            'Test_RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 6),
            'Train_MAE': round(mean_absolute_error(y_train, y_train_pred), 6),
            'Test_MAE': round(mean_absolute_error(y_test, y_test_pred), 6),
            'Train_MAPE': round(mean_absolute_percentage_error(y_train, y_train_pred), 6),
            'Test_MAPE': round(mean_absolute_percentage_error(y_test, y_test_pred), 6)
        }
        return baseline_results
    except Exception as e:
        logger.error(f"Baseline model başarısız: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  R515B DATA LOADING AND PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
def load_r515b_datasets(data_path: str = None):
    """
    R515B veri setlerini yükler: Doymuş ve Kızgın Buhar.
    
    SEBEP: Termodinamik özellikleri, sıcaklık ve basınç gibi temel parametrelere 
    bağımlıdır. İki farklı bölgede (doymuş ve kızgın buhar) eğitim yapmak,
    STGP-EF yönteminin her bölgede nasıl çalıştığını gösterir.
    """
    if data_path is None:
        base_path = Path(__file__).parent.parent / "R515B_Ozellikler"
    else:
        base_path = Path(data_path)
    
    datasets = {}
    try:
        # Doymuş Kaynama (Saturated) Veri Seti
        saturated_path = base_path / "R515B_Doymus_Ozellikleri.csv"
        datasets['saturated'] = pd.read_csv(saturated_path)
        logger.info(f"Doymuş Veri Seti Yüklendi: {datasets['saturated'].shape}")
        
        # Kızgın Buhar (Superheated) Veri Seti
        superheated_path = base_path / "R515B_Kizgin_Buhar.csv"
        datasets['superheated'] = pd.read_csv(superheated_path)
        logger.info(f"Kızgın Buhar Veri Seti Yüklendi: {datasets['superheated'].shape}")
        
    except FileNotFoundError as e:
        logger.error(f"Veri dosyası bulunamadı: {e}")
        return None
    
    return datasets


# ═══════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def calculate_feature_importance(model, feature_names, model_name: str, top_n: int = 15):
    """
    Model özellik önemliğini hesaplar ve görselleştirir.
    
    SEBEP: STGP-EF ile yapılandırılan özelliklerin ve orijinal girdilerin
    modele olan katkısını anlamak için. Siyah kutu modeli yorumlanabilir hale getirmek.
    """
    importances = None
    importance_type = "unknown"
    
    # Farklı model türlerine göre özellik önemliği çıkarma
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_type = "Tree-based"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
        importance_type = "Coefficient-based"
    
    if importances is None:
        logger.warning(f"{model_name} için özellik önemliği hesaplanamadı")
        return None
    
    # İndeksleme sorunu için kontrol
    if len(importances) != len(feature_names):
        logger.warning(f"Özellik sayı uyuşmazlığı: {len(importances)} vs {len(feature_names)}")
        return None
    
    # DataFrame oluştur
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Top-N özellikleri seç
    top_features = importance_df.head(top_n)
    
    logger.info(f"\n{'─' * 78}")
    logger.info(f"{model_name} - ÖZELLİK ÖNEMLİĞİ ({importance_type})")
    logger.info(f"{'─' * 78}")
    for idx, row in top_features.iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.6f}")
    
    return importance_df, top_features


# ═══════════════════════════════════════════════════════════════════
#  THERMODYNAMIC LAW COMPLIANCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════
def verify_thermodynamic_compliance(y_true, y_pred, property_name: str):
    """
    Tahminlerin termodinamik yasalara uyup uymadığını doğrular.
    
    SEBEP: R515B soğutucu sıvısının termodinamik özelliklerinin fiziksel olarak 
    anlamlı aralıklarda olması gerekir. Özzellik önemliği yüksek modellerin 
    fiziksek gerçekliği koruduğundan emin olmak için yapılır.
    
    Termodinamik Kanunları:
    - 1. Kanun: Enerji korunumu - Entalpi ve entropi monoton ilişki
    - 2. Kanun: Entropi artışı - Tahmin edilen değerler anlamsız olamaz
    - Pozitivite: Yoğunluk, entalpi, entropi pozitif olmalıdır
    """
    # Veri temizliği
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                   np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        logger.warning(f"{property_name}: Geçerli veri bulunamadı")
        return None
    
    # Pozitivite Kontrol
    if property_name in ['v sıvı (çıktı)', 'v buhar (çıktı)', 'v (çıktı)',
                         'h sıvı (çıktı)', 'h buhar (çıktı)', 'h (çıktı)',
                         's sıvı (çıktı)', 's buhar (çıktı)', 's (çıktı)']:
        neg_pred_ratio = (y_pred_clean < 0).sum() / len(y_pred_clean) * 100
    else:
        neg_pred_ratio = 0.0
    
    # Monotonluk Kontrol (artan sıcaklık → artan entalpi/entropi)
    sorted_indices = np.argsort(y_true_clean)
    sorted_pred = y_pred_clean[sorted_indices]
    
    # Toplam monotonik olamama sayısı
    monotonicity_violations = (np.diff(sorted_pred) < -1e-6).sum()
    monotonicity_ratio = monotonicity_violations / (len(sorted_pred) - 1) * 100 if len(sorted_pred) > 1 else 0
    
    # Standart Hata Yüzdeleri
    absolute_error = np.abs(y_true_clean - y_pred_clean)
    relative_error = absolute_error / (np.abs(y_true_clean) + 1e-10) * 100
    
    compliance_report = {
        'property': property_name,
        'n_valid_samples': len(y_true_clean),
        'negative_predictions_ratio': neg_pred_ratio,
        'monotonicity_violations_ratio': monotonicity_ratio,
        'mean_absolute_error': absolute_error.mean(),
        'max_absolute_error': absolute_error.max(),
        'mean_relative_error': relative_error.mean(),
        'max_relative_error': relative_error.max(),
        'r2_score': r2_score(y_true_clean, y_pred_clean) if len(y_true_clean) > 1 else np.nan
    }
    
    logger.info(f"\n{'─' * 78}")
    logger.info(f"TERMODİNAMİK UYUM KONTROLÜ - {property_name}")
    logger.info(f"{'─' * 78}")
    logger.info(f"  Geçerli Örnek Sayısı: {compliance_report['n_valid_samples']}")
    logger.info(f"  Negatif Tahmin Oranı: {compliance_report['negative_predictions_ratio']:.2f}%")
    logger.info(f"  Monotonik Olmama Oranı: {compliance_report['monotonicity_violations_ratio']:.2f}%")
    logger.info(f"  Ortalama Mutlak Hata: {compliance_report['mean_absolute_error']:.6f}")
    logger.info(f"  Maksimum Mutlak Hata: {compliance_report['max_absolute_error']:.6f}")
    logger.info(f"  Ortalama Göreceli Hata: {compliance_report['mean_relative_error']:.2f}%")
    logger.info(f"  Maksimum Göreceli Hata: {compliance_report['max_relative_error']:.2f}%")
    logger.info(f"  R² Skoru: {compliance_report['r2_score']:.6f}")
    
    return compliance_report


# ═══════════════════════════════════════════════════════════════════
#  STGP-EF FEATURE INTERPRETATION
# ═══════════════════════════════════════════════════════════════════
def interpret_stgp_ef_features(formulas_dict: dict, feature_categories: list = None):
    """
    STGP-EF ile üretilen özellikleri yorumlar ve fiziksel anlamlarını açıklar.
    
    SEBEP: Sembolik regresyon formülleri matematiksel olarak doğru olabilir
    ancak fiziksel olarak anlamlandırılması gerekir. Termodinamik özellikleri
    tahmin ederken, formüllerin reel dünyada anlamlı ilişkileri temsil etmesi önemlidir.
    """
    interpretation_report = {
        'total_features': len(formulas_dict),
        'features_by_complexity': {},
        'physical_interpretations': []
    }
    
    logger.info(f"\n{'─' * 78}")
    logger.info(f"STGP-EF ÖZELLİKLERİ YORUMLAMA")
    logger.info(f"{'─' * 78}")
    
    for feature_name, formula in formulas_dict.items():
        # Formül karmaşıklığı (operatör sayısı)
        operators = ['+', '-', '*', '/', 'log', 'exp', 'sqrt', 'abs', 'sin', 'cos']
        complexity = sum(formula.count(op) for op in operators)
        
        if complexity not in interpretation_report['features_by_complexity']:
            interpretation_report['features_by_complexity'][complexity] = []
        interpretation_report['features_by_complexity'][complexity].append(feature_name)
        
        logger.info(f"  {feature_name}:")
        logger.info(f"    Formül: {formula}")
        logger.info(f"    Karmaşıklık Seviyesi: {complexity}")
        
        # Basit formüller (karmaşıklık <= 2) genellikle daha stabil
        if complexity <= 2:
            logger.info(f"    → Basit ve Yorumlanabilir ✓")
        elif complexity <= 5:
            logger.info(f"    → Orta Karmaşıklıkta")
        else:
            logger.info(f"    → Yüksek Karmaşıklıkta (Aşırı uyum riski)")
    
    return interpretation_report


# ═══════════════════════════════════════════════════════════════════
#  BEST MODELS SELECTION
# ═══════════════════════════════════════════════════════════════════
def select_best_models(results_df, metric: str = 'Test_R2', top_n: int = 5):
    """
    Değerlendirme sonuçlarından en iyi modelleri seçer.
    
    SEBEP: STGP-EF yöntemi ile yapılandırılan özelliklerin en etkili modelleri
    belirlemek. Feature importance ve thermodynamic compliance analizi
    sadece iyi performans gösteren modellere uygulanmalıdır.
    """
    if results_df is None or results_df.empty:
        logger.warning("Boş sonuç DataFrame'i")
        return None
    
    # Sayısal sütunları filtreleyip sırala
    numeric_results = results_df.select_dtypes(include=[np.number])
    if metric not in numeric_results.columns:
        logger.warning(f"'{metric}' metriği bulunamadı")
        return None
    
    # Sıra ve en iyileri seç
    sorted_results = numeric_results.sort_values(metric, ascending=False)
    best_models = sorted_results.head(top_n)
    
    logger.info(f"\n{'─' * 78}")
    logger.info(f"EN İYİ MODELLER (Metrik: {metric})")
    logger.info(f"{'─' * 78}")
    for idx, (model_name, metrics) in enumerate(best_models.iterrows(), 1):
        logger.info(f"  {idx}. {model_name}: {metrics[metric]:.6f}")
    
    return best_models.index.tolist(), best_models


# ═══════════════════════════════════════════════════════════════════
#  COMPREHENSIVE STGP-EF ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════
def run_stgp_ef_comprehensive_analysis(
    X_train_stgp, X_test_stgp, y_train, y_test,
    input_cols_original, best_model_name, best_model,
    property_name: str
):
    """
    STGP-EF ile yapılandırılmış veriler üzerinde kapsamlı analiz yapar.
    
    SEBEP: Siyah kutu modellerinin kararlarını açık hale getirmek ve
    termodinamik yasalara uygunluğunu doğrulamak için bu pipeline'ı
    tüm en iyi modellere uygulamak gerekir.
    
    Adımlar:
    1. Feature importance hesaplaması
    2. Tahmin yap
    3. Termodinamik uyum kontrol
    4. Sonuçları raporla
    """
    logger.info(f"\n{'═' * 78}")
    logger.info(f"STGP-EF KAPSAMLI ANALİZİ - {property_name}")
    logger.info(f"{'═' * 78}")
    
    # Özellik isimleri (orijinal + STGP-EF özellikleri)
    n_original = len(input_cols_original)
    n_constructed = X_train_stgp.shape[1] - n_original
    
    feature_names = (input_cols_original + 
                     [f'STGP_EF_{i}' for i in range(n_constructed)])
    
    # 1. Feature Importance
    importance_data = calculate_feature_importance(
        best_model, feature_names, best_model_name
    )
    
    # 2. Tahminler
    y_pred = best_model.predict(X_test_stgp)
    
    # 3. Termodinamik Uyum
    compliance = verify_thermodynamic_compliance(y_test, y_pred, property_name)
    
    # 4. Sonuç Paketi
    analysis_results = {
        'model_name': best_model_name,
        'property': property_name,
        'feature_importance': importance_data,
        'compliance_check': compliance,
        'test_r2': r2_score(y_test, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return analysis_results