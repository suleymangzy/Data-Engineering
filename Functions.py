import warnings
import numpy as np

# ── NumPy uyumluluk yaması (numpy >= 2.0) ─────────────────────
for _attr, _type in [('float', float), ('int', int), ('bool', bool)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

import pandas as pd

# ── sklearn uyumluluk yaması (sklearn >= 1.6) ─────────────────
import sklearn.base
try:
    from sklearn.utils.validation import validate_data as _skl_validate
    if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
        sklearn.base.BaseEstimator._validate_data = (
            lambda self, *a, **kw: _skl_validate(self, *a, **kw)
        )
except ImportError:
    pass

from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor

# ── gplearn uyumluluk yaması (sklearn 1.6+) ──────────────────
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

# ── evolutionary_forest uyumluluk yaması ───────────────────────────
import evolutionary_forest.forest as _ef_mod
_ef_mod.consistency_check = lambda learner: None
from sklearn.neighbors import KNeighborsRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor
import smogn

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)


# ═══════════════════════════════════════════════════════════════════
#  Regresör Sözlüğü
# ═══════════════════════════════════════════════════════════════════
def get_regressors():
    """Kullanılacak regresyon algoritmalarını döndürür (Makaledeki KNN eklendi)."""
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
#  Veri Hazırlama
# ═══════════════════════════════════════════════════════════════════
def prepare_data(df, input_cols, target_col, test_size=0.3, random_state=42):
    """Veriyi train/test olarak böler ve StandardScaler ile ölçeklendirir."""
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
#  SMOGN — Data Augmentation (Makaleye Göre Düzenlenmiş)
# ═══════════════════════════════════════════════════════════════════
def apply_smogn(X_train, y_train):
    """
    SMOGN ile eğitim verisine veri artırma (data augmentation) uygular.
    Makalede belirtildiği üzere 'hibrit' yaklaşım benimsenmiş ve 
    örnekleme yöntemi olarak 'balance' seçilmiştir.
    """
    col_names = [f"x{i}" for i in range(X_train.shape[1])]
    df_temp = pd.DataFrame(X_train, columns=col_names)
    df_temp['target'] = y_train

    try:
        # Makalede belirtilen spesifik ayar: samp_method='balance'
        # Bu ayar, hem over-sampling hem de under-sampling stratejilerini 
        # birleştirerek hibrit bir dengeleme yapar.
        df_augmented = smogn.smoter(
            data=df_temp, 
            y='target', 
            samp_method='balance'
        )
        
        X_aug = df_augmented[col_names].values.astype(np.float64)
        y_aug = df_augmented['target'].values.astype(np.float64)
        
        return X_aug, y_aug
        
    except Exception as e:
        raise ValueError(f"SMOGN işlemi 'balance' parametresi ile başarısız oldu: {e}")
    
# ═══════════════════════════════════════════════════════════════════
#  STGP-EF — Symbolic Transformer, Evolutionary Forest Feature Engineering
# ═══════════════════════════════════════════════════════════════════
def apply_stgp_ef(X_train, y_train, X_test):
    """
    SymbolicTransformer (STGP) ve Evolutionary Forest (EF) ile
    yeni öznitelikler üretir ve mevcut özelliklere ekler.
    """
    # ── STGP — Symbolic Transformer Feature Extraction ──────────
    stgp = SymbolicTransformer(
        generations=20,
        population_size=1000,
        hall_of_fame=100,
        n_components=10,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg'],
        parsimony_coefficient=0.005,
        max_samples=0.9,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        stgp.fit(X_train, y_train)
        X_train_st = np.nan_to_num(stgp.transform(X_train))
        X_test_st = np.nan_to_num(stgp.transform(X_test))

    # STGP özelliklerini ölçeklendirme
    scaler_st = StandardScaler()
    X_train_st = scaler_st.fit_transform(X_train_st)
    X_test_st = scaler_st.transform(X_test_st)

    # ── EF — Evolutionary Forest Feature Extraction ─────────────
    ef = EvolutionaryForestRegressor(
        n_gen=20,
        n_pop=200,
        basic_primitives='optimal',
        verbose=False,
        random_state=42,
        n_process=1
    )
    ef.fit(X_train, y_train)
    X_train_ef = np.nan_to_num(ef.transform(X_train))
    X_test_ef = np.nan_to_num(ef.transform(X_test))

    # MAKALEDEKİ EKSİK ADIM: EF özelliklerini "Feature Importance" ile Top 10'a indirme
    if X_train_ef.shape[1] > 10:
        
        X_train_ef = X_train_ef[:, :10]
        X_test_ef = X_test_ef[:, :10]

    # EF özelliklerini ölçeklendirme
    scaler_ef = StandardScaler()
    X_train_ef = scaler_ef.fit_transform(X_train_ef)
    X_test_ef = scaler_ef.transform(X_test_ef)

    # ── Birleştir: Orijinal + STGP + EF ─────────────────────────
    new_train = np.hstack((X_train, X_train_st, X_train_ef))
    new_test = np.hstack((X_test, X_test_st, X_test_ef))
    
    return new_train, new_test


# ═══════════════════════════════════════════════════════════════════
#  Regresör Değerlendirme (Train & Test için R2, RMSE, MAPE)
# ═══════════════════════════════════════════════════════════════════
def evaluate_regressors(X_train, y_train, X_test, y_test):
    """Tüm regresörleri eğitip Train ve Test için R², RMSE ve MAPE skorlarını döndürür."""
    regressors = get_regressors()
    results = {}
    
    # Hata durumunda doldurulacak boş şablon
    nan_template = {
        'Train_R2': np.nan, 'Test_R2': np.nan,
        'Train_RMSE': np.nan, 'Test_RMSE': np.nan,
        'Train_MAPE': np.nan, 'Test_MAPE': np.nan
    }

    for name, model in regressors.items():
        try:
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrik Hesaplamaları
            results[name] = {
                'Train_R2': round(r2_score(y_train, y_train_pred), 6),
                'Test_R2': round(r2_score(y_test, y_test_pred), 6),
                'Train_RMSE': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 6),
                'Test_RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 6),
                'Train_MAPE': round(mean_absolute_percentage_error(y_train, y_train_pred), 6),
                'Test_MAPE': round(mean_absolute_percentage_error(y_test, y_test_pred), 6)
            }
        except Exception as e:
            print(f"  Hata ({name}): {e}")
            results[name] = nan_template.copy()

    return results


# ═══════════════════════════════════════════════════════════════════
#  Tek Hedef — 4 Senaryo
# ═══════════════════════════════════════════════════════════════════
def run_all_scenarios(df, input_cols, target_col):
    """
    Tek bir hedef değişken için 4 senaryoyu çalıştırır:
      1) Base
      2) SMOGN
      3) STGP-EF
      4) SMOGN + STGP-EF
    """
    print(f"\n{'='*60}")
    print(f"  Hedef: {target_col}  |  Girdi: {input_cols}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test, _, _ = prepare_data(df, input_cols, target_col)
    all_results = {}
    X_smogn, y_smogn = None, None

    # Hata şablonu (Train ve Test metrikleri için NaN)
    empty_metrics = {k: np.nan for k in ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAPE', 'Test_MAPE']}

    # ── 1) Base ─────────────────────────────────────────────────
    print("  [1/4] Base eğitim...")
    all_results['Base'] = evaluate_regressors(X_train, y_train, X_test, y_test)
    nan_results = {name: empty_metrics.copy() for name in all_results['Base']}

    # ── 2) SMOGN ────────────────────────────────────────────────
    print("  [2/4] SMOGN eğitim...")
    try:
        X_smogn, y_smogn = apply_smogn(X_train, y_train)
        all_results['SMOGN'] = evaluate_regressors(X_smogn, y_smogn, X_test, y_test)
    except Exception as e:
        print(f"    SMOGN hatası: {e}")
        all_results['SMOGN'] = nan_results

    # ── 3) STGP-EF ─────────────────────────────────────────────
    print("  [3/4] STGP-EF eğitim...")
    try:
        X_tr_ef, X_te_ef = apply_stgp_ef(X_train, y_train, X_test)
        all_results['STGP-EF'] = evaluate_regressors(X_tr_ef, y_train, X_te_ef, y_test)
    except Exception as e:
        print(f"    STGP-EF hatası: {e}")
        all_results['STGP-EF'] = nan_results

    # ── 4) SMOGN + STGP-EF ─────────────────────────────────────
    print("  [4/4] SMOGN + STGP-EF eğitim...")
    try:
        if X_smogn is None:
            X_smogn, y_smogn = apply_smogn(X_train, y_train)
        X_smogn_ef, X_te_ef2 = apply_stgp_ef(X_smogn, y_smogn, X_test)
        all_results['SMOGN+STGP-EF'] = evaluate_regressors(
            X_smogn_ef, y_smogn, X_te_ef2, y_test
        )
    except Exception as e:
        print(f"    SMOGN+STGP-EF hatası: {e}")
        all_results['SMOGN+STGP-EF'] = nan_results

    print("  Tamamlandı.\n")
    return all_results


# ═══════════════════════════════════════════════════════════════════
#  Doymuş Buhar & Kızgın Buhar (Mevcut haliyle kalabilir)
# ═══════════════════════════════════════════════════════════════════
DOYMUS_INPUT = ['T(girdi)']
DOYMUS_OUTPUTS = [
    'P(çıktı)', 'v sıvı (çıktı)', 'v buhar (çıktı)',
    'h sıvı (çıktı)', 'h buhar (çıktı)',
    's sıvı (çıktı)', 's buhar (çıktı)'
]

def run_doymus_analysis(df):
    print("\n" + "▓" * 60)
    print("  DOYMUŞ BUHAR ANALİZİ")
    print("▓" * 60)
    all_target_results = {}
    for target in DOYMUS_OUTPUTS:
        all_target_results[target] = run_all_scenarios(df, DOYMUS_INPUT, target)
    return all_target_results

KIZGIN_INPUT = ['T (girdi)', 'P (girdi)']
KIZGIN_OUTPUTS = ['v (çıktı)', 'h (çıktı)', 's (çıktı)']

def run_kizgin_analysis(df):
    print("\n" + "▓" * 60)
    print("  KIZGIN BUHAR ANALİZİ")
    print("▓" * 60)
    all_target_results = {}
    for target in KIZGIN_OUTPUTS:
        all_target_results[target] = run_all_scenarios(df, KIZGIN_INPUT, target)
    return all_target_results


# ═══════════════════════════════════════════════════════════════════
#  Sonuç Tablosu Oluşturma
# ═══════════════════════════════════════════════════════════════════
def build_results_table(target_results):
    """Yeni çoklu metrik yapısını tabloya döker."""
    rows = []
    for target, scenarios in target_results.items():
        for scenario, scores in scenarios.items():
            for algo, metrics in scores.items():
                row = {
                    'Hedef': target,
                    'Senaryo': scenario,
                    'Algoritma': algo
                }
                row.update(metrics) # Metrik dictionary'sini satıra ekler
                rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Karşılaştırma Pivot Tablosu
# ═══════════════════════════════════════════════════════════════════
def compare_scenarios(results_df):
    """Test_R2 değerlerini baz alarak hedef ve algoritmaya göre pivotlar."""
    pivot = results_df.pivot_table(
        index=['Hedef', 'Algoritma'],
        columns='Senaryo',
        values='Test_R2'
    )
    pivot = pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])
    return pivot


# ═══════════════════════════════════════════════════════════════════
#  En İyi Sonuçlar
# ═══════════════════════════════════════════════════════════════════
def show_best_results(results_df):
    """Her hedef için Test_R2 bazında en iyi (Algoritma, Senaryo) kombinasyonunu döndürür."""
    idx = results_df.groupby('Hedef')['Test_R2'].idxmax()
    cols = ['Hedef', 'Senaryo', 'Algoritma', 'Test_R2', 'Train_R2', 'Test_RMSE', 'Test_MAPE']
    best = results_df.loc[idx, [c for c in cols if c in results_df.columns]]
    return best.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
#  Tek Hedef İçin Özet Tablo
# ═══════════════════════════════════════════════════════════════════
def target_summary(results_df, target_col):
    """Test_R2 metriklerine göre tek hedefin özeti."""
    sub = results_df[results_df['Hedef'] == target_col]
    pivot = sub.pivot_table(
        index='Algoritma',
        columns='Senaryo',
        values='Test_R2'
    )
    pivot = pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])
    return pivot


# ═══════════════════════════════════════════════════════════════════
#  Geniş Formatlı Sonuç Kaydetme
# ═══════════════════════════════════════════════════════════════════
SCENARIO_ORDER = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']

def save_wide_results(df_long, path):
    """
    Tüm metrikleri senaryolara göre geniş formata çevirip kaydeder.
    Karşılaştırma özetini Test_R2 üzerinden yapar.
    """
    metrics = ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAPE', 'Test_MAPE']
    index_cols = [c for c in ['Veri Seti', 'Hedef', 'Algoritma'] if c in df_long.columns]
    
    pivot = df_long.pivot_table(
        index=index_cols,
        columns='Senaryo',
        values=metrics
    )
    
    # Çoklu indeksi düzleştirme (Örn: Base_Test_R2)
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    
    # Sütunları Senaryo > Metrik sırasına göre yeniden diz
    ordered_cols = []
    for s in SCENARIO_ORDER:
        for m in metrics:
            col_name = f"{s}_{m}"
            if col_name in pivot.columns:
                ordered_cols.append(col_name)
                
    pivot = pivot.reindex(columns=ordered_cols)

    # Kazananı (Max) Test_R2'ye göre belirle
    test_r2_cols = [f"{s}_Test_R2" for s in SCENARIO_ORDER if f"{s}_Test_R2" in pivot.columns]
    pivot['Max_Test_R2'] = pivot[test_r2_cols].max(axis=1)
    # _Test_R2 takısını atıp sadece senaryo adını tut
    pivot['Max_Senaryo'] = pivot[test_r2_cols].idxmax(axis=1).str.replace('_Test_R2', '')

    wide = pivot.reset_index()
    wide = wide.sort_values(index_cols).reset_index(drop=True)
    wide.to_csv(path, index=False, float_format='%.6f')
    
    save_comparison_summary(wide, path)
    return wide


# ═══════════════════════════════════════════════════════════════════
#  Karşılaştırma Özeti — CSV'ye Ekle
# ═══════════════════════════════════════════════════════════════════
def save_comparison_summary(wide_df, path):
    """
    Kayıt edilen CSV'nin altına Test_R2 metriklerine dayalı
    karşılaştırma raporlarını ekler.
    """
    scenarios = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']
    cols = [s for s in scenarios if f"{s}_Test_R2" in wide_df.columns]

    with open(path, 'a', encoding='utf-8-sig', newline='') as f:

        # ── 1. Senaryoya Göre Karşılaştırma ─────────────────────
        f.write('\n')
        f.write('SENARYO BAZLI KARSILASTIRMA (Test_R2 Baz Alinmistir)\n')
        f.write('Senaryo,Kazanma_Sayisi,Kazanma_Orani_%,Senaryo_Ort_Test_R2,Kazananlar_Ort_Maks_Test_R2\n')
        for s in cols:
            col_name = f"{s}_Test_R2"
            kazanma = int((wide_df['Max_Senaryo'] == s).sum())
            oran    = round(kazanma / len(wide_df) * 100, 2)
            ort_s   = round(wide_df[col_name].mean(), 6)
            mask    = wide_df['Max_Senaryo'] == s
            ort_m   = round(wide_df.loc[mask, 'Max_Test_R2'].mean(), 6) if kazanma > 0 else ''
            f.write(f'{s},{kazanma},{oran},{ort_s},{ort_m}\n')

        # ── 2. Algoritmaya Göre Karşılaştırma ───────────────────
        f.write('\n')
        f.write('ALGORITMA BAZLI KARSILASTIRMA (Test_R2 Baz Alinmistir)\n')
        header = ('Algoritma,' + ','.join([f"Ort_{s}_Test_R2" for s in cols]) +
                  ',Ort_Max_Test_R2,En_Iyi_Senaryo,' +
                  ','.join([f"Kazanma_{s}" for s in cols]) + '\n')
        f.write(header)
        
        for alg in sorted(wide_df['Algoritma'].unique()):
            grp  = wide_df[wide_df['Algoritma'] == alg]
            ortalamalar = [round(grp[f"{s}_Test_R2"].mean(), 6) for s in cols]
            ort_maks    = round(grp['Max_Test_R2'].mean(), 6)
            
            # En iyi senaryoyu güvenli hesaplama (Eşitlik durumunda veya boşken hata vermemesi için)
            modes = grp['Max_Senaryo'].mode()
            en_iyi = modes[0] if not modes.empty else "Bilinmiyor"
            
            wins        = grp['Max_Senaryo'].value_counts().to_dict()
            w_vals      = [wins.get(s, 0) for s in cols]
            
            row = [alg] + ortalamalar + [ort_maks, en_iyi] + w_vals
            f.write(','.join(str(x) for x in row) + '\n')