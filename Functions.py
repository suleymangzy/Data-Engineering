import warnings
import numpy as np

# ── NumPy uyumluluk yaması (numpy >= 2.0) ─────────────────────
# evolutionary_forest/tpot, kaldırılmış np.float vb. kullanır
for _attr, _type in [('float', float), ('int', int), ('bool', bool)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

import pandas as pd

# ── sklearn uyumluluk yaması (sklearn >= 1.6) ─────────────────
# gplearn, kaldırılmış BaseEstimator._validate_data metodunu kullanır
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
    RandomForestRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from gplearn.genetic import SymbolicTransformer

# ── gplearn uyumluluk yaması (sklearn 1.6+) ──────────────────
# sklearn 1.6+ __sklearn_tags__ MRO zinciri gplearn ile uyumsuz;
# _validate_data'yı BaseSymbolic üzerinde geçersiz kılıyoruz.
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
# basic_primitives consistency_check bugını atla (v0.2.4)
import evolutionary_forest.forest as _ef_mod
_ef_mod.consistency_check = lambda learner: None

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
    """Kullanılacak regresyon algoritmalarını döndürür."""
    return {
        'RF': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'ET': ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=200, random_state=42),
        'GBDT': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'DART': LGBMRegressor(n_jobs=1, n_estimators=200, boosting_type='dart',
                              verbose=-1, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=200, verbosity=0, random_state=42),
        'LightGBM': LGBMRegressor(n_jobs=1, n_estimators=200, verbose=-1, random_state=42),
        'CatBoost': CatBoostRegressor(n_estimators=200, thread_count=1,
                                      verbose=False, allow_writing_files=False,
                                      random_seed=42),
    }


# ═══════════════════════════════════════════════════════════════════
#  Veri Hazırlama
# ═══════════════════════════════════════════════════════════════════
def prepare_data(df, input_cols, target_col, test_size=0.2, random_state=42):
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
#  SMOGN — Data Augmentation
# ═══════════════════════════════════════════════════════════════════
def apply_smogn(X_train, y_train):
    """SMOGN ile eğitim verisine veri artırma (data augmentation) uygular."""
    col_names = [f"x{i}" for i in range(X_train.shape[1])]
    df_temp = pd.DataFrame(X_train, columns=col_names)
    df_temp['target'] = y_train

    y_min = float(df_temp['target'].min())
    y_med = float(df_temp['target'].median())
    y_max = float(df_temp['target'].max())

    strategies = [
        {},
        {'rel_xtrm_type': 'both', 'rel_coef': 1.5},
        {'rel_xtrm_type': 'both', 'rel_coef': 0.5},
        {'rel_method': 'manual',
         'rel_ctrl_pts_rg': [[y_min, 1, 0],
                             [y_med, 0, 0],
                             [y_max, 1, 0]]},
    ]

    for params in strategies:
        try:
            df_augmented = smogn.smoter(data=df_temp, y='target', **params)
            X_aug = df_augmented[col_names].values.astype(np.float64)
            y_aug = df_augmented['target'].values.astype(np.float64)
            return X_aug, y_aug
        except (ValueError, Exception):
            continue

    raise ValueError("SMOGN: tüm relevanslık stratejileri başarısız oldu")


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

    # ── Birleştir: Orijinal + STGP + EF ─────────────────────────
    new_train = np.hstack((X_train, X_train_st, X_train_ef))
    new_test = np.hstack((X_test, X_test_st, X_test_ef))
    return new_train, new_test


# ═══════════════════════════════════════════════════════════════════
#  Regresör Değerlendirme
# ═══════════════════════════════════════════════════════════════════
def evaluate_regressors(X_train, y_train, X_test, y_test):
    """Tüm regresörleri eğitip R² skorlarını döndürür."""
    regressors = get_regressors()
    results = {}

    for name, model in regressors.items():
        try:
            m = clone(model)
            m.fit(X_train, y_train)
            score = r2_score(y_test, m.predict(X_test))
            results[name] = round(score, 6)
        except Exception as e:
            print(f"  Hata ({name}): {e}")
            results[name] = np.nan

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

    # ── 1) Base ─────────────────────────────────────────────────
    print("  [1/4] Base eğitim...")
    all_results['Base'] = evaluate_regressors(X_train, y_train, X_test, y_test)

    # ── 2) SMOGN ────────────────────────────────────────────────
    print("  [2/4] SMOGN eğitim...")
    try:
        X_smogn, y_smogn = apply_smogn(X_train, y_train)
        all_results['SMOGN'] = evaluate_regressors(X_smogn, y_smogn, X_test, y_test)
    except Exception as e:
        print(f"    SMOGN hatası: {e}")
        all_results['SMOGN'] = {name: np.nan for name in get_regressors()}

    # ── 3) STGP-EF ─────────────────────────────────────────────
    print("  [3/4] STGP-EF eğitim...")
    try:
        X_tr_ef, X_te_ef = apply_stgp_ef(X_train, y_train, X_test)
        all_results['STGP-EF'] = evaluate_regressors(X_tr_ef, y_train, X_te_ef, y_test)
    except Exception as e:
        print(f"    STGP-EF hatası: {e}")
        all_results['STGP-EF'] = {name: np.nan for name in get_regressors()}

    # ── 4) SMOGN + STGP-EF ─────────────────────────────────────
    print("  [4/4] SMOGN + STGP-EF eğitim...")
    try:
        X_smogn2, y_smogn2 = apply_smogn(X_train, y_train)
        X_smogn_ef, X_te_ef2 = apply_stgp_ef(X_smogn2, y_smogn2, X_test)
        all_results['SMOGN+STGP-EF'] = evaluate_regressors(
            X_smogn_ef, y_smogn2, X_te_ef2, y_test
        )
    except Exception as e:
        print(f"    SMOGN+STGP-EF hatası: {e}")
        all_results['SMOGN+STGP-EF'] = {name: np.nan for name in get_regressors()}

    print("  Tamamlandı.\n")
    return all_results


# ═══════════════════════════════════════════════════════════════════
#  Doymuş Buhar — Tüm Çıktılar İçin Eğitim
# ═══════════════════════════════════════════════════════════════════
DOYMUS_INPUT = ['T(girdi)']
DOYMUS_OUTPUTS = [
    'P(çıktı)', 'v sıvı (çıktı)', 'v buhar (çıktı)',
    'h sıvı (çıktı)', 'h buhar (çıktı)',
    's sıvı (çıktı)', 's buhar (çıktı)'
]


def run_doymus_analysis(df):
    """Doymuş buhar veri seti için tüm çıktı değişkenlerini ayrı ayrı eğitir."""
    print("\n" + "▓" * 60)
    print("  DOYMUŞ BUHAR ANALİZİ")
    print("▓" * 60)

    all_target_results = {}
    for target in DOYMUS_OUTPUTS:
        all_target_results[target] = run_all_scenarios(df, DOYMUS_INPUT, target)
    return all_target_results


# ═══════════════════════════════════════════════════════════════════
#  Kızgın Buhar — Tüm Çıktılar İçin Eğitim
# ═══════════════════════════════════════════════════════════════════
KIZGIN_INPUT = ['T (girdi)', 'P (girdi)']
KIZGIN_OUTPUTS = ['v (çıktı)', 'h (çıktı)', 's (çıktı)']


def run_kizgin_analysis(df):
    """Kızgın buhar veri seti için tüm çıktı değişkenlerini ayrı ayrı eğitir."""
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
    """
    run_doymus_analysis veya run_kizgin_analysis çıktısını
    düzenli bir DataFrame'e dönüştürür.
    """
    rows = []
    for target, scenarios in target_results.items():
        for scenario, scores in scenarios.items():
            for algo, score in scores.items():
                rows.append({
                    'Hedef': target,
                    'Senaryo': scenario,
                    'Algoritma': algo,
                    'R2_Score': score
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Karşılaştırma Pivot Tablosu
# ═══════════════════════════════════════════════════════════════════
def compare_scenarios(results_df):
    """
    Hedef × Algoritma satırlarında, Senaryo sütunlarında R² skorları.
    """
    pivot = results_df.pivot_table(
        index=['Hedef', 'Algoritma'],
        columns='Senaryo',
        values='R2_Score'
    )
    scenario_order = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']
    pivot = pivot.reindex(columns=[s for s in scenario_order if s in pivot.columns])
    return pivot


# ═══════════════════════════════════════════════════════════════════
#  En İyi Sonuçlar
# ═══════════════════════════════════════════════════════════════════
def show_best_results(results_df):
    """Her hedef değişken için en iyi (Algoritma, Senaryo) kombinasyonunu döndürür."""
    idx = results_df.groupby('Hedef')['R2_Score'].idxmax()
    best = results_df.loc[idx, ['Hedef', 'Senaryo', 'Algoritma', 'R2_Score']]
    return best.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
#  Tek Hedef İçin Özet Tablo
# ═══════════════════════════════════════════════════════════════════
def target_summary(results_df, target_col):
    """Belirli bir hedef değişken için senaryo karşılaştırma tablosu."""
    sub = results_df[results_df['Hedef'] == target_col]
    pivot = sub.pivot_table(
        index='Algoritma',
        columns='Senaryo',
        values='R2_Score'
    )
    scenario_order = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']
    pivot = pivot.reindex(columns=[s for s in scenario_order if s in pivot.columns])
    return pivot


# ═══════════════════════════════════════════════════════════════════
#  Geniş Formatlı Sonuç Kaydetme
# ═══════════════════════════════════════════════════════════════════
SCENARIO_ORDER = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']


def save_wide_results(df_long, path):
    """
    Uzun formattaki sonuç DataFrame'ini geniş (pivot) formata çevirip
    okunabilir bir CSV olarak kaydeder.

    Sütunlar: Veri Seti | Hedef | Algoritma | Base | SMOGN | STGP-EF |
              SMOGN+STGP-EF | Maks_R2 | Maks_Senaryo
    """
    pivot = df_long.pivot_table(
        index=['Veri Seti', 'Hedef', 'Algoritma'],
        columns='Senaryo',
        values='R2_Score'
    )
    cols = [s for s in SCENARIO_ORDER if s in pivot.columns]
    pivot = pivot.reindex(columns=cols)

    pivot['Maks_R2'] = pivot[cols].max(axis=1)
    pivot['Maks_Senaryo'] = pivot[cols].idxmax(axis=1)

    wide = pivot.reset_index()
    wide = wide.sort_values(['Veri Seti', 'Hedef', 'Algoritma']).reset_index(drop=True)
    wide.to_csv(path, index=False, float_format='%.6f')
    return wide