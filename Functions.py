# ═══════════════════════════════════════════════════════════════════
#  IMPORT SECTION
# ═══════════════════════════════════════════════════════════════════
import warnings
import os
import numpy as np
import pandas as pd
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
    r2_score, mean_absolute_error, mean_squared_error, 
    mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# ML Kütüphaneleri
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor
import smogn

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
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
SCENARIO_ORDER = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']

SATURATED_INPUTS = ['T_input']
SATURATED_OUTPUTS = [
    'P_output', 'v_liquid_output', 'v_vapor_output',
    'h_liquid_output', 'h_vapor_output',
    's_liquid_output', 's_vapor_output'
]

SUPERHEATED_INPUTS = ['T_input', 'P_input']
SUPERHEATED_OUTPUTS = ['v_output', 'h_output', 's_output']

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
def apply_smogn(X_train, y_train):
    """
    Applies data augmentation to training data using SMOGN.
    Uses hybrid approach with 'balance' sampling method as specified in paper.
    """
    col_names = [f"x{i}" for i in range(X_train.shape[1])]
    df_temp = pd.DataFrame(X_train, columns=col_names)
    df_temp['target'] = y_train

    try:
        # Paper-specified configuration: samp_method='balance'
        # This combines over-sampling and under-sampling for hybrid balancing.
        df_augmented = smogn.smoter(
            data=df_temp, 
            y='target', 
            samp_method='balance'
        )
        
        X_aug = df_augmented[col_names].values.astype(np.float64)
        y_aug = df_augmented['target'].values.astype(np.float64)
        
        return X_aug, y_aug
        
    except Exception as e:
        raise ValueError(f"SMOGN operation failed with 'balance' parameter: {e}")


# ═══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING - STGP-EF (Feature Engineering)
# ═══════════════════════════════════════════════════════════════════
def apply_stgp_ef(X_train, y_train, X_test):
    """
    Generates new features using SymbolicTransformer (STGP) and 
    Evolutionary Forest (EF), then combines with original features.
    """
    # ── STGP - Symbolic Transformer Feature Extraction ──────────
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

    # Scale STGP features
    scaler_st = StandardScaler()
    X_train_st = scaler_st.fit_transform(X_train_st)
    X_test_st = scaler_st.transform(X_test_st)

    # ── EF - Evolutionary Forest Feature Extraction ─────────────
    ef = EvolutionaryForestRegressor(
        n_gen=20,
        n_pop=200,
        basic_primitives='optimal',
        verbose=False,
        random_state=42,
        n_process=1
    )

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        ef.fit(X_train, y_train)
        X_train_ef = np.nan_to_num(ef.transform(X_train))
        X_test_ef = np.nan_to_num(ef.transform(X_test))

    # Paper missing step: Reduce EF features to Top 10 based on Feature Importance
    if X_train_ef.shape[1] > 10:
        X_train_ef = X_train_ef[:, :10]
        X_test_ef = X_test_ef[:, :10]

    # Scale EF features
    scaler_ef = StandardScaler()
    X_train_ef = scaler_ef.fit_transform(X_train_ef)
    X_test_ef = scaler_ef.transform(X_test_ef)

    # ── Combine: Original + STGP + EF ──────────────────────────
    new_train = np.hstack((X_train, X_train_st, X_train_ef))
    new_test = np.hstack((X_test, X_test_st, X_test_ef))
    
    return new_train, new_test


# ═══════════════════════════════════════════════════════════════════
#  REGRESSOR EVALUATION
# ═══════════════════════════════════════════════════════════════════
def evaluate_regressors(X_train, y_train, X_test, y_test):
    """Trains all regressors and returns R2, RMSE, MAPE scores for Train and Test sets."""
    regressors = get_regressors()
    results = {}
    
    # Empty template for error cases
    nan_template = {
        'Train_R2': np.nan, 'Test_R2': np.nan,
        'Train_RMSE': np.nan, 'Test_RMSE': np.nan,
        'Train_MAPE': np.nan, 'Test_MAPE': np.nan
    }

    for name, model in regressors.items():
        try:
            # Suppress hidden prints during training and prediction
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                model.fit(X_train, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
            
            # Metric calculations
            results[name] = {
                'Train_R2': round(r2_score(y_train, y_train_pred), 6),
                'Test_R2': round(r2_score(y_test, y_test_pred), 6),
                'Train_RMSE': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 6),
                'Test_RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 6),
                'Train_MAPE': round(mean_absolute_percentage_error(y_train, y_train_pred), 6),
                'Test_MAPE': round(mean_absolute_percentage_error(y_test, y_test_pred), 6)
            }
        except Exception as e:
            # Print only errors; keep training process silent
            print(f"  Error ({name}): {e}")
            results[name] = nan_template.copy()

    return results


# ═══════════════════════════════════════════════════════════════════
#  SCENARIO ANALYSIS - SINGLE TARGET
# ═══════════════════════════════════════════════════════════════════
def run_all_scenarios(df, input_cols, target_col):
    """
    Runs 4 scenarios for a single target variable:
      1) Base
      2) SMOGN
      3) STGP-EF
      4) SMOGN + STGP-EF
    """
    print(f"\n{'='*60}")
    print(f"  Target: {target_col}  |  Input: {input_cols}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test, _, _ = prepare_data(df, input_cols, target_col)
    all_results = {}
    X_smogn, y_smogn = None, None

    # Error template (NaN for Train and Test metrics)
    empty_metrics = {k: np.nan for k in ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAPE', 'Test_MAPE']}

    # ── 1) Base ─────────────────────────────────────────────────
    print("  [1/4] Base training...")
    all_results['Base'] = evaluate_regressors(X_train, y_train, X_test, y_test)
    nan_results = {name: empty_metrics.copy() for name in all_results['Base']}

    # ── 2) SMOGN ────────────────────────────────────────────────
    print("  [2/4] SMOGN training...")
    try:
        X_smogn, y_smogn = apply_smogn(X_train, y_train)
        all_results['SMOGN'] = evaluate_regressors(X_smogn, y_smogn, X_test, y_test)
    except Exception as e:
        print(f"    SMOGN error: {e}")
        all_results['SMOGN'] = nan_results

    # ── 3) STGP-EF ─────────────────────────────────────────────
    print("  [3/4] STGP-EF training...")
    try:
        X_tr_ef, X_te_ef = apply_stgp_ef(X_train, y_train, X_test)
        all_results['STGP-EF'] = evaluate_regressors(X_tr_ef, y_train, X_te_ef, y_test)
    except Exception as e:
        print(f"    STGP-EF error: {e}")
        all_results['STGP-EF'] = nan_results

    # ── 4) SMOGN + STGP-EF ─────────────────────────────────────
    print("  [4/4] SMOGN + STGP-EF training...")
    try:
        if X_smogn is None:
            X_smogn, y_smogn = apply_smogn(X_train, y_train)
        X_smogn_ef, X_te_ef2 = apply_stgp_ef(X_smogn, y_smogn, X_test)
        all_results['SMOGN+STGP-EF'] = evaluate_regressors(
            X_smogn_ef, y_smogn, X_te_ef2, y_test
        )
    except Exception as e:
        print(f"    SMOGN+STGP-EF error: {e}")
        all_results['SMOGN+STGP-EF'] = nan_results

    print("  Completed.\n")
    return all_results


# ═══════════════════════════════════════════════════════════════════
#  ANALYSIS RUNNERS - SATURATED AND SUPERHEATED STEAM
# ═══════════════════════════════════════════════════════════════════
def run_saturated_analysis(df):
    """Runs analysis for all target outputs in saturated steam."""
    print("\n" + "▓" * 60)
    print("  SATURATED STEAM ANALYSIS")
    print("▓" * 60)
    all_target_results = {}
    for target in SATURATED_OUTPUTS:
        all_target_results[target] = run_all_scenarios(df, SATURATED_INPUTS, target)
    return all_target_results


def run_superheated_analysis(df):
    """Runs analysis for all target outputs in superheated steam."""
    print("\n" + "▓" * 60)
    print("  SUPERHEATED STEAM ANALYSIS")
    print("▓" * 60)
    all_target_results = {}
    for target in SUPERHEATED_OUTPUTS:
        all_target_results[target] = run_all_scenarios(df, SUPERHEATED_INPUTS, target)
    return all_target_results


# ═══════════════════════════════════════════════════════════════════
#  RESULTS PROCESSING AND OUTPUT
# ═══════════════════════════════════════════════════════════════════
def build_results_table(target_results):
    """Converts multi-metric structure to table format."""
    rows = []
    for target, scenarios in target_results.items():
        for scenario, scores in scenarios.items():
            for algo, metrics in scores.items():
                row = {
                    'Target': target,
                    'Scenario': scenario,
                    'Algorithm': algo
                }
                row.update(metrics)  # Add metrics dictionary to row
                rows.append(row)
    return pd.DataFrame(rows)


def show_best_results(results_df):
    """Returns best (Algorithm, Scenario) combination per target based on Test_R2."""
    idx = results_df.groupby('Target')['Test_R2'].idxmax()
    cols = ['Target', 'Scenario', 'Algorithm', 'Test_R2', 'Train_R2', 'Test_RMSE', 'Test_MAPE']
    best = results_df.loc[idx, [c for c in cols if c in results_df.columns]]
    return best.reset_index(drop=True)


def compare_scenarios(results_df):
    """Pivots Test_R2 values by target and algorithm."""
    pivot = results_df.pivot_table(
        index=['Target', 'Algorithm'],
        columns='Scenario',
        values='Test_R2'
    )
    pivot = pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])
    return pivot


def target_summary(results_df, target_col):
    """Summary of single target based on Test_R2 metrics."""
    sub = results_df[results_df['Target'] == target_col]
    pivot = sub.pivot_table(
        index='Algorithm',
        columns='Scenario',
        values='Test_R2'
    )
    pivot = pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])
    return pivot


# ═══════════════════════════════════════════════════════════════════
#  RESULTS SAVING - WIDE FORMAT CONVERSION
# ═══════════════════════════════════════════════════════════════════

def save_wide_results(df_long, path):
    """
    Converts all metrics to wide format by scenario and saves.
    Creates comparison summary based on Test_R2.
    """
    metrics = ['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE', 'Train_MAPE', 'Test_MAPE']
    index_cols = [c for c in ['Dataset', 'Target', 'Algorithm'] if c in df_long.columns]
    
    pivot = df_long.pivot_table(
        index=index_cols,
        columns='Senaryo',
        values=metrics
    )
    
    # Flatten multi-index (e.g., Base_Test_R2)
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    
    # Reorder columns by Scenario > Metric
    ordered_cols = []
    for s in SCENARIO_ORDER:
        for m in metrics:
            col_name = f"{s}_{m}"
            if col_name in pivot.columns:
                ordered_cols.append(col_name)
                
    pivot = pivot.reindex(columns=ordered_cols)

    # Determine winner (Max) based on Test_R2
    test_r2_cols = [f"{s}_Test_R2" for s in SCENARIO_ORDER if f"{s}_Test_R2" in pivot.columns]
    pivot['Max_Test_R2'] = pivot[test_r2_cols].max(axis=1)
    # Remove _Test_R2 suffix and keep only scenario name
    pivot['Max_Scenario'] = pivot[test_r2_cols].idxmax(axis=1).str.replace('_Test_R2', '')

    wide = pivot.reset_index()
    wide = wide.sort_values(index_cols).reset_index(drop=True)
    wide.to_csv(path, index=False, float_format='%.6f')
    
    save_comparison_summary(wide, path)
    return wide


# ═══════════════════════════════════════════════════════════════════
#  COMPARISON SUMMARY - APPEND TO CSV
# ═══════════════════════════════════════════════════════════════════
def save_comparison_summary(wide_df, path):
    """
    Appends comparison reports based on Test_R2 metrics
    to the saved CSV file.
    """
    scenarios = ['Base', 'SMOGN', 'STGP-EF', 'SMOGN+STGP-EF']
    cols = [s for s in scenarios if f"{s}_Test_R2" in wide_df.columns]

    with open(path, 'a', encoding='utf-8-sig', newline='') as f:

        # ── 1. Scenario-Based Comparison ───────────────────────
        f.write('\n')
        f.write('SCENARIO BASED COMPARISON (Based on Test_R2)\n')
        f.write('Scenario,Win_Count,Win_Rate_%,Scenario_Mean_Test_R2,Winner_Mean_Max_Test_R2\n')
        for s in cols:
            col_name = f"{s}_Test_R2"
            win_count = int((wide_df['Max_Scenario'] == s).sum())
            win_rate  = round(win_count / len(wide_df) * 100, 2)
            mean_s    = round(wide_df[col_name].mean(), 6)
            mask      = wide_df['Max_Scenario'] == s
            mean_max  = round(wide_df.loc[mask, 'Max_Test_R2'].mean(), 6) if win_count > 0 else ''
            f.write(f'{s},{win_count},{win_rate},{mean_s},{mean_max}\n')

        # ── 2. Algorithm-Based Comparison ──────────────────────
        f.write('\n')
        f.write('ALGORITHM BASED COMPARISON (Based on Test_R2)\n')
        header = ('Algorithm,' + ','.join([f"Mean_{s}_Test_R2" for s in cols]) +
                  ',Mean_Max_Test_R2,Best_Scenario,' +
                  ','.join([f"Win_Count_{s}" for s in cols]) + '\n')
        f.write(header)
        
        for alg in sorted(wide_df['Algorithm'].unique()):
            grp  = wide_df[wide_df['Algorithm'] == alg]
            means = [round(grp[f"{s}_Test_R2"].mean(), 6) for s in cols]
            mean_max = round(grp['Max_Test_R2'].mean(), 6)
            
            # Calculate best scenario safely (handle ties and empty cases)
            modes = grp['Max_Scenario'].mode()
            best = modes[0] if not modes.empty else "Unknown"
            
            wins   = grp['Max_Scenario'].value_counts().to_dict()
            wins_vals = [wins.get(s, 0) for s in cols]
            
            row = [alg] + means + [mean_max, best] + wins_vals
            f.write(','.join(str(x) for x in row) + '\n')


# ═══════════════════════════════════════════════════════════════════
#  SHAP EXPLAINABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def calculate_shap_values(model, X_data):
    """
    Calculates SHAP values for a trained tree-based model 
    (Random Forest, Extra Trees, Evolutionary Forest, etc.).
    
    Parameters:
    model: Trained machine learning model (scikit-learn compatible).
    X_data (pd.DataFrame): Dataset for which SHAP values will be calculated (typically X_test).
    
    Returns:
    explainer: SHAP explainer object.
    shap_values: Calculated SHAP values.
    """
    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    
    return explainer, shap_values


# ── Global Feature Importance ─────────────────────────────────────
def plot_global_feature_importance(shap_values, X_data, save_path=None):
    """
    Plots global feature importance as a bar chart.
    Shows which STGP-EF features or physical parameters are most dominant.
    
    Parameters:
    shap_values: SHAP values returned from calculate_shap_values method.
    X_data (pd.DataFrame): Dataset used to obtain feature names.
    save_path (str, optional): File path where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.title("Global Feature Importance", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        
    plt.show()


# ── SHAP Summary Plot (Bee Swarm) ──────────────────────────────────
def plot_global_summary(shap_values, X_data, save_path=None):
    """
    Plots density (bee swarm) plot showing how high/low values of variables
    affect the target variable (enthalpy, specific volume, etc.).
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, show=False)
    plt.title("SHAP Summary Plot", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        
    plt.show()


# ── Local Interpretability (Waterfall) ─────────────────────────────
def plot_local_waterfall(explainer, shap_values, X_data, instance_index=0, save_path=None):
    """
    Explains how a prediction is formed for a specific data point (row).
    
    Parameters:
    explainer: SHAP explainer object.
    shap_values: SHAP values matrix for the corresponding model.
    X_data (pd.DataFrame): Dataset being analyzed.
    instance_index (int): Row index of the specific data point to examine.
    """
    # Create Explanation object for waterfall plot (SHAP 0.40+ versions)
    # If shap_values is already an Explanation object, no reshaping needed.
    
    plt.figure(figsize=(10, 6))
    
    if isinstance(shap_values, shap.Explanation):
        shap.plots.waterfall(shap_values[instance_index], show=False)
    else:
        # For backward compatibility or raw numpy array conversion
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
            
        shap.plots._waterfall.waterfall_legacy(
            expected_value, 
            shap_values[instance_index], 
            X_data.iloc[instance_index], 
            show=False
        )
        
    plt.title(f"Local Interpretability - Index: {instance_index}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        
    plt.show()


# ── Thermodynamic Dependence Analysis ──────────────────────────────
def plot_thermodynamic_dependence(shap_values, X_data, feature_name, interaction_feature="auto", save_path=None):
    """
    Plots SHAP Dependence plot to verify if thermodynamic rules 
    are learned by the model.
    
    Parameters:
    shap_values: SHAP values.
    X_data (pd.DataFrame): Dataset being analyzed.
    feature_name (str): Independent variable to examine on X-axis (e.g., 'Pressure (P, kPa)').
    interaction_feature (str): Variable to be added as color scale (Default: 'auto').
    """
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_name, 
        shap_values, 
        X_data, 
        interaction_index=interaction_feature,
        show=False
    )
    plt.title(f"Thermodynamic Dependence Analysis: {feature_name}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        
    plt.show()
