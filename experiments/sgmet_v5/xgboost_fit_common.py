"""Shared XGBoost fitting rules from the original Stage 1 runner."""
import numpy as np
import optuna
import xgboost as xgb

SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 50
TARGET_LABELS = {
    "arthritis": "Arthritis",
    "asthma": "Asthma",
    "thyroid_condition": "Thyroid",
    "any_cancer": "Any cancer",
    "copd": "COPD",
    "gout": "Gout",
    "liver_condition": "Liver",
    "stroke": "Stroke",
    "myocardial_infarction": "MI",
    "emphysema": "Emphysema",
    "chd": "CHD",
}
def subset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y)
    return X[mask], y[mask].astype("int8")


def params_from_trial(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 0.8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
    }


def fit_model(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = N_ESTIMATORS,
) -> xgb.XGBClassifier:
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=n_estimators,
        early_stopping_rounds=min(EARLY_STOPPING, max(5, n_estimators // 4)),
        n_jobs=1,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
