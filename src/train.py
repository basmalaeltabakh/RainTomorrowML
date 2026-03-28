import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import RANDOM_STATE, TEST_SIZE, MODELS_DIR
from src.preprocessing import full_pipeline


def get_models(scale_pos: float) -> dict:
    return {
        # ── Bagging
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'BaggingDT': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, class_weight='balanced'),
            n_estimators=100, max_samples=0.8, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        # ── Boosting
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
        ),
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=4),
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            auto_class_weights='Balanced',
            random_seed=RANDOM_STATE, verbose=0
        ),
    }


def train_all(data_path: str):
    print("Loading & preprocessing...")
    X, y = full_pipeline(data_path, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    models    = get_models(scale_pos)
    results   = []

    for name, model in models.items():
        print(f"Training {name}...", end=' ')
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'model'    : name,
            'type'     : 'Bagging' if name in ['RandomForest','ExtraTrees','BaggingDT'] else 'Boosting',
            'accuracy' : round(accuracy_score(y_test, y_pred), 4),
            'f1'       : round(f1_score(y_test, y_pred), 4),
            'roc_auc'  : round(roc_auc_score(y_test, y_proba), 4),
        }
        results.append(metrics)
        print(f"AUC={metrics['roc_auc']}  F1={metrics['f1']}")

        # Save each model
        with open(MODELS_DIR / f'{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Save results summary
    results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    results_df.to_csv(MODELS_DIR / 'results.csv', index=False)

    # Save test sets for comparison page
    with open(MODELS_DIR / 'test_data.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)

    print("\nDone! Best model:", results_df.iloc[0]['model'])
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    from src.config import DATA_RAW
    train_all(str(DATA_RAW))