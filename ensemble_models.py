"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.array(X, dtype=float)


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    seed = data.random_seed

    X_train_full = to_numpy(data.train_features)
    y_train_full = np.asarray(data.train_target)
    X_test = to_numpy(data.test_features)
    y_test = np.asarray(data.test_target)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed, stratify=y_train_full
    )

    scaler_val = StandardScaler()
    X_fit_scaled = scaler_val.fit_transform(X_fit)
    X_val_scaled = scaler_val.transform(X_val)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=seed, early_stopping=True)
    }

    best_name = None
    best_model_template = None
    best_roc_auc = -1.0

    for name, model in models.items():
        model.fit(X_fit_scaled, y_fit)
        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        val_roc_auc = roc_auc_score(y_val, val_probs)
        print(f"[Валидация] {name} ROC-AUC: {val_roc_auc:.4f}")

        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            best_name = name
            best_model_template = model

    print(f"-> Выбрана лучшая модель: {best_name}")

    scaler_final = StandardScaler()
    X_train_scaled_final = scaler_final.fit_transform(X_train_full)
    X_test_scaled_final = scaler_final.transform(X_test)

    final_model = clone(best_model_template)
    final_model.fit(X_train_scaled_final, y_train_full)

    probabilities = final_model.predict_proba(X_test_scaled_final)[:, 1]
    predictions = final_model.predict(X_test_scaled_final)

    return EnsembleResult(
        model_name=best_name,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0))
    )


if __name__ == "__main__":
    print("Запуск эксперимента с ансамблями (с валидационным сплитом)...")

    df = pd.read_csv("german.csv", sep=";")
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class MockInput:
        def __init__(self):
            self.train_features = X_train
            self.train_target = y_train.values
            self.test_features = X_test
            self.test_target = y_test.values
            self.random_seed = 42

    data = MockInput()
    result = run_ensemble_experiment(data)

    print(f"\nИТОГОВЫЕ МЕТРИКИ НА ТЕСТЕ ({result.model_name}):")
    print(f"ROC-AUC:   {result.roc_auc:.4f}")
    print(f"Accuracy:  {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1:        {result.f1:.4f}")

    results_dict = {
        "best_model": result.model_name,
        "metrics": {
            "roc_auc": result.roc_auc,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1
        }
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    print("\nФайл experiment_results.json обновлён!")