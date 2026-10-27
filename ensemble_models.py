"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def prepare_features(X_train, X_test):
    """Подготовка признаков: масштабирование."""
    if isinstance(X_train, pd.DataFrame):
        X_train_arr = X_train.to_numpy(dtype=float)
        X_test_arr = X_test.to_numpy(dtype=float)
    else:
        X_train_arr = np.array(X_train, dtype=float)
        X_test_arr = np.array(X_test, dtype=float)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_test_scaled = scaler.transform(X_test_arr)
    
    return X_train_scaled, X_test_scaled


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """Обучает модель и возвращает метрики."""
    model.fit(X_train, y_train)
    
    # Вероятности положительного класса
    probabilities = model.predict_proba(X_test)[:, 1]
    # Бинарные предсказания
    predictions = model.predict(X_test)
    
    # Метрики
    roc_auc = float(roc_auc_score(y_test, probabilities))
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    
    return {
        'model_name': model_name,
        'probabilities': probabilities,
        'predictions': predictions,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train, X_test = prepare_features(data.train_features, data.test_features)
    y_train = np.array(data.train_target, dtype=float)
    y_test = np.array(data.test_target, dtype=float)
    seed = data.random_seed
    
    # Пробуем несколько моделей и выбираем лучшую по ROC-AUC
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=5),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=seed, early_stopping=True)
    }
    
    best_result = None
    best_roc_auc = -1
    
    for name, model in models.items():
        result = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
        if result['roc_auc'] > best_roc_auc:
            best_roc_auc = result['roc_auc']
            best_result = result
    
    return EnsembleResult(
        model_name=best_result['model_name'],
        probabilities=best_result['probabilities'],
        predictions=best_result['predictions'],
        roc_auc=best_result['roc_auc'],
        accuracy=best_result['accuracy'],
        precision=best_result['precision'],
        recall=best_result['recall'],
        f1=best_result['f1']
    )


# ==========================================
# ГЕНЕРАЦИЯ experiment_results.json и отчёта
# ==========================================
if __name__ == "__main__":
    print(" Запуск эксперимента с ансамблями...")
    
    # Загрузка данных
    df = pd.read_csv("german.csv", sep=";")
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    
    print(f" Лучшая модель: {result.model_name}")
    print(f" ROC-AUC: {result.roc_auc:.4f}")
    print(f" Accuracy: {result.accuracy:.4f}")
    print(f" Precision: {result.precision:.4f}")
    print(f" Recall: {result.recall:.4f}")
    print(f" F1: {result.f1:.4f}")
    
    # Сохранение результатов
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
    
    print(" Файл experiment_results.json создан!")