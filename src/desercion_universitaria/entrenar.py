"""
Entrena el modelo de deserción universitaria y lo guarda en /modelo.

Uso:
    uv run entrenar

Aplica las decisiones del "Resumen de decisiones" de la Entrega 1:
  * Partición estratificada (80 % entrenamiento / 20 % prueba).
  * Se comparan los modelos de la propuesta (regresión logística, árbol de decisión,
    random forest y perceptrón multicapa) con validación cruzada estratificada de 5 pliegues.
  * El desbalance se trata con class_weight="balanced" y, si está instalado
    imbalanced-learn, también con SMOTE dentro del pipeline (solo sobre entrenamiento).
  * Se elige el mejor por F1 macro y se reportan balanced accuracy y recall por clase.
"""
import json
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             make_scorer, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from desercion_universitaria.core import (BINARIAS, CARPETA_MODELO, CATEGORICAS, CLASES,
                                                   COLUMNAS, MACRO, NUMERICAS, RUTA_DATOS,
                                                   RUTA_METRICAS, RUTA_MODELO, SEMILLA, leer_csv)

try:  # SMOTE es opcional:  uv add imbalanced-learn
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as PipelineImb
except ImportError:
    SMOTE = None


def preprocesador() -> ColumnTransformer:
    """
    Categóricas -> one-hot. Las categorías con menos del 1 % de los datos (las "raras" del EDA)
    se agrupan en una sola columna, y un código nunca visto cae en ese mismo grupo.
    Numéricas y binarias -> estandarización.
    """
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                              sparse_output=False), CATEGORICAS),
        ("num", StandardScaler(), NUMERICAS + BINARIAS),
    ])


def crear_modelo(nombre: str, pesos: bool):
    """pesos=True -> class_weight="balanced" (el MLP no lo admite: recibe sample_weight al entrenar)."""
    cw = "balanced" if pesos else None
    return {
        "Regresión logística": lambda: LogisticRegression(max_iter=3000, class_weight=cw),
        "Árbol de decisión": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                            class_weight=cw, random_state=SEMILLA),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                        class_weight=cw, n_jobs=-1,
                                                        random_state=SEMILLA),
        "Perceptrón multicapa": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                      early_stopping=True, max_iter=500,
                                                      random_state=SEMILLA),
    }[nombre]()


ALGORITMOS = ["Regresión logística", "Árbol de decisión", "Random Forest", "Perceptrón multicapa"]


def crear_pipeline(nombre: str, balanceo: str):
    modelo = crear_modelo(nombre, pesos=(balanceo == "class_weight"))
    if balanceo == "SMOTE":
        return PipelineImb([("prep", preprocesador()), ("smote", SMOTE(random_state=SEMILLA)),
                            ("modelo", modelo)])
    return Pipeline([("prep", preprocesador()), ("modelo", modelo)])


def parametros_fit(nombre: str, balanceo: str, y) -> dict:
    """El MLP no tiene class_weight: le pasamos pesos por muestra equivalentes."""
    if nombre == "Perceptrón multicapa" and balanceo == "class_weight":
        return {"modelo__sample_weight": compute_sample_weight("balanced", y)}
    return {}


METRICAS_CV = {
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "recall_dropout": make_scorer(recall_score, labels=["Dropout"], average="macro"),
}


def datos_formulario(X: pd.DataFrame) -> dict:
    """Opciones y valores típicos para construir el formulario de la app sin leer el CSV."""
    opciones = {c: [int(v) for v in X[c].value_counts().index] for c in CATEGORICAS}
    tipicos = {c: int(X[c].mode()[0]) for c in CATEGORICAS + BINARIAS}
    for c in NUMERICAS:
        mediana = X[c].median()
        tipicos[c] = int(round(mediana)) if pd.api.types.is_integer_dtype(X[c]) else round(float(mediana), 1)
    macro = (X[MACRO].value_counts().rename("n").reset_index()
             .sort_values(MACRO).to_dict(orient="records"))
    mas_comun = X[MACRO].value_counts().index[0]
    tipicos.update({k: float(v) for k, v in zip(MACRO, mas_comun)})
    return {"opciones": opciones, "tipicos": tipicos, "macro": macro}


def main():
    if not RUTA_DATOS.exists():
        raise SystemExit(f"Falta el dataset. Copia el data.csv de UCI #697 en: {RUTA_DATOS}")

    df = leer_csv(RUTA_DATOS)
    X, y = df[COLUMNAS], df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
    print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas | "
          f"{len(COLUMNAS)} variables\n")

    balanceos = ["class_weight"] + (["SMOTE"] if SMOTE else [])
    if not SMOTE:
        print("ℹ️  imbalanced-learn no está instalado: se compara solo class_weight.\n"
              "    Para incluir SMOTE:  uv add imbalanced-learn\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    comparacion = []
    print(f"{'Modelo':<22}{'Balanceo':<14}{'F1 macro':>10}{'Bal. acc.':>11}{'Recall Dropout':>16}")
    for nombre in ALGORITMOS:
        for balanceo in balanceos:
            inicio = time.perf_counter()
            r = cross_validate(crear_pipeline(nombre, balanceo), X_train, y_train, cv=cv,
                               scoring=METRICAS_CV, params=parametros_fit(nombre, balanceo, y_train),
                               n_jobs=-1)
            fila = {"modelo": nombre, "balanceo": balanceo,
                    "f1_macro": round(r["test_f1_macro"].mean(), 4),
                    "f1_macro_std": round(r["test_f1_macro"].std(), 4),
                    "balanced_accuracy": round(r["test_balanced_accuracy"].mean(), 4),
                    "recall_dropout": round(r["test_recall_dropout"].mean(), 4),
                    "segundos": round(time.perf_counter() - inicio, 1)}
            comparacion.append(fila)
            print(f"{nombre:<22}{balanceo:<14}{fila['f1_macro']:>10.3f}"
                  f"{fila['balanced_accuracy']:>11.3f}{fila['recall_dropout']:>16.3f}")

    mejor = max(comparacion, key=lambda f: f["f1_macro"])
    print(f"\n🏆 Mejor en validación cruzada: {mejor['modelo']} + {mejor['balanceo']} "
          f"(F1 macro {mejor['f1_macro']:.3f})")

    # Reentrenar el mejor con todo el conjunto de entrenamiento y evaluar en prueba
    pipeline = crear_pipeline(mejor["modelo"], mejor["balanceo"])
    pipeline.fit(X_train, y_train, **parametros_fit(mejor["modelo"], mejor["balanceo"], y_train))
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)
    clases = list(pipeline.classes_)
    prueba = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "auc_ovr": round(roc_auc_score(y_test, proba, multi_class="ovr", labels=clases), 4),
        "recall": dict(zip(CLASES, np.round(recall_score(y_test, pred, labels=CLASES, average=None), 4).tolist())),
        "precision": dict(zip(CLASES, np.round(precision_score(y_test, pred, labels=CLASES, average=None,
                                                               zero_division=0), 4).tolist())),
    }
    print(f"\nEn prueba ({len(X_test)} estudiantes nunca vistos):")
    print(f"  F1 macro = {prueba['f1_macro']:.3f} | balanced accuracy = {prueba['balanced_accuracy']:.3f} | "
          f"accuracy = {prueba['accuracy']:.3f} | AUC = {prueba['auc_ovr']:.3f}")
    print("  Recall por clase: " + ", ".join(f"{c} {v:.2f}" for c, v in prueba["recall"].items()))

    print("\nCalculando la importancia de cada variable (permutación)...")
    imp = permutation_importance(pipeline, X_test, y_test, scoring="f1_macro", n_repeats=5,
                                 random_state=SEMILLA, n_jobs=-1)
    importancias = (pd.Series(imp.importances_mean, index=COLUMNAS)
                    .sort_values(ascending=False).round(4).to_dict())

    CARPETA_MODELO.mkdir(exist_ok=True)
    joblib.dump(pipeline, RUTA_MODELO, compress=3)
    metricas = {
        "modelo": mejor["modelo"],
        "balanceo": mejor["balanceo"],
        "fecha": date.today().isoformat(),
        "filas_entrenamiento": len(X_train),
        "filas_prueba": len(X_test),
        "variables": COLUMNAS,
        "comparacion": comparacion,
        "prueba": prueba,
        "clases": CLASES,
        "matriz_confusion": confusion_matrix(y_test, pred, labels=CLASES).tolist(),
        "importancias": importancias,
        "indices_prueba": [int(i) for i in X_test.index],
        "formulario": datos_formulario(X),
    }
    RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Guardado el modelo ({mejor['modelo']} + {mejor['balanceo']}) en {CARPETA_MODELO}/")


if __name__ == "__main__":
    main()
