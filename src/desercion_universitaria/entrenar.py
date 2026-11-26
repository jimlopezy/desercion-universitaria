"""
Entrena el modelo de deserción universitaria y lo guarda en /modelo.

Uso:
    uv run entrenar            # usa los hiperparámetros que eligió la Entrega 2
    uv run entrenar --buscar   # repite la búsqueda aleatoria de hiperparámetros

Replica el modelado de la Entrega 2 (Entrega2Proyecto.ipynb):
  1. Ingeniería de características (core.datos_modelado) y partición estratificada 80/20.
  2. Balanceo de clases: compara métodos con regresión logística (validación cruzada de 5 pliegues).
     Sin imbalanced-learn solo se comparan "sin_balanceo" y "class_weight".
  3. Modelos con categóricas nativas: HistGradientBoosting (siempre) y LightGBM / CatBoost
     (si están instalados), con los hiperparámetros que encontró la búsqueda del notebook
     (RandomizedSearchCV, 15 combinaciones, Average Precision). Con --buscar se repite la búsqueda.
  4. Evaluación en los mismos 5 pliegues (media e IC95 %) con la regresión logística como base.
  5. Se elige el mejor por AP en validación cruzada, se evalúa en prueba con umbral 0,5
     y se guarda el artefacto para la app.
"""
import json
import sys
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from scipy.stats import t as t_dist
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from desercion_universitaria.core import (BINARIAS, CARPETA_MODELO, CATEGORICAS, COLUMNAS, ESCALAS,
                                          NUMERICAS, RANGOS, RUTA_DATOS, RUTA_METRICAS, RUTA_MODELO,
                                          SEMILLA, UMBRAL, a_categorias, cargar_datos, datos_modelado)

N_ITER = 15

# Mejores hiperparámetros de la Entrega 2 (salida de la celda de RandomizedSearchCV del notebook).
# Son los valores exactos que muestrea RandomizedSearchCV(random_state=42); el notebook los imprime redondeados.
PARAMETROS_ENTREGA2 = {
    "HistGB": {"l2_regularization": 0.017654048052495083, "learning_rate": 0.04749239763680407,
               "max_iter": 205, "max_leaf_nodes": 11, "min_samples_leaf": 63},
    "LightGBM": {"colsample_bytree": 0.9687496940092467, "learning_rate": 0.01303561122512888,
                 "min_child_samples": 71, "n_estimators": 395, "num_leaves": 28,
                 "reg_lambda": 0.03586816498627549, "subsample": 0.7085396127095583, "subsample_freq": 1},
    "CatBoost": {"depth": 8, "iterations": 543, "l2_leaf_reg": 12.10689693600216,
                 "learning_rate": 0.018891200276189388},
}
AP_CV_ENTREGA2 = {"HistGB": 0.8425, "LightGBM": 0.8414, "CatBoost": 0.8425}
SCORING = {"auc": "roc_auc", "ap": "average_precision", "recall": "recall",
           "precision": "precision", "f1": "f1", "bal_acc": "balanced_accuracy"}
METRICAS = list(SCORING)


def media_ci(valores):
    arr = np.asarray(valores, dtype=float)
    media = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    tcrit = t_dist.ppf(0.975, df=len(arr) - 1)
    return float(media), float(media - tcrit * se), float(media + tcrit * se)


def resumen_cv(r) -> dict:
    return {k: dict(zip(["media", "ic_inf", "ic_sup"], [round(v, 4) for v in media_ci(r[f"test_{k}"])]))
            for k in METRICAS}


def preprocesador_logit() -> ColumnTransformer:
    num = [c for c in COLUMNAS if c not in CATEGORICAS]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("num", StandardScaler(), num),
    ])


def metodos_balanceo() -> dict:
    metodos = {"sin_balanceo": None, "class_weight": "cw"}
    try:
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        metodos.update({
            "random_over": RandomOverSampler(random_state=SEMILLA),
            "random_under": RandomUnderSampler(random_state=SEMILLA),
            "smote": SMOTE(random_state=SEMILLA), "adasyn": ADASYN(random_state=SEMILLA),
            "borderline_smote": BorderlineSMOTE(random_state=SEMILLA),
            "smote_tomek": SMOTETomek(random_state=SEMILLA), "smote_enn": SMOTEENN(random_state=SEMILLA),
        })
    except ImportError:
        print("ℹ️  imbalanced-learn no está instalado: el balanceo compara solo sin_balanceo y class_weight.\n"
              "    Para comparar los 9 métodos del notebook:  uv add imbalanced-learn\n")
    return metodos


def pipeline_logit(sampler):
    modelo = LogisticRegression(max_iter=3000)
    if sampler == "cw":
        modelo.set_params(class_weight="balanced")
    if sampler is None or sampler == "cw":
        return Pipeline([("pre", preprocesador_logit()), ("model", modelo)])
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([("pre", preprocesador_logit()), ("sampler", sampler), ("model", modelo)])


def modelos_boosting() -> dict:
    """Modelos base y espacios de búsqueda del notebook (opción B: categóricas nativas)."""
    modelos = {
        "HistGB": (HistGradientBoostingClassifier(
            categorical_features="from_dtype", class_weight="balanced", early_stopping=False,
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31, random_state=SEMILLA), {
            "learning_rate": loguniform(0.01, 0.2), "max_iter": randint(100, 600),
            "max_leaf_nodes": randint(8, 64), "min_samples_leaf": randint(10, 80),
            "l2_regularization": loguniform(1e-3, 10)}, False),
    }
    try:
        from lightgbm import LGBMClassifier
        modelos["LightGBM"] = (LGBMClassifier(
            class_weight="balanced", n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=SEMILLA, n_jobs=-1, verbose=-1), {
            "learning_rate": loguniform(0.01, 0.2), "n_estimators": randint(100, 600),
            "num_leaves": randint(8, 64), "min_child_samples": randint(10, 80),
            "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
            "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": loguniform(1e-3, 10)}, False)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        modelos["CatBoost"] = (CatBoostClassifier(
            cat_features=CATEGORICAS, auto_class_weights="Balanced", iterations=300, learning_rate=0.05,
            depth=6, random_seed=SEMILLA, thread_count=-1, verbose=0, allow_writing_files=False), {
            "learning_rate": loguniform(0.01, 0.2), "iterations": randint(200, 800),
            "depth": randint(4, 9), "l2_leaf_reg": loguniform(1, 20)}, True)
    except ImportError:
        pass
    if len(modelos) < 3:
        print("ℹ️  LightGBM y/o CatBoost no están instalados: se entrena con los disponibles.\n"
              "    Para comparar los 3 modelos del notebook:  uv add lightgbm catboost\n")
    return modelos


def curva(x, y, n=60):
    """Reduce una curva a ~n puntos para guardarla en el JSON."""
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return {"x": np.round(np.asarray(x)[idx], 4).tolist(), "y": np.round(np.asarray(y)[idx], 4).tolist()}


def datos_formulario(X_train: pd.DataFrame, df_crudo: pd.DataFrame) -> dict:
    """Opciones (códigos UCI) y valores típicos para el formulario de la app."""
    crudo = df_crudo.loc[X_train.index]
    opciones, tipicos = {}, {}
    for c in CATEGORICAS:
        validos = crudo[c][X_train[c].notna()]
        opciones[c] = [int(v) for v in validos.value_counts().index]
    for c, escala in ESCALAS.items():
        opciones[c] = [int(v) for v in crudo[c].value_counts().index if int(v) in escala]
    for c in CATEGORICAS + list(ESCALAS) + BINARIAS:
        tipicos[c] = int(crudo[c].mode()[0])
    for c in RANGOS:
        mediana = X_train[c].median()
        tipicos[c] = int(round(mediana)) if pd.api.types.is_integer_dtype(X_train[c]) else round(float(mediana), 1)
    return {"opciones": opciones, "tipicos": tipicos}


def main():
    if not RUTA_DATOS.exists():
        raise SystemExit(f"Falta el dataset. Copia el data.csv de UCI #697 en: {RUTA_DATOS}")

    crudo = cargar_datos()
    X, y = datos_modelado(crudo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
    categorias = {c: list(X_train[c].cat.categories.union(X_test[c].cat.categories, sort=False))
                  for c in CATEGORICAS}
    X_train, X_test = a_categorias(X_train, categorias), a_categorias(X_test, categorias)
    print(f"Datos de modelado: {len(X)} estudiantes de {len(crudo)} | train {X_train.shape} | test {X_test.shape}")
    print(f"  y_train {y_train.value_counts().to_dict()} | y_test {y_test.value_counts().to_dict()}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    # ---- 1. Balanceo de clases con regresión logística ----
    balanceo, cv_logit = [], {}
    print(f"{'Método':<18}{'AUC':>8}{'AP':>8}{'Recall':>8}{'Prec.':>8}{'F1':>8}{'BalAcc':>8}")
    for nombre, sampler in metodos_balanceo().items():
        r = cross_validate(pipeline_logit(sampler), X_train, y_train, cv=skf, scoring=SCORING, n_jobs=-1)
        cv_logit[nombre] = r
        fila = {"metodo": nombre, **{k: v["media"] for k, v in resumen_cv(r).items()}}
        balanceo.append(fila)
        print(f"{nombre:<18}" + "".join(f"{fila[k]:>8.4f}" for k in METRICAS))

    # ---- 2. Hiperparámetros ----
    modelos = modelos_boosting()
    datos = {n: ((X_train.astype({c: str for c in CATEGORICAS}), X_test.astype({c: str for c in CATEGORICAS}))
                 if texto else (X_train, X_test)) for n, (_, _, texto) in modelos.items()}
    buscar = "--buscar" in sys.argv
    mejores, busqueda = {}, {}
    if buscar:
        print(f"\nBúsqueda aleatoria ({N_ITER} combinaciones, AP, 5 pliegues):")
    else:
        print("\nHiperparámetros elegidos en la Entrega 2 (para repetir la búsqueda: uv run entrenar --buscar):")
    for nombre, (base, espacio, _) in modelos.items():
        t0 = time.perf_counter()
        if buscar:
            rs = RandomizedSearchCV(clone(base), espacio, n_iter=N_ITER, scoring="average_precision", cv=skf,
                                    random_state=SEMILLA, n_jobs=-1, refit=False)
            rs.fit(datos[nombre][0], y_train)
            elegidos, ap_cv, origen = rs.best_params_, float(rs.best_score_), "búsqueda repetida"
        else:
            elegidos, ap_cv, origen = PARAMETROS_ENTREGA2[nombre], AP_CV_ENTREGA2[nombre], "Entrega 2"
        mejores[nombre] = clone(base).set_params(**elegidos)
        params = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
                      if isinstance(v, (np.integer, int)) else v) for k, v in elegidos.items()}
        busqueda[nombre] = {"ap_cv": round(ap_cv, 4), "segundos": round(time.perf_counter() - t0, 1),
                            "parametros": params, "origen": origen}
        print(f"  {nombre:<9} AP_cv = {ap_cv:.4f} | {params}")

    # ---- 3. Comparación en los mismos pliegues ----
    cv_modelos = {n: cross_validate(clone(m), datos[n][0], y_train, cv=skf, scoring=SCORING,
                                    return_train_score=True, n_jobs=-1) for n, m in mejores.items()}
    comparacion = [{"modelo": n, **resumen_cv(r), "ap_train": round(float(np.mean(r["train_ap"])), 4)}
                   for n, r in cv_modelos.items()]
    comparacion.append({"modelo": "LogReg (base, class_weight)", **resumen_cv(cv_logit["class_weight"])})
    print("\nValidación cruzada (media [IC95 %]):")
    for f in comparacion:
        print(f"  {f['modelo']:<28} AP {f['ap']['media']:.4f} [{f['ap']['ic_inf']:.4f}, {f['ap']['ic_sup']:.4f}]"
              f" | F1 {f['f1']['media']:.4f} | AUC {f['auc']['media']:.4f}")

    # ---- 4. Evaluación en prueba ----
    prueba, finales, probas = [], {}, {}
    for nombre, modelo in mejores.items():
        Xtr, Xte = datos[nombre]
        m = clone(modelo).fit(Xtr, y_train)
        p = m.predict_proba(Xte)[:, 1]
        yhat = (p >= UMBRAL).astype(int)
        finales[nombre], probas[nombre] = m, p
        prueba.append({"modelo": nombre, "auc": round(roc_auc_score(y_test, p), 4),
                       "ap": round(average_precision_score(y_test, p), 4),
                       "recall": round(recall_score(y_test, yhat), 4),
                       "precision": round(precision_score(y_test, yhat), 4),
                       "f1": round(f1_score(y_test, yhat), 4),
                       "bal_acc": round(balanced_accuracy_score(y_test, yhat), 4)})
    print(f"\nPrueba ({len(X_test)} estudiantes, umbral {UMBRAL}):")
    for f in prueba:
        print(f"  {f['modelo']:<9} " + " | ".join(f"{k.upper()} {f[k]:.4f}" for k in METRICAS))

    mejor = max(cv_modelos, key=lambda n: np.mean(cv_modelos[n]["test_ap"]))
    print(f"\n🏆 Mejor modelo por AP en validación cruzada: {mejor}")

    print("Calculando la importancia por permutación (AP, 10 repeticiones)...")
    Xte = datos[mejor][1]
    imp = permutation_importance(finales[mejor], Xte, y_test, scoring="average_precision",
                                 n_repeats=10, random_state=SEMILLA, n_jobs=-1)
    importancias = (pd.DataFrame({"variable": Xte.columns, "media": imp.importances_mean.round(4),
                                  "desv": imp.importances_std.round(4)})
                    .sort_values("media", ascending=False).to_dict(orient="records"))

    p = probas[mejor]
    fpr, tpr, _ = roc_curve(y_test, p)
    prec, rec, _ = precision_recall_curve(y_test, p)

    CARPETA_MODELO.mkdir(exist_ok=True)
    artefacto = {"modelo": finales[mejor], "nombre": mejor, "columnas": list(X_train.columns),
                 "categoricas": CATEGORICAS, "categorias": categorias,
                 "requiere_texto": mejor == "CatBoost", "umbral": UMBRAL}
    joblib.dump(artefacto, RUTA_MODELO)
    metricas = {
        "modelo": mejor, "umbral": UMBRAL, "fecha": date.today().isoformat(),
        "filas": {"total": len(crudo), "modelado": len(X), "entrenamiento": len(X_train), "prueba": len(X_test)},
        "clases_entrenamiento": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "balanceo": balanceo, "busqueda": busqueda, "comparacion": comparacion, "prueba": prueba,
        "matriz_confusion": confusion_matrix(y_test, (p >= UMBRAL).astype(int), labels=[0, 1]).tolist(),
        "curva_roc": curva(fpr, tpr), "curva_pr": curva(rec[::-1], prec[::-1]),
        "importancias": importancias,
        "indices_prueba": [int(i) for i in X_test.index],
        "formulario": datos_formulario(X_train, crudo),
    }
    RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Guardado el modelo ({mejor}) en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
