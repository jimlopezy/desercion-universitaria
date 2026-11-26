"""
Lógica compartida del sistema de ALERTA TEMPRANA DE DESERCIÓN UNIVERSITARIA.

El modelo vive aquí, separado de la interfaz: la app web (app.py) solo arma los datos del
estudiante, llama a predecir() y muestra el resultado.

Replica la ingeniería de características de la Entrega 2 (Entrega2Proyecto.ipynb):
  * Objetivo binario: 1 = abandono (Dropout), 0 = no abandono (Enrolled o Graduate).
  * Se quitan los estudiantes sin unidades inscritas en el 1.er o en el 2.º semestre.
  * Se descartan variables irrelevantes o correlacionadas: Unemployment rate, Inflation rate, GDP,
    Nacionality, S2 (enrolled), S2 (approved), S2 (grade) y S2 (evaluations).
  * 5 categóricas nominales pasan a texto (tipo category) y 3 de escolaridad a escala ordinal.
  * Quedan 28 variables. El modelo (HistGradientBoosting) usa las categóricas de forma nativa.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Raíz del proyecto (donde está pyproject.toml): este archivo vive en
# src/desercion_universitaria/, así que subimos dos carpetas.
RAIZ = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
RUTA_DATOS = RAIZ / "data" / "data.csv"
CARPETA_MODELO = RAIZ / "modelo"
RUTA_MODELO = CARPETA_MODELO / "modelo_desercion.joblib"
RUTA_METRICAS = CARPETA_MODELO / "metricas_desercion.json"

SEMILLA = 42
UMBRAL = 0.5

CLASES = {0: "No abandono", 1: "Abandono"}
COLORES = {"Abandono": "#d95f5f", "No abandono": "#4c9be0"}

# ---------------------------------------------------------------------------
# Variables (nombres tal como quedan después de preparar())
# ---------------------------------------------------------------------------
DESCARTADAS = ["Unemployment rate", "Inflation rate", "GDP", "Nacionality",
               "S2 (enrolled)", "S2 (approved)", "S2 (grade)", "S2 (evaluations)"]

CATEGORICAS = ["Marital status", "Application mode", "Course", "Mother's occupation", "Father's occupation"]
ORDINALES = ["Previous qualification", "Mother's qualification", "Father's qualification"]
BINARIAS = ["Gender", "International", "Displaced", "Educational special needs",
            "Daytime/evening attendance", "Scholarship holder", "Debtor", "Tuition fees up to date"]

# Las 28 variables del modelo, en el orden del notebook
COLUMNAS = [
    "Marital status", "Application mode", "Course", "Mother's occupation", "Father's occupation",
    "Gender", "Age at enrollment", "International", "Displaced", "Educational special needs",
    "Application order", "Previous qualification", "Previous qualification (grade)", "Admission grade",
    "Daytime/evening attendance",
    "Mother's qualification", "Father's qualification", "Scholarship holder", "Debtor",
    "Tuition fees up to date",
    "S1 (credited)", "S1 (enrolled)", "S1 (evaluations)", "S1 (approved)", "S1 (grade)",
    "S1 (without evaluations)",
    "S2 (credited)", "S2 (without evaluations)",
]
NUMERICAS = [c for c in COLUMNAS if c not in CATEGORICAS]

# Nombres en español para la interfaz
NOMBRES = {
    "Marital status": "Estado civil",
    "Application mode": "Modo de postulación",
    "Course": "Programa",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Gender": "Género",
    "Age at enrollment": "Edad al matricularse",
    "International": "Estudiante internacional",
    "Displaced": "Vive desplazado de su residencia",
    "Educational special needs": "Necesidades educativas especiales",
    "Application order": "Orden de preferencia de la postulación (0 = primera opción)",
    "Previous qualification": "Formación previa",
    "Previous qualification (grade)": "Nota de la formación previa (0-200)",
    "Admission grade": "Nota de admisión (0-200)",
    "Daytime/evening attendance": "Jornada",
    "Mother's qualification": "Escolaridad de la madre",
    "Father's qualification": "Escolaridad del padre",
    "Scholarship holder": "Tiene beca",
    "Debtor": "Tiene deudas con la institución",
    "Tuition fees up to date": "Matrícula al día",
    "S1 (credited)": "1.er semestre · unidades convalidadas",
    "S1 (enrolled)": "1.er semestre · unidades inscritas",
    "S1 (evaluations)": "1.er semestre · evaluaciones presentadas",
    "S1 (approved)": "1.er semestre · unidades aprobadas",
    "S1 (grade)": "1.er semestre · nota promedio (0-20)",
    "S1 (without evaluations)": "1.er semestre · unidades sin evaluación",
    "S2 (credited)": "2.º semestre · unidades convalidadas",
    "S2 (without evaluations)": "2.º semestre · unidades sin evaluación",
    # usadas solo en el análisis exploratorio
    "S2 (approved)": "2.º semestre · unidades aprobadas",
    "S2 (grade)": "2.º semestre · nota promedio (0-20)",
    "S2 (enrolled)": "2.º semestre · unidades inscritas",
}

# Rangos de los controles numéricos del formulario: (mínimo, máximo, paso)
RANGOS = {
    "Application order": (0, 9, 1),
    "Previous qualification (grade)": (0.0, 200.0, 0.1),
    "Admission grade": (0.0, 200.0, 0.1),
    "Age at enrollment": (15, 80, 1),
    "S1 (credited)": (0, 30, 1), "S1 (enrolled)": (0, 30, 1), "S1 (evaluations)": (0, 50, 1),
    "S1 (approved)": (0, 30, 1), "S1 (grade)": (0.0, 20.0, 0.1), "S1 (without evaluations)": (0, 15, 1),
    "S2 (credited)": (0, 30, 1), "S2 (without evaluations)": (0, 15, 1),
}

# ---------------------------------------------------------------------------
# Equivalencias de la Entrega 2 (código UCI -> texto / nivel ordinal)
# ---------------------------------------------------------------------------
ETIQUETAS = {
    "Marital status": {1: "Soltero", 2: "Casado", 3: "Viudo", 4: "Divorciado",
                       5: "Unión de hecho", 6: "Separado legalmente"},
    "Application mode": {
        1: "1ª fase - contingente general", 2: "Ordenanza 612/93",
        5: "1ª fase - contingente especial (Azores)", 7: "Titulares de otros cursos superiores",
        10: "Ordenanza 854-B/99", 15: "Estudiante internacional",
        16: "1ª fase - contingente especial (Madeira)", 17: "2ª fase - contingente general",
        18: "3ª fase - contingente general", 26: "Ordenanza 533-A/99 (b2)",
        27: "Ordenanza 533-A/99 (b3)", 39: "Mayor de 23 años", 42: "Transferencia",
        43: "Cambio de curso", 44: "Diploma de especialización tecnológica",
        51: "Cambio de institución/curso", 53: "Diploma de ciclo corto",
        57: "Cambio de institución/curso (internacional)"},
    "Course": {
        33: "Tecnologías de biocombustibles", 171: "Animación y diseño multimedia",
        8014: "Servicio social (nocturno)", 9003: "Agronomía", 9070: "Diseño de comunicación",
        9085: "Enfermería veterinaria", 9119: "Ing. Informática", 9130: "Equinocultura",
        9147: "Gestión", 9238: "Servicio social", 9254: "Turismo", 9500: "Enfermería",
        9556: "Higiene oral", 9670: "Gestión de publicidad y marketing",
        9773: "Periodismo y comunicación", 9853: "Educación básica",
        9991: "Gestión (nocturno)"},
}
OCUPACION = {
    0: "Estudiante",
    1: "Representantes del Poder Legislativo y de Órganos Ejecutivos, Directores, Directores y Gerentes Ejecutivos",
    2: "Especialistas en Actividades Intelectuales y Científicas",
    3: "Técnicos y Profesiones de Nivel Intermedio", 4: "Personal Administrativo",
    5: "Trabajadores de Servicios Personales, de Seguridad y Vendedores",
    6: "Agricultores y Trabajadores Calificados de la Agricultura, Pesca y Silvicultura",
    7: "Trabajadores Calificados de la Industria, Construcción y Artesanos",
    8: "Operadores de Instalaciones y Máquinas y Trabajadores de Ensamblaje",
    9: "Trabajadores No Calificados", 10: "Profesiones de las Fuerzas Armadas", 90: "Otra Situación",
    99: "En blanco", 122: "Profesionales de la salud", 123: "Profesores",
    125: "Especialistas en tecnologías de la información y la comunicación (TIC)",
    131: "Técnicos y profesiones de nivel intermedio de ciencias e ingeniería",
    132: "Técnicos y profesionales de nivel intermedio de la salud",
    134: "Técnicos de nivel intermedio de servicios jurídicos, sociales, deportivos, culturales y similares",
    141: "Empleados de oficina, secretarios en general y operadores de procesamiento de datos",
    143: "Operadores de datos, contabilidad, estadística, servicios financieros y registros",
    144: "Otro personal de apoyo administrativo",
    151: "Trabajadores de servicios personales", 152: "Vendedores",
    153: "Trabajadores de cuidado personal y similares",
    171: "Trabajadores calificados de la construcción y similares, excepto electricistas",
    173: "Trabajadores calificados de la impresión, fabricación de instrumentos de precisión, joyeros, artesanos y similares",
    175: "Trabajadores de la transformación de alimentos, carpintería, confección y otras industrias y oficios",
    191: "Trabajadores de limpieza",
    192: "Trabajadores no calificados de la agricultura, producción animal, pesca y silvicultura",
    193: "Trabajadores no calificados de la industria extractiva, construcción, manufactura y transporte",
    194: "Asistentes de preparación de comidas",
}
ETIQUETAS["Mother's occupation"] = OCUPACION
ETIQUETAS["Father's occupation"] = OCUPACION

NIVEL_ACADEMICO = {  # escala ordinal de la formación previa del estudiante
    38: 0, 19: 1, 15: 2, 14: 3, 10: 4, 12: 5, 9: 6, 1: 7, 39: 8, 42: 9, 6: 10,
    2: 11, 3: 11, 40: 11, 4: 12, 43: 12, 5: 13,
}
NIVEL_ACADEMICO_PADRES = {  # escala ordinal de la escolaridad de los padres
    35: 0, 36: 1, 37: 2, 38: 3, 11: 4, 26: 5, 30: 6, 29: 7, 19: 8, 14: 9, 10: 10, 12: 11,
    27: 12, 9: 13, 1: 14, 18: 15, 22: 16, 39: 17, 42: 18, 6: 19, 2: 20, 3: 20, 40: 20,
    41: 21, 4: 22, 43: 22, 5: 23, 44: 23,
}
ESCALAS = {"Previous qualification": NIVEL_ACADEMICO,
           "Mother's qualification": NIVEL_ACADEMICO_PADRES,
           "Father's qualification": NIVEL_ACADEMICO_PADRES}

ESCOLARIDAD = {  # texto de cada código de escolaridad (para mostrar en el formulario)
    1: "Secundaria (12.º año) o equivalente", 2: "Superior: pregrado (bachelor)", 3: "Superior: grado",
    4: "Superior: maestría", 5: "Superior: doctorado", 6: "Frecuencia de educación superior",
    9: "12.º año no completado", 10: "11.º año no completado", 11: "7.º año (antiguo)",
    12: "Otro: 11.º año", 14: "10.º año", 15: "10.º año no completado", 18: "Curso de comercio general",
    19: "Básica 3.er ciclo (9.º-11.º año)", 22: "Curso técnico-profesional", 26: "7.º año",
    27: "2.º ciclo del bachillerato general", 29: "9.º año no completado", 30: "8.º año",
    35: "No sabe leer ni escribir", 36: "Sabe leer sin tener 4.º año", 37: "Básica 1.er ciclo (4.º-5.º año)",
    38: "Básica 2.º ciclo (6.º-8.º año)", 39: "Curso de especialización tecnológica",
    40: "Superior: grado (1.er ciclo)", 41: "Estudios superiores especializados",
    42: "Técnico superior profesional", 43: "Superior: maestría (2.º ciclo)",
    44: "Superior: doctorado (3.er ciclo)",
}

TEXTO_BINARIAS = {
    "Tuition fees up to date": {0: "No al día", 1: "Al día"},
    "Scholarship holder": {0: "Sin beca", 1: "Con beca"},
    "Debtor": {0: "Sin deuda", 1: "Con deuda"},
    "Gender": {0: "Femenino", 1: "Masculino"},
    "Displaced": {0: "No desplazado", 1: "Desplazado"},
    "Daytime/evening attendance": {0: "Nocturno", 1: "Diurno"},
    "International": {0: "Nacional", 1: "Internacional"},
    "Educational special needs": {0: "Sin necesidades especiales", 1: "Con necesidades especiales"},
}


def etiqueta(columna: str, codigo) -> str:
    """Texto legible del código UCI de una variable categórica, ordinal o binaria."""
    try:
        codigo = int(codigo)
    except (TypeError, ValueError):
        return str(codigo)
    if columna in ORDINALES:
        return ESCOLARIDAD.get(codigo, f"Código {codigo}")
    if columna in TEXTO_BINARIAS:
        return TEXTO_BINARIAS[columna].get(codigo, str(codigo))
    return ETIQUETAS.get(columna, {}).get(codigo, f"Código {codigo}")


# ---------------------------------------------------------------------------
# Datos e ingeniería de características
# ---------------------------------------------------------------------------
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Estandarización de nombres del notebook: quita espacios/tabuladores y abrevia los semestres."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.columns = (df.columns.str.replace("Curricular units 1st sem", "S1", regex=False)
                            .str.replace("Curricular units 2nd sem", "S2", regex=False))
    return df


def leer_csv(fuente) -> pd.DataFrame:
    """Lee un CSV con el formato de UCI (separado por ';') o separado por comas."""
    if hasattr(fuente, "seek"):  # archivo subido: Streamlit puede entregarlo ya leído en otra ejecución
        fuente.seek(0)
    df = pd.read_csv(fuente, sep=";", encoding="utf-8-sig")  # utf-8-sig: acepta CSV guardados por Excel
    if df.shape[1] == 1:
        if hasattr(fuente, "seek"):
            fuente.seek(0)
        df = pd.read_csv(fuente, sep=",", encoding="utf-8-sig")
    return preparar(df)


def cargar_datos() -> pd.DataFrame:
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No encontré {RUTA_DATOS}. Copia ahí el data.csv del dataset UCI #697.")
    return leer_csv(RUTA_DATOS)


def objetivo_binario(target: pd.Series) -> pd.Series:
    """'Dropout' -> 1 y 'Enrolled'/'Graduate' -> 0. También acepta una columna que ya venga en 0/1."""
    if pd.api.types.is_numeric_dtype(target):
        return target.astype(int)
    return (target.astype(str).str.strip() == "Dropout").astype(int)


def sin_unidades(df: pd.DataFrame) -> pd.Series:
    """Estudiantes que el notebook excluye: sin unidades inscritas en el 1.er o en el 2.º semestre."""
    s1 = df["S1 (enrolled)"] == 0
    s2 = df["S2 (enrolled)"] == 0 if "S2 (enrolled)" in df.columns else False
    return s1 | s2


def transformar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la ingeniería de características a datos con los códigos de UCI y devuelve las 28 columnas.
    Los códigos que el notebook no tiene en sus equivalencias quedan como vacío (NaN).
    """
    X = df[[c for c in COLUMNAS if c in df.columns]].copy()
    for c in CATEGORICAS:
        X[c] = pd.to_numeric(X[c], errors="coerce").map(ETIQUETAS[c])
    for c, escala in ESCALAS.items():
        X[c] = pd.to_numeric(X[c], errors="coerce").map(escala)
    return X[COLUMNAS]


def columnas_faltantes(df: pd.DataFrame) -> list[str]:
    return [c for c in COLUMNAS if c not in df.columns]


def datos_modelado(df: pd.DataFrame):
    """Pasos del notebook para armar X, y (objetivo binario, filtro, descarte y dropna)."""
    df = df.copy()
    df["Target"] = objetivo_binario(df["Target"])
    df = df.loc[~(df["S1 (enrolled)"] == 0) & ~(df["S2 (enrolled)"] == 0)]
    X = transformar(df)
    validas = X.notna().all(axis=1)
    X, y = X.loc[validas], df.loc[validas, "Target"]
    for c in CATEGORICAS:
        X[c] = X[c].astype("category").cat.remove_unused_categories()
    return X, y


def a_categorias(X: pd.DataFrame, categorias: dict, como_texto: bool = False) -> pd.DataFrame:
    """Usa exactamente las categorías del entrenamiento; una categoría desconocida queda como vacío."""
    X = X.copy()
    for c in CATEGORICAS:
        X[c] = pd.Categorical(X[c], categories=categorias[c])
        if como_texto:  # CatBoost recibe las categóricas como texto
            X[c] = X[c].astype(str)
    return X


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def cargar_modelo() -> dict:
    """Carga el artefacto entrenado (modelo + columnas + categorías + umbral) una sola vez."""
    if not RUTA_MODELO.exists():
        raise FileNotFoundError("No encontré el modelo. Ejecuta primero:  uv run entrenar")
    return joblib.load(RUTA_MODELO)


@lru_cache(maxsize=1)
def cargar_metricas() -> dict:
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


def nivel_riesgo(prob: float, umbral: float = UMBRAL) -> str:
    if prob >= umbral:
        return "Alto"
    if prob >= 0.3:
        return "Medio"
    return "Bajo"


def predecir(datos) -> pd.DataFrame:
    """
    Recibe un estudiante (dict) o varios (DataFrame) con los códigos de UCI y devuelve por fila:
    la probabilidad de abandono, la clasificación con el umbral del modelo, la confianza,
    un nivel de riesgo y si había códigos que el modelo no conoce.
    """
    df = pd.DataFrame([datos]) if isinstance(datos, dict) else preparar(datos)
    faltan = columnas_faltantes(df)
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(faltan))

    art = cargar_modelo()
    X = transformar(df)
    desconocidos = X.isna().any(axis=1)
    X = a_categorias(X, art["categorias"], art.get("requiere_texto", False))[art["columnas"]]
    prob = art["modelo"].predict_proba(X)[:, 1]
    umbral = art.get("umbral", UMBRAL)

    res = pd.DataFrame(index=df.index)
    res["prob_abandono"] = prob
    res["prediccion"] = (prob >= umbral).astype(int)
    res["clasificacion"] = res["prediccion"].map(CLASES)
    res["confianza"] = np.where(res["prediccion"] == 1, prob, 1 - prob)
    res["nivel_riesgo"] = [nivel_riesgo(p, umbral) for p in prob]
    res["codigos_desconocidos"] = desconocidos.values
    res["fuera_de_alcance"] = sin_unidades(df).values  # el modelo no se entrenó con estos casos
    return res


def senales_de_riesgo(d: dict) -> list[str]:
    """Señales que el análisis de los datos asoció con más abandono."""
    senales = []
    if int(d["Tuition fees up to date"]) == 0:
        senales.append("La matrícula **no está al día** (la variable categórica más asociada con el abandono).")
    if int(d["Debtor"]) == 1:
        senales.append("Tiene **deudas** con la institución.")
    if int(d["Scholarship holder"]) == 0:
        senales.append("**No tiene beca** (los becados abandonan menos).")
    if int(d["S1 (approved)"]) <= 2:
        senales.append("Aprobó **0-2 unidades en el 1.er semestre**: más del 70 % de ese grupo abandonó.")
    return senales


def formato_pct(x: float) -> str:
    """0.734 -> '73,4 %'"""
    return f"{x * 100:.1f} %".replace(".", ",")
