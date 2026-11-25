"""
Lógica compartida del sistema de ALERTA TEMPRANA DE DESERCIÓN UNIVERSITARIA.

El modelo vive aquí, separado de la interfaz: la app web (app.py) solo arma
los datos del estudiante, llama a predecir() y muestra el resultado.

La preparación de los datos replica las decisiones de la Entrega 1 (EDA):
  * Se limpian los nombres de columnas y se abrevian los semestres (S1, S2).
  * Se descartan 3 variables casi constantes: Educational special needs,
    International y Nacionality.
  * El objetivo conserva sus 3 clases (Dropout / Enrolled / Graduate).
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# Raíz del proyecto (donde está pyproject.toml): este archivo vive en
# src/desercion_universitaria/, así que subimos dos carpetas.
RAIZ = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
RUTA_DATOS = RAIZ / "data" / "data.csv"
CARPETA_MODELO = RAIZ / "modelo"
RUTA_MODELO = CARPETA_MODELO / "modelo_desercion.pkl"
RUTA_METRICAS = CARPETA_MODELO / "metricas_desercion.json"

SEMILLA = 42

# ---------------------------------------------------------------------------
# Variable objetivo
# ---------------------------------------------------------------------------
CLASES = ["Dropout", "Enrolled", "Graduate"]
NOMBRE_CLASE = {"Dropout": "Deserta", "Enrolled": "Sigue matriculado", "Graduate": "Se gradúa"}
COLORES = {"Dropout": "#d95f5f", "Enrolled": "#e0b84c", "Graduate": "#4c9be0"}  # paleta del notebook

# ---------------------------------------------------------------------------
# Variables (nombres tal como quedan después de preparar())
# ---------------------------------------------------------------------------
DESCARTADAS = ["Educational special needs", "International", "Nacionality"]

CATEGORICAS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
BINARIAS = [
    "Daytime/evening attendance", "Displaced", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder",
]
SEMESTRE = ["credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"]
NUMERICAS = (
    ["Application order", "Previous qualification (grade)", "Admission grade", "Age at enrollment"]
    + [f"S1 ({c})" for c in SEMESTRE]
    + [f"S2 ({c})" for c in SEMESTRE]
    + ["Unemployment rate", "Inflation rate", "GDP"]
)
MACRO = ["Unemployment rate", "Inflation rate", "GDP"]

# Orden de las 33 columnas que espera el modelo (36 originales - 3 descartadas)
COLUMNAS = CATEGORICAS + BINARIAS + NUMERICAS

# Nombres en español para mostrar en la interfaz
NOMBRES = {
    "Marital status": "Estado civil",
    "Application mode": "Modo de postulación",
    "Application order": "Orden de preferencia de la postulación (0 = primera opción)",
    "Course": "Programa",
    "Daytime/evening attendance": "Jornada",
    "Previous qualification": "Formación previa",
    "Previous qualification (grade)": "Nota de la formación previa (0-200)",
    "Mother's qualification": "Escolaridad de la madre",
    "Father's qualification": "Escolaridad del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Admission grade": "Nota de admisión (0-200)",
    "Displaced": "Vive desplazado de su residencia",
    "Debtor": "Tiene deudas con la institución",
    "Tuition fees up to date": "Matrícula al día",
    "Gender": "Género",
    "Scholarship holder": "Tiene beca",
    "Age at enrollment": "Edad al matricularse",
    "Unemployment rate": "Tasa de desempleo (%)",
    "Inflation rate": "Inflación (%)",
    "GDP": "PIB (variación %)",
    **{f"S{s} ({c})": f"{n} semestre · {t}" for s, n in [(1, "1.er"), (2, "2.º")]
       for c, t in zip(SEMESTRE, ["unidades convalidadas", "unidades inscritas",
                                  "evaluaciones presentadas", "unidades aprobadas",
                                  "nota promedio (0-20)", "unidades sin evaluación"])},
}

# Rangos de los controles del formulario: (mínimo, máximo, paso)
RANGOS = {
    "Application order": (0, 9, 1),
    "Previous qualification (grade)": (0.0, 200.0, 0.1),
    "Admission grade": (0.0, 200.0, 0.1),
    "Age at enrollment": (15, 80, 1),
    **{f"S{s} ({c})": r for s in (1, 2) for c, r in {
        "credited": (0, 30, 1), "enrolled": (0, 30, 1), "evaluations": (0, 50, 1),
        "approved": (0, 30, 1), "grade": (0.0, 20.0, 0.1), "without evaluations": (0, 15, 1),
    }.items()},
}

# ---------------------------------------------------------------------------
# Equivalencias código -> texto (diccionario de datos UCI #697)
# ---------------------------------------------------------------------------
ESCOLARIDAD = {
    1: "Secundaria completa (12.º año)", 2: "Educación superior: bachillerato universitario",
    3: "Educación superior: licenciatura", 4: "Educación superior: maestría",
    5: "Educación superior: doctorado", 6: "Cursó algo de educación superior",
    9: "12.º año sin terminar", 10: "11.º año sin terminar", 11: "7.º año (plan antiguo)",
    12: "Otro: 11.º año", 13: "2.º año de curso complementario de secundaria",
    14: "10.º año", 15: "10.º año sin terminar", 18: "Curso general de comercio",
    19: "Educación básica 3.er ciclo (9.º-11.º año)", 20: "Curso complementario de secundaria",
    22: "Curso técnico-profesional", 25: "Curso complementario de secundaria sin terminar",
    26: "7.º año", 27: "2.º ciclo de secundaria general", 29: "9.º año sin terminar",
    30: "8.º año", 31: "Curso general de administración y comercio",
    33: "Contabilidad y administración complementaria", 34: "Desconocido",
    35: "No sabe leer ni escribir", 36: "Sabe leer sin haber cursado 4.º año",
    37: "Educación básica 1.er ciclo (4.º-5.º año)", 38: "Educación básica 2.º ciclo (6.º-8.º año)",
    39: "Curso de especialización tecnológica", 40: "Educación superior: grado (1.er ciclo)",
    41: "Curso de estudios superiores especializados", 42: "Curso técnico superior profesional",
    43: "Educación superior: maestría (2.º ciclo)", 44: "Educación superior: doctorado (3.er ciclo)",
}

GRUPOS_OCUPACION = {
    0: "Estudiante",
    1: "Directivos, legisladores y gerentes",
    2: "Especialistas en actividades intelectuales y científicas",
    3: "Técnicos y profesionales de nivel medio",
    4: "Personal administrativo",
    5: "Servicios personales, seguridad y ventas",
    6: "Agricultores y trabajadores calificados del agro y la pesca",
    7: "Trabajadores calificados de la industria y la construcción",
    8: "Operadores de máquinas y ensambladores",
    9: "Trabajadores no calificados",
    10: "Fuerzas armadas",
    90: "Otra situación",
    99: "Sin información",
}

ETIQUETAS = {  # tomadas del notebook (celda de análisis bivariado)
    "Tuition fees up to date": {0: "No al día", 1: "Al día"},
    "Scholarship holder": {0: "Sin beca", 1: "Con beca"},
    "Debtor": {0: "Sin deuda", 1: "Con deuda"},
    "Gender": {0: "Femenino", 1: "Masculino"},
    "Displaced": {0: "No desplazado", 1: "Desplazado"},
    "Daytime/evening attendance": {0: "Nocturno", 1: "Diurno"},
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
    "Previous qualification": ESCOLARIDAD,
    "Mother's qualification": ESCOLARIDAD,
    "Father's qualification": ESCOLARIDAD,
}


def etiqueta(columna: str, codigo) -> str:
    """Texto legible para el código de una variable categórica o binaria."""
    try:
        codigo = int(codigo)
    except (TypeError, ValueError):
        return str(codigo)
    if columna in ("Mother's occupation", "Father's occupation"):
        if codigo in GRUPOS_OCUPACION:
            return GRUPOS_OCUPACION[codigo]
        if codigo >= 100:  # subgrupos de 3 dígitos: 1 + grupo + subgrupo (101-103 = fuerzas armadas)
            grupo = (codigo // 10) % 10 or 10
            return f"{GRUPOS_OCUPACION[grupo]} (subgrupo {codigo})"
        return f"Código {codigo}"
    return ETIQUETAS.get(columna, {}).get(codigo, f"Código {codigo}")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de nombres de la Entrega 1: quita espacios/tabuladores y abrevia los semestres."""
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


def columnas_faltantes(df: pd.DataFrame) -> list[str]:
    return [c for c in COLUMNAS if c not in df.columns]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el pipeline entrenado una sola vez (las siguientes llamadas usan la caché)."""
    if not RUTA_MODELO.exists():
        raise FileNotFoundError("No encontré el modelo. Ejecuta primero:  uv run entrenar")
    return joblib.load(RUTA_MODELO)


@lru_cache(maxsize=1)
def cargar_metricas() -> dict:
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


def nivel_riesgo(prob_desercion: float) -> str:
    if prob_desercion >= 0.6:
        return "Alto"
    if prob_desercion >= 0.3:
        return "Medio"
    return "Bajo"


def predecir(datos) -> pd.DataFrame:
    """
    Recibe un estudiante (dict) o varios (DataFrame) y devuelve, por fila:
    la probabilidad de cada clase, la clase predicha, la alerta Desertor / No desertor
    con su confianza y un nivel de riesgo según la probabilidad de deserción.
    """
    X = pd.DataFrame([datos]) if isinstance(datos, dict) else preparar(datos)
    faltan = columnas_faltantes(X)
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(faltan))

    modelo = cargar_modelo()
    proba = modelo.predict_proba(X[COLUMNAS])
    clases = list(modelo.classes_)
    res = pd.DataFrame(proba, columns=[f"prob_{c}" for c in clases], index=X.index)
    res["prediccion"] = [clases[i] for i in proba.argmax(axis=1)]
    res["resultado_estimado"] = res["prediccion"].map(NOMBRE_CLASE)
    res["alerta"] = res["prediccion"].map(lambda c: "Desertor" if c == "Dropout" else "No desertor")
    res["confianza"] = proba.max(axis=1)
    res["nivel_riesgo"] = res["prob_Dropout"].map(nivel_riesgo)
    return res


def senales_de_riesgo(d: dict) -> list[str]:
    """Señales que el análisis exploratorio (Entrega 1) asoció con más deserción."""
    senales = []
    if int(d["Tuition fees up to date"]) == 0:
        senales.append("La matrícula **no está al día** (la variable más asociada con el resultado, V de Cramér 0,43).")
    if int(d["Debtor"]) == 1:
        senales.append("Tiene **deudas** con la institución (V de Cramér 0,24).")
    if int(d["Scholarship holder"]) == 0:
        senales.append("**No tiene beca** (los becados desertan menos; V de Cramér 0,30).")
    if int(d["S1 (approved)"]) <= 2:
        senales.append("Aprobó **0-2 unidades en el 1.er semestre**: en el EDA, más del 70 % de ese grupo desertó.")
    if int(d["S2 (enrolled)"]) > 0 and int(d["S2 (approved)"]) == 0:
        senales.append("No aprobó ninguna unidad del **2.º semestre** teniendo unidades inscritas.")
    return senales


def formato_pct(x: float) -> str:
    """0.734 -> '73,4 %'"""
    return f"{x * 100:.1f} %".replace(".", ",")
