"""
App WEB (Streamlit) del sistema de alerta temprana de deserción universitaria.

Ejecutar en local:   uv run app
Publicar:            ver la sección "Poner la app en línea" del README.

Pestañas:
  1. Evaluar estudiante    -> formulario + probabilidad de abandono (salida de la propuesta).
  2. Evaluar grupo (CSV)   -> panel para muchos estudiantes a la vez, descargable.
  3. Análisis de los datos -> exploración del dataset con el objetivo binario de la Entrega 2.
  4. Sobre el modelo       -> balanceo, comparación de modelos, métricas en prueba e importancia.
"""
import sys
from pathlib import Path

# En Streamlit Community Cloud o Docker el paquete puede no estar instalado:
# agregamos src/ al path para que el import funcione igual que en local.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from desercion_universitaria.core import (  # noqa: E402
    BINARIAS, CATEGORICAS, CLASES, COLORES, COLUMNAS, DESCARTADAS, NOMBRES, ORDINALES, RANGOS, RUTA_DATOS,
    cargar_datos, cargar_metricas, cargar_modelo, columnas_faltantes, etiqueta, formato_pct, leer_csv,
    objetivo_binario, predecir, senales_de_riesgo)

st.set_page_config(page_title="Alerta temprana de deserción universitaria", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.tarjeta {color: white; padding: 1.6rem; border-radius: 1rem; text-align: center;}
.tarjeta.alto  {background: linear-gradient(135deg, #b91c1c, #d95f5f);}
.tarjeta.medio {background: linear-gradient(135deg, #b45309, #e0b84c);}
.tarjeta.bajo  {background: linear-gradient(135deg, #1d4ed8, #4c9be0);}
.tarjeta .titulo {font-size: .8rem; letter-spacing: .12em; opacity: .85;}
.tarjeta .valor {font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: .3rem 0;}
.tarjeta .sub {opacity: .9;}
</style>
""", unsafe_allow_html=True)

ORDEN_CLASES = ["No abandono", "Abandono"]
ESCALA_CLASES = alt.Scale(domain=ORDEN_CLASES, range=[COLORES[c] for c in ORDEN_CLASES])
COLOR_RIESGO = {"Alto": "#d95f5f", "Medio": "#e0b84c", "Bajo": "#4c9be0"}
NOMBRE_METRICA = {"auc": "AUC", "ap": "AP", "recall": "Recall", "precision": "Precisión", "f1": "F1",
                  "bal_acc": "Balanced acc."}
INTEGRANTES = ["David Santiago Roa Mayor", "Maira Alejandra Balanta Peña", "Jimmy Lopez Yule",
               "Marco Antonio Aragon Vivas", "Esteban Cobo Gómez"]


# ---------------------------------------------------------------------------
# Carga (con caché: se hace una vez por sesión del servidor)
# ---------------------------------------------------------------------------
@st.cache_resource
def modelo_y_metricas():
    return cargar_modelo(), cargar_metricas()


@st.cache_data
def datos():
    return cargar_datos()


try:
    ART, M = modelo_y_metricas()
except FileNotFoundError as e:
    st.error(f"{e}\n\nPasos: copia `data.csv` en la carpeta `data/` y ejecuta `uv run entrenar`.")
    st.stop()

F = M["formulario"]
UMBRAL = M["umbral"]
PRUEBA = next(f for f in M["prueba"] if f["modelo"] == M["modelo"])
hay_datos = RUTA_DATOS.exists()

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎓 Proyecto")
    st.write("Sistema de análisis, monitoreo y alerta temprana de la deserción universitaria "
             "con aprendizaje automático.")
    st.markdown("**Integrantes**\n\n" + "\n".join(f"- {n}" for n in INTEGRANTES))
    st.divider()
    st.markdown(f"**Modelo:** {M['modelo']} (HistGradientBoosting, `class_weight`)  \n"
                f"**AP en prueba:** {PRUEBA['ap']:.3f} · **AUC:** {PRUEBA['auc']:.3f}  \n"
                f"**Recall de abandono:** {PRUEBA['recall']:.0%}  \n"
                f"**Umbral de decisión:** {UMBRAL}  \n"
                f"**Entrenado:** {M['fecha']}")
    st.caption("Datos: [UCI #697 Predict Students' Dropout and Academic Success]"
               "(https://archive.ics.uci.edu/dataset/697) · CC BY 4.0")

st.title("🎓 Alerta temprana de deserción universitaria")
st.caption("Clasifica a cada estudiante como **abandono** o **no abandono**, con su nivel de confianza, "
           "para ofrecer apoyo a tiempo. Herramienta de apoyo a la decisión: no reemplaza el criterio "
           "de bienestar universitario.")

tab_est, tab_grupo, tab_eda, tab_modelo = st.tabs(
    ["🧑‍🎓 Evaluar estudiante", "📂 Evaluar grupo (CSV)", "📊 Análisis de los datos", "🤖 Sobre el modelo"])


# ---------------------------------------------------------------------------
# Utilidades de interfaz
# ---------------------------------------------------------------------------
def selector(col, contenedor):
    opciones = F["opciones"][col]
    tipico = F["tipicos"][col]
    return contenedor.selectbox(NOMBRES[col], opciones,
                                index=opciones.index(tipico) if tipico in opciones else 0,
                                format_func=lambda c, col=col: etiqueta(col, c))


def numero(col, contenedor, etiqueta_corta=None):
    minimo, maximo, paso = RANGOS[col]
    valor = min(max(type(minimo)(F["tipicos"][col]), minimo), maximo)
    return contenedor.number_input(etiqueta_corta or NOMBRES[col], min_value=minimo, max_value=maximo,
                                   value=valor, step=paso, key=col)


def binaria(col, contenedor):
    return int(contenedor.toggle(NOMBRES[col], value=bool(F["tipicos"][col])))


def radio_binaria(col, contenedor, orden):
    return contenedor.radio(NOMBRES[col], orden, horizontal=True,
                            index=orden.index(F["tipicos"][col]) if F["tipicos"][col] in orden else 0,
                            format_func=lambda v: etiqueta(col, v))


def grafico_probabilidad(prob):
    d = pd.DataFrame({"Resultado": ORDEN_CLASES, "Probabilidad": [1 - float(prob), float(prob)]})
    base = alt.Chart(d).encode(
        y=alt.Y("Resultado:N", sort=ORDEN_CLASES, title=None),
        x=alt.X("Probabilidad:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None))
    barras = base.mark_bar().encode(color=alt.Color("Resultado:N", scale=ESCALA_CLASES, legend=None),
                                    tooltip=["Resultado", alt.Tooltip("Probabilidad:Q", format=".1%")])
    texto = base.mark_text(align="left", dx=4).encode(text=alt.Text("Probabilidad:Q", format=".0%"))
    regla = alt.Chart(pd.DataFrame({"x": [float(UMBRAL)]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
    return (barras + texto + regla).properties(height=110)


def matriz_confusion(matriz):
    filas = [{"Real": ORDEN_CLASES[i], "Predicho": ORDEN_CLASES[j], "n": int(matriz[i][j])}
             for i in range(2) for j in range(2)]
    d = pd.DataFrame(filas)
    base = alt.Chart(d).encode(x=alt.X("Predicho:N", sort=ORDEN_CLASES), y=alt.Y("Real:N", sort=ORDEN_CLASES))
    celdas = base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                     tooltip=["Real", "Predicho", "n"])
    texto = base.mark_text(fontSize=16).encode(
        text="n:Q", color=alt.condition(alt.datum.n > float(d["n"].max()) / 2, alt.value("white"),
                                        alt.value("black")))
    return (celdas + texto).properties(height=230)


def miles(n) -> str:
    """4065 -> '4.065'"""
    return f"{int(n):,}".replace(",", ".")


def texto_ic(m):
    return f"{m['media']:.4f} [{m['ic_inf']:.4f}, {m['ic_sup']:.4f}]"


# ===========================================================================
# 1. Evaluar estudiante
# ===========================================================================
with tab_est:
    col_form, col_res = st.columns([3, 2], gap="large")

    with col_form, st.form("estudiante", border=True):
        st.markdown("##### 📝 Matrícula y datos personales")
        c1, c2 = st.columns(2)
        d = {"Course": selector("Course", c1), "Application mode": selector("Application mode", c2)}
        c1, c2, c3 = st.columns(3)
        d["Application order"] = numero("Application order", c1, "Orden de la postulación (0 = 1.ª opción)")
        d["Daytime/evening attendance"] = radio_binaria("Daytime/evening attendance", c2, [1, 0])
        d["Gender"] = radio_binaria("Gender", c3, [0, 1])
        c1, c2, c3 = st.columns(3)
        d["Age at enrollment"] = numero("Age at enrollment", c1)
        d["Marital status"] = selector("Marital status", c2)
        d["Previous qualification"] = selector("Previous qualification", c3)
        c1, c2 = st.columns(2)
        d["Previous qualification (grade)"] = numero("Previous qualification (grade)", c1)
        d["Admission grade"] = numero("Admission grade", c2)

        st.markdown("##### 💰 Situación económica y personal")
        c1, c2, c3 = st.columns(3)
        for col, cont in [("Tuition fees up to date", c1), ("Debtor", c2), ("Scholarship holder", c3),
                          ("Displaced", c1), ("International", c2), ("Educational special needs", c3)]:
            d[col] = binaria(col, cont)

        st.markdown("##### 📘 Rendimiento del 1.er semestre")
        cols = st.columns(6)
        for c, campo in zip(cols, ["enrolled", "evaluations", "approved", "grade", "credited", "without evaluations"]):
            col = f"S1 ({campo})"
            d[col] = numero(col, c, NOMBRES[col].split("· ")[1].capitalize())

        st.markdown("##### 📗 2.º semestre")
        c1, c2, _ = st.columns(3)
        for col, cont in [("S2 (credited)", c1), ("S2 (without evaluations)", c2)]:
            d[col] = numero(col, cont, NOMBRES[col].split("· ")[1].capitalize())

        with st.expander("👪 Familia (opcional: viene con los valores más comunes)"):
            c1, c2 = st.columns(2)
            for col, cont in [("Mother's qualification", c1), ("Father's qualification", c2),
                              ("Mother's occupation", c1), ("Father's occupation", c2)]:
                d[col] = selector(col, cont)

        enviado = st.form_submit_button("Evaluar riesgo de abandono", type="primary", width="stretch")

    if enviado:
        st.session_state["evaluacion"] = (d, predecir(d).iloc[0])

    with col_res:
        if "evaluacion" not in st.session_state:
            st.info("Completa el formulario y presiona **Evaluar riesgo de abandono**. "
                    "Los campos vienen con los valores más comunes del dataset.")
        else:
            d_eval, r = st.session_state["evaluacion"]
            nivel = r["nivel_riesgo"]
            st.markdown(f"""
            <div class="tarjeta {nivel.lower()}">
              <div class="titulo">PROBABILIDAD DE ABANDONO</div>
              <div class="valor">{formato_pct(r['prob_abandono'])}</div>
              <div class="sub">Clasificación: <b>{r['clasificacion']}</b> · confianza {formato_pct(r['confianza'])}</div>
              <div class="sub">Riesgo <b>{nivel.lower()}</b> · umbral de decisión {formato_pct(UMBRAL)}</div>
            </div>""", unsafe_allow_html=True)
            st.write("")
            st.altair_chart(grafico_probabilidad(r["prob_abandono"]))
            st.caption("Línea punteada: umbral de decisión del modelo.")

            avisos = []
            if r["fuera_de_alcance"]:
                avisos.append("El estudiante no tiene unidades inscritas en el 1.er semestre: el modelo se "
                              "entrenó sin esos casos, así que el resultado es menos confiable.")
            if d_eval["S1 (approved)"] > d_eval["S1 (enrolled)"]:
                avisos.append("Hay más unidades aprobadas que inscritas en el 1.er semestre.")
            if d_eval["S1 (approved)"] > d_eval["S1 (evaluations)"]:
                avisos.append("Hay más unidades aprobadas que evaluaciones en el 1.er semestre.")
            if avisos:
                st.warning("Revisa los datos: " + " ".join(avisos))

            senales = senales_de_riesgo(d_eval)
            st.markdown("**Señales de riesgo encontradas en los datos**")
            st.markdown("\n".join(f"- {s}" for s in senales) if senales
                        else "- Ninguna de las señales principales está presente.")
            accion = {"Alto": "Prioridad alta: remitir a acompañamiento académico y financiero.",
                      "Medio": "Hacer seguimiento durante el semestre.",
                      "Bajo": "Sin alerta por ahora."}[nivel]
            st.markdown(f"**Sugerencia:** {accion}")


# ===========================================================================
# 2. Evaluar grupo (CSV)
# ===========================================================================
with tab_grupo:
    st.write("Sube un CSV con las columnas del dataset UCI (separado por `;` o por `,`). "
             "Si trae la columna `Target`, también se compara la predicción con el resultado real.")
    c1, c2 = st.columns([3, 2])
    archivo = c1.file_uploader("Archivo CSV de estudiantes", type=["csv"])
    with c2:
        st.write("")
        if hay_datos:
            plantilla = datos().drop(columns="Target").head(5)
            st.download_button("⬇️ Descargar plantilla (5 estudiantes)",
                               plantilla.to_csv(sep=";", index=False).encode("utf-8-sig"),
                               file_name="plantilla_estudiantes.csv", mime="text/csv")
            demo = st.toggle("Usar los estudiantes del conjunto de prueba (ejemplo)",
                             help="Los 813 estudiantes que el modelo no vio al entrenar, con su resultado real.")
        else:
            demo = False

    grupo = None
    if archivo is not None:
        try:
            grupo = leer_csv(archivo)
        except Exception as e:  # CSV mal formado
            st.error(f"No pude leer el archivo: {e}")
    elif demo:
        df_todo = datos()
        grupo = df_todo.loc[[i for i in M["indices_prueba"] if i in df_todo.index]]

    if grupo is not None:
        faltan = columnas_faltantes(grupo)
        if faltan:
            st.error("Al archivo le faltan estas columnas: " + ", ".join(f"`{c}`" for c in faltan))
        else:
            res = predecir(grupo)
            resultado = pd.concat([grupo.reset_index(drop=True), res.reset_index(drop=True)], axis=1)

            n = len(resultado)
            conteo = resultado["nivel_riesgo"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Estudiantes", miles(n))
            c2.metric("Riesgo alto (abandono)", int(conteo.get("Alto", 0)), border=True)
            c3.metric("Riesgo medio", int(conteo.get("Medio", 0)), border=True)
            c4.metric("Clasificados como abandono", f"{resultado['prediccion'].mean():.0%}")

            n_desc, n_fuera = int(resultado["codigos_desconocidos"].sum()), int(resultado["fuera_de_alcance"].sum())
            if n_desc or n_fuera:
                st.warning(
                    (f"{n_desc} estudiante(s) tienen códigos que el modelo no conoce (se tratan como vacíos). "
                     if n_desc else "")
                    + (f"{n_fuera} estudiante(s) no tienen unidades inscritas en algún semestre: el modelo "
                       "se entrenó sin esos casos." if n_fuera else ""))

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Distribución de la probabilidad de abandono**")
                hist = alt.Chart(resultado[["prob_abandono", "nivel_riesgo"]]).mark_bar().encode(
                    x=alt.X("prob_abandono:Q", bin=alt.Bin(step=0.05), axis=alt.Axis(format="%"),
                            title="Probabilidad de abandono"),
                    y=alt.Y("count():Q", title="Estudiantes"),
                    color=alt.Color("nivel_riesgo:N", title="Riesgo",
                                    scale=alt.Scale(domain=list(COLOR_RIESGO), range=list(COLOR_RIESGO.values()))))
                st.altair_chart(hist.properties(height=280))
            with g2:
                st.markdown("**Clasificación**")
                est = resultado["clasificacion"].value_counts().rename_axis("Resultado").reset_index(name="n")
                st.altair_chart(alt.Chart(est).mark_bar().encode(
                    x=alt.X("n:Q", title="Estudiantes"), y=alt.Y("Resultado:N", sort=ORDEN_CLASES, title=None),
                    color=alt.Color("Resultado:N", scale=ESCALA_CLASES, legend=None),
                    tooltip=["Resultado", "n"]).properties(height=280))

            if "Target" in resultado.columns:
                from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                             precision_score, recall_score)
                real, pred = objetivo_binario(resultado["Target"]), resultado["prediccion"]
                st.markdown("**Comparación con el resultado real**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recall de abandono", f"{recall_score(real, pred, zero_division=0):.1%}"
                          if real.any() else "Sin abandonos reales")
                c2.metric("Precisión", f"{precision_score(real, pred, zero_division=0):.1%}")
                c3.metric("F1", f"{f1_score(real, pred, zero_division=0):.3f}")
                c4.metric("AP", f"{average_precision_score(real, resultado['prob_abandono']):.3f}"
                          if real.any() and not real.all() else "—")
                st.altair_chart(matriz_confusion(confusion_matrix(real, pred, labels=[0, 1]).tolist()))

            st.markdown("**Estudiantes ordenados por riesgo**")
            vista = resultado.sort_values("prob_abandono", ascending=False)
            columnas_vista = (["prob_abandono", "nivel_riesgo", "clasificacion"]
                              + (["Target"] if "Target" in vista.columns else [])
                              + ["Course", "Tuition fees up to date", "Debtor", "Scholarship holder",
                                 "S1 (approved)", "S1 (grade)"])
            tabla = vista[columnas_vista].copy()
            tabla["Course"] = tabla["Course"].map(lambda c: etiqueta("Course", c))
            config = {
                "prob_abandono": st.column_config.ProgressColumn("Prob. abandono", format="percent",
                                                                 min_value=0.0, max_value=1.0),
                "nivel_riesgo": "Riesgo", "clasificacion": "Clasificación", "Target": "Resultado real",
                "Course": "Programa", "Tuition fees up to date": "Matrícula al día", "Debtor": "Deudor",
                "Scholarship holder": "Becado", "S1 (approved)": "Aprobadas S1", "S1 (grade)": "Nota S1"}
            st.dataframe(tabla, hide_index=True, width="stretch", height=380,
                         column_config={k: v for k, v in config.items() if k in tabla.columns})
            st.download_button("⬇️ Descargar resultados (CSV)",
                               vista.to_csv(sep=";", index=False).encode("utf-8-sig"),
                               file_name="resultados_desercion.csv", mime="text/csv", type="primary")


# ===========================================================================
# 3. Análisis de los datos
# ===========================================================================
@st.cache_data
def eda_basico():
    df = datos().copy()
    df["abandono"] = objetivo_binario(df["Target"])
    return df


@st.cache_data
def v_cramer_todas():
    df = eda_basico()

    def v_cramer(x, y):
        tabla = pd.crosstab(x, y)
        chi2 = chi2_contingency(tabla, correction=False)[0]
        r, k = tabla.shape
        return np.sqrt((chi2 / tabla.values.sum()) / (min(r, k) - 1))

    return (pd.Series({NOMBRES[c]: v_cramer(df[c], df["abandono"]) for c in BINARIAS + CATEGORICAS + ORDINALES})
            .sort_values(ascending=False).rename_axis("Variable").reset_index(name="V"))


with tab_eda:
    if not hay_datos:
        st.info("Para ver el análisis de los datos, incluye `data/data.csv` en el proyecto.")
    else:
        df = eda_basico()
        tasa_global = float(df["abandono"].mean())
        filas = M["filas"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estudiantes en el dataset", miles(filas["total"]))
        c2.metric("Usados para modelar", miles(filas["modelado"]))
        c3.metric("Variables del modelo", len(COLUMNAS))
        c4.metric("Tasa de abandono", f"{tasa_global:.1%}")

        st.subheader("Variable objetivo: abandono frente a no abandono")
        g1, g2 = st.columns([2, 3])
        conteo = df["Target"].value_counts().rename_axis("Original").reset_index(name="Estudiantes")
        conteo["Resultado"] = np.where(conteo["Original"] == "Dropout", "Abandono", "No abandono")
        conteo["Original"] = conteo["Original"].map({"Dropout": "Deserta", "Enrolled": "Sigue matriculado",
                                                     "Graduate": "Se gradúa"})
        g1.altair_chart(alt.Chart(conteo).mark_bar().encode(
            x=alt.X("Resultado:N", sort=ORDEN_CLASES, title=None), y=alt.Y("Estudiantes:Q"),
            color=alt.Color("Original:N", title="Clase original",
                            scale=alt.Scale(domain=["Se gradúa", "Sigue matriculado", "Deserta"],
                                            range=["#4c9be0", "#93c5fd", "#d95f5f"])),
            tooltip=["Original", "Estudiantes"]).properties(height=300))
        ce = M["clases_entrenamiento"]
        razon = f"{ce['0'] / ce['1']:.2f}".replace(".", ",")
        g2.markdown(f"""
- La variable original tiene 3 clases. La Entrega 2 la convierte en **binaria**: *Dropout* = abandono (1);
  *Enrolled* y *Graduate* = no abandono (0).
- Desbalance en entrenamiento: {miles(ce['0'])} no abandono frente a {miles(ce['1'])} abandono
  (razón {razon} a 1). Se corrige con `class_weight="balanced"`.
- Por el desbalance, el modelo se elige por **Average Precision (AP)**: resume la curva
  precisión-recall de la clase minoritaria (abandono).
""")

        st.subheader("Tasa de abandono por categoría")
        opciones_cat = ["Tuition fees up to date", "Scholarship holder", "Debtor", "Gender", "Displaced",
                        "Daytime/evening attendance", "Marital status", "Application mode", "Course"]
        var = st.selectbox("Variable", opciones_cat, format_func=lambda c: NOMBRES[c], key="eda_cat")
        tasas = df.groupby(var)["abandono"].agg(["mean", "size"]).reset_index()
        tasas["Categoría"] = tasas[var].map(lambda k: etiqueta(var, k))
        tasas = tasas.rename(columns={"mean": "Tasa", "size": "Estudiantes"})[["Categoría", "Tasa", "Estudiantes"]]
        barras = alt.Chart(tasas).mark_bar().encode(
            x=alt.X("Tasa:Q", axis=alt.Axis(format="%"), title="Tasa de abandono"),
            y=alt.Y("Categoría:N", sort="-x", title=None),
            color=alt.condition(alt.datum.Tasa > tasa_global, alt.value(COLORES["Abandono"]), alt.value("#9ca3af")),
            tooltip=["Categoría", alt.Tooltip("Tasa:Q", format=".1%"), "Estudiantes"])
        regla = alt.Chart(pd.DataFrame({"x": [tasa_global]})).mark_rule(color="red", strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((barras + regla).properties(height=max(160, 26 * len(tasas))))
        st.caption(f"Línea roja: tasa global de abandono ({tasa_global:.1%}). En rojo, las categorías por encima.")

        st.subheader("Distribución de las variables numéricas")
        opciones_num = ["S1 (approved)", "S1 (grade)", "S2 (approved)", "S2 (grade)",
                        "Age at enrollment", "Admission grade", "Previous qualification (grade)"]
        var_num = st.selectbox("Variable", opciones_num, format_func=lambda c: NOMBRES[c], key="eda_num")
        dist = pd.DataFrame({"valor": df[var_num].astype(float), "Resultado": df["abandono"].map(CLASES)})
        st.altair_chart(alt.Chart(dist).transform_density(
            "valor", groupby=["Resultado"], as_=["valor", "densidad"]).mark_area(opacity=0.45).encode(
            x=alt.X("valor:Q", title=NOMBRES[var_num]), y=alt.Y("densidad:Q", title="Densidad"),
            color=alt.Color("Resultado:N", scale=ESCALA_CLASES)).properties(height=300))
        if var_num.startswith("S2 (") and var_num != "S2 (credited)":
            st.caption("Esta variable del 2.º semestre se descartó del modelo por su alta correlación con la del 1.er semestre.")

        st.subheader("El rendimiento del 1.er semestre es la señal más fuerte")
        rangos = pd.cut(df["S1 (approved)"], bins=[-1, 0, 2, 4, 6, 30], labels=["0", "1-2", "3-4", "5-6", "7+"])
        curva = df.assign(rango=rangos).groupby("rango", observed=True)["abandono"].mean().reset_index()
        st.altair_chart(alt.Chart(curva).mark_line(point=True, strokeWidth=3, color=COLORES["Abandono"]).encode(
            x=alt.X("rango:N", sort=["0", "1-2", "3-4", "5-6", "7+"], title="Unidades aprobadas en el 1.er semestre"),
            y=alt.Y("abandono:Q", axis=alt.Axis(format="%"), title="Tasa de abandono"),
            tooltip=["rango", alt.Tooltip("abandono:Q", format=".0%")]).properties(height=280))

        st.subheader("Asociación de cada variable con el abandono")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**V de Cramér** (categóricas, ordinales y binarias)")
            vc = v_cramer_todas()
            st.altair_chart(alt.Chart(vc).mark_bar(color=COLORES["No abandono"]).encode(
                x=alt.X("V:Q", title="V de Cramér"), y=alt.Y("Variable:N", sort="-x", title=None),
                tooltip=["Variable", alt.Tooltip("V:Q", format=".3f")]).properties(height=400))
        with g2:
            st.markdown("**Correlación entre variables numéricas**")
            cols_corr = ["Admission grade", "Age at enrollment", "S1 (enrolled)", "S1 (approved)", "S1 (grade)",
                         "S2 (enrolled)", "S2 (approved)", "S2 (grade)", "abandono"]
            corr = df[cols_corr].corr().rename(index={"abandono": "Abandono"}, columns={"abandono": "Abandono"})
            largo = corr.stack().rename_axis(["fila", "columna"]).reset_index(name="r")
            orden = list(corr.columns)
            base = alt.Chart(largo).encode(x=alt.X("columna:N", sort=orden, title=None),
                                           y=alt.Y("fila:N", sort=orden, title=None))
            st.altair_chart((base.mark_rect().encode(
                color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), legend=None),
                tooltip=["fila", "columna", alt.Tooltip("r:Q", format=".2f")])
                + base.mark_text(fontSize=10).encode(text=alt.Text("r:Q", format=".2f")))
                .properties(height=400))
            st.caption("Las variables del 1.er y 2.º semestre están muy correlacionadas: por eso se descartan "
                       "las del 2.º semestre (inscritas, aprobadas, nota y evaluaciones).")

        st.subheader("Ingeniería de características (Entrega 2)")
        st.markdown(f"""
- **Objetivo binario:** abandono (1) frente a no abandono (0).
- **Filtro:** se quitan los estudiantes sin unidades inscritas en el 1.er o en el 2.º semestre, y los que tienen
  códigos fuera de las equivalencias: quedan {miles(filas['modelado'])} de {miles(filas['total'])}.
- **Variables descartadas** (irrelevantes o correlacionadas): {", ".join(f"`{c}`" for c in DESCARTADAS)}.
- **Nominales → texto** (tipo *category*): estado civil, modo de postulación, programa y ocupación de la madre y del padre.
- **Escolaridad → escala ordinal:** formación previa (0-13) y escolaridad de la madre y del padre (0-23).
- **Resultado:** {len(COLUMNAS)} variables. HistGradientBoosting usa las categóricas de forma nativa.
- **Partición:** estratificada 80/20 ({miles(filas['entrenamiento'])} / {miles(filas['prueba'])}); validación cruzada
  estratificada de 5 pliegues.
""")


# ===========================================================================
# 4. Sobre el modelo
# ===========================================================================
with tab_modelo:
    st.subheader(f"{M['modelo']} · umbral de decisión {UMBRAL}")
    st.write(f"Entrenado con {miles(M['filas']['entrenamiento'])} estudiantes y evaluado con "
             f"{miles(M['filas']['prueba'])} que no vio durante el entrenamiento.")
    cols = st.columns(6)
    ayudas = {"recall": "De los estudiantes que realmente abandonaron, qué porcentaje detecta el modelo.",
              "precision": "De los que el modelo marca como abandono, qué porcentaje realmente abandonó.",
              "ap": "Average Precision: resume la curva precisión-recall de la clase abandono."}
    for c, k in zip(cols, ["ap", "auc", "recall", "precision", "f1", "bal_acc"]):
        c.metric(NOMBRE_METRICA[k], f"{PRUEBA[k]:.3f}", border=True, help=ayudas.get(k))

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Matriz de confusión (prueba)**")
        st.altair_chart(matriz_confusion(M["matriz_confusion"]))
    with g2:
        st.markdown(f"**Curva ROC** (AUC {PRUEBA['auc']:.3f})")
        roc = pd.DataFrame({"FPR": M["curva_roc"]["x"], "TPR": M["curva_roc"]["y"]})
        azar = alt.Chart(pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="a:Q", y="b:Q")
        st.altair_chart((alt.Chart(roc).mark_line(color=COLORES["Abandono"], strokeWidth=3).encode(
            x=alt.X("FPR:Q", title="Tasa de falsos positivos"), y=alt.Y("TPR:Q", title="Recall"))
            + azar).properties(height=230))
    with g3:
        st.markdown(f"**Curva precisión-recall** (AP {PRUEBA['ap']:.3f})")
        pr = pd.DataFrame({"Recall": M["curva_pr"]["x"], "Precision": M["curva_pr"]["y"]})
        base_pr = M["matriz_confusion"][1][0] + M["matriz_confusion"][1][1]
        azar = alt.Chart(pd.DataFrame({"y": [base_pr / M["filas"]["prueba"]]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((alt.Chart(pr).mark_line(color=COLORES["No abandono"], strokeWidth=3).encode(
            x=alt.X("Recall:Q"), y=alt.Y("Precision:Q", title="Precisión", scale=alt.Scale(domain=[0, 1])))
            + azar).properties(height=230))

    st.markdown("**Comparación de modelos** (validación cruzada estratificada de 5 pliegues: media [IC95 %])")
    comp = pd.DataFrame([{"Modelo": f["modelo"], **{NOMBRE_METRICA[k]: texto_ic(f[k]) for k in NOMBRE_METRICA}}
                         for f in M["comparacion"]])
    st.dataframe(comp, hide_index=True, width="stretch")
    if len(M["comparacion"]) < 4:
        st.caption("En la Entrega 2 también se compararon LightGBM y CatBoost (AP en CV 0,8414 y 0,8425), sin "
                   "diferencias significativas con HistGB. Para incluirlos aquí: `uv add lightgbm catboost` y "
                   "`uv run entrenar`.")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Métricas en prueba** (umbral 0,5)")
        st.dataframe(pd.DataFrame(M["prueba"]).rename(columns={"modelo": "Modelo", **NOMBRE_METRICA}),
                     hide_index=True, width="stretch")
        st.markdown("**Hiperparámetros**")
        for nombre, b in M["busqueda"].items():
            st.markdown(f"- **{nombre}** ({b['origen']}, AP en CV {b['ap_cv']:.4f}): "
                        + ", ".join(f"`{k}={v}`" for k, v in b["parametros"].items()))
    with g2:
        st.markdown("**Balanceo de clases** (regresión logística, validación cruzada)")
        st.dataframe(pd.DataFrame(M["balanceo"]).rename(columns={"metodo": "Método", **NOMBRE_METRICA}),
                     hide_index=True, width="stretch")
        if len(M["balanceo"]) < 9:
            st.caption("En la Entrega 2 se compararon 9 métodos; `class_weight` mejora el recall y el F1 frente a "
                       "no balancear, y el sobremuestreo no lo supera. Para verlos aquí, con `imbalanced-learn` instalado, "
                       "ejecuta `uv run entrenar`.")

    st.markdown("**Variables más importantes** (caída del AP al desordenar cada variable, en prueba)")
    imp = pd.DataFrame(M["importancias"]).head(15)
    imp["Variable"] = imp["variable"].map(lambda c: NOMBRES.get(c, c))
    imp["inf"], imp["sup"] = imp["media"] - imp["desv"], imp["media"] + imp["desv"]
    base = alt.Chart(imp).encode(y=alt.Y("Variable:N", sort="-x", title=None))
    st.altair_chart((base.mark_bar(color=COLORES["Abandono"]).encode(
        x=alt.X("media:Q", title="Caída del AP"), tooltip=["Variable", alt.Tooltip("media:Q", format=".4f")])
        + base.mark_rule().encode(x="inf:Q", x2="sup:Q")).properties(height=420))
    st.caption("Datos de una institución de educación superior de Portugal: antes de usarlo con estudiantes de otra "
               "institución hay que reentrenarlo con sus propios datos.")
