"""
App WEB (Streamlit) del sistema de alerta temprana de deserción universitaria.

Ejecutar en local:   uv run app
Publicar:            ver la sección "Poner la app en línea" del README.

Pestañas:
  1. Evaluar estudiante  -> formulario + probabilidad de deserción (salida de la propuesta).
  2. Evaluar grupo (CSV) -> panel para muchos estudiantes a la vez, descargable.
  3. Análisis exploratorio -> el EDA de la Entrega 1, ahora interactivo.
  4. Sobre el modelo      -> comparación de modelos, métricas en prueba e importancia de variables.
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
    BINARIAS, CATEGORICAS, CLASES, COLORES, COLUMNAS, MACRO, NOMBRE_CLASE, NOMBRES, RANGOS,
    RUTA_DATOS, cargar_datos, cargar_metricas, cargar_modelo, columnas_faltantes, etiqueta,
    formato_pct, leer_csv, predecir, senales_de_riesgo)

st.set_page_config(page_title="Alerta temprana de deserción", page_icon="🎓", layout="wide")

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

ESCALA_CLASES = alt.Scale(domain=[NOMBRE_CLASE[c] for c in CLASES], range=[COLORES[c] for c in CLASES])
COLOR_RIESGO = {"Alto": COLORES["Dropout"], "Medio": COLORES["Enrolled"], "Bajo": COLORES["Graduate"]}
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
    modelo, M = modelo_y_metricas()
except FileNotFoundError as e:
    st.error(f"{e}\n\nPasos: copia `data.csv` en la carpeta `data/` y ejecuta `uv run entrenar`.")
    st.stop()

F = M["formulario"]
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
    st.markdown(f"**Modelo:** {M['modelo']} ({M['balanceo']})  \n"
                f"**F1 macro (prueba):** {M['prueba']['f1_macro']:.2f}  \n"
                f"**Recall de deserción:** {M['prueba']['recall']['Dropout']:.0%}  \n"
                f"**Entrenado:** {M['fecha']}")
    st.caption("Datos: [UCI #697 Predict Students' Dropout and Academic Success]"
               "(https://archive.ics.uci.edu/dataset/697) · CC BY 4.0")

st.title("🎓 Alerta temprana de deserción universitaria")
st.caption("Clasifica a cada estudiante como desertor o no desertor, con su nivel de confianza, "
           "para ofrecer apoyo a tiempo. Herramienta de apoyo a la decisión: no reemplaza el criterio "
           "de bienestar universitario.")

tab_est, tab_grupo, tab_eda, tab_modelo = st.tabs(
    ["🧑‍🎓 Evaluar estudiante", "📂 Evaluar grupo (CSV)", "📊 Análisis exploratorio", "🤖 Sobre el modelo"])


# ---------------------------------------------------------------------------
# Utilidades de interfaz
# ---------------------------------------------------------------------------
def selector(col, contenedor):
    opciones = F["opciones"][col]
    tipico = F["tipicos"][col]
    return contenedor.selectbox(NOMBRES[col], opciones,
                                index=opciones.index(tipico) if tipico in opciones else 0,
                                format_func=lambda c, col=col: etiqueta(col, c))


def numero(col, contenedor):
    minimo, maximo, paso = RANGOS[col]
    valor = min(max(type(minimo)(F["tipicos"][col]), minimo), maximo)
    return contenedor.number_input(NOMBRES[col], min_value=minimo, max_value=maximo, value=valor, step=paso)


def binaria(col, contenedor):
    return int(contenedor.toggle(NOMBRES[col], value=bool(F["tipicos"][col])))


def texto_macro(r):
    return (f"Desempleo {r['Unemployment rate']:.1f} % · Inflación {r['Inflation rate']:.1f} % · "
            f"PIB {r['GDP']:+.2f} %").replace(".", ",")


def grafico_probabilidades(fila):
    d = pd.DataFrame({"Resultado": [NOMBRE_CLASE[c] for c in CLASES],
                      "Probabilidad": [float(fila[f"prob_{c}"]) for c in CLASES]})
    base = alt.Chart(d).encode(
        y=alt.Y("Resultado:N", sort=[NOMBRE_CLASE[c] for c in CLASES], title=None),
        x=alt.X("Probabilidad:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None))
    barras = base.mark_bar().encode(color=alt.Color("Resultado:N", scale=ESCALA_CLASES, legend=None),
                                    tooltip=["Resultado", alt.Tooltip("Probabilidad:Q", format=".1%")])
    texto = base.mark_text(align="left", dx=4).encode(text=alt.Text("Probabilidad:Q", format=".0%"))
    return (barras + texto).properties(height=140)


def matriz_confusion(matriz, clases, titulo_y="Real", titulo_x="Predicho"):
    filas = [{"Real": NOMBRE_CLASE[r], "Predicho": NOMBRE_CLASE[p], "n": int(matriz[i][j])}
             for i, r in enumerate(clases) for j, p in enumerate(clases)]
    d = pd.DataFrame(filas)
    orden = [NOMBRE_CLASE[c] for c in clases]
    base = alt.Chart(d).encode(x=alt.X("Predicho:N", sort=orden, title=titulo_x),
                               y=alt.Y("Real:N", sort=orden, title=titulo_y))
    celdas = base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                     tooltip=["Real", "Predicho", "n"])
    texto = base.mark_text(fontSize=15).encode(
        text="n:Q", color=alt.condition(alt.datum.n > float(d["n"].max()) / 2, alt.value("white"),
                                        alt.value("black")))
    return (celdas + texto).properties(height=260)


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
        d["Application order"] = numero("Application order", c1)
        d["Daytime/evening attendance"] = c2.radio(NOMBRES["Daytime/evening attendance"], [1, 0], horizontal=True,
                                                   index=[1, 0].index(F["tipicos"]["Daytime/evening attendance"]),
                                                   format_func=lambda v: etiqueta("Daytime/evening attendance", v))
        d["Gender"] = c3.radio(NOMBRES["Gender"], [0, 1], horizontal=True,
                               index=[0, 1].index(F["tipicos"]["Gender"]),
                               format_func=lambda v: etiqueta("Gender", v))
        c1, c2, c3 = st.columns(3)
        d["Age at enrollment"] = numero("Age at enrollment", c1)
        d["Marital status"] = selector("Marital status", c2)
        d["Previous qualification"] = selector("Previous qualification", c3)
        c1, c2, c3 = st.columns(3)
        d["Previous qualification (grade)"] = numero("Previous qualification (grade)", c1)
        d["Admission grade"] = numero("Admission grade", c2)
        d["Displaced"] = binaria("Displaced", c3)

        st.markdown("##### 💰 Situación económica")
        c1, c2, c3 = st.columns(3)
        d["Tuition fees up to date"] = binaria("Tuition fees up to date", c1)
        d["Debtor"] = binaria("Debtor", c2)
        d["Scholarship holder"] = binaria("Scholarship holder", c3)

        for s, titulo in [(1, "📘 Rendimiento del 1.er semestre"), (2, "📗 Rendimiento del 2.º semestre")]:
            st.markdown(f"##### {titulo}")
            cols = st.columns(6)
            for c, campo in zip(cols, ["enrolled", "evaluations", "approved", "grade", "credited",
                                       "without evaluations"]):
                col = f"S{s} ({campo})"
                minimo, maximo, paso = RANGOS[col]
                valor = min(max(type(minimo)(F["tipicos"][col]), minimo), maximo)
                d[col] = c.number_input(NOMBRES[col].split("· ")[1].capitalize(), min_value=minimo,
                                        max_value=maximo, value=valor, step=paso, key=col)

        with st.expander("👪 Familia y contexto económico (opcional: viene con los valores más comunes)"):
            c1, c2 = st.columns(2)
            for col, cont in [("Mother's qualification", c1), ("Father's qualification", c2),
                              ("Mother's occupation", c1), ("Father's occupation", c2)]:
                d[col] = selector(col, cont)
            opciones_macro = F["macro"]
            tipico_macro = next((i for i, r in enumerate(opciones_macro)
                                 if all(r[k] == F["tipicos"][k] for k in MACRO)), 0)
            elegido = st.selectbox("Contexto económico del año de matrícula", range(len(opciones_macro)),
                                   index=tipico_macro, format_func=lambda i: texto_macro(opciones_macro[i]))
            d.update({k: opciones_macro[elegido][k] for k in MACRO})

        enviado = st.form_submit_button("Evaluar riesgo de deserción", type="primary", width="stretch")

    if enviado:
        st.session_state["evaluacion"] = (d, predecir(d).iloc[0])

    with col_res:
        if "evaluacion" not in st.session_state:
            st.info("Completa el formulario y presiona **Evaluar riesgo de deserción**. "
                    "Los campos vienen con los valores más comunes del dataset.")
        else:
            d_eval, r = st.session_state["evaluacion"]
            nivel = r["nivel_riesgo"]
            st.markdown(f"""
            <div class="tarjeta {nivel.lower()}">
              <div class="titulo">PROBABILIDAD DE DESERCIÓN</div>
              <div class="valor">{formato_pct(r['prob_Dropout'])}</div>
              <div class="sub">Clasificación: <b>{r['alerta']}</b> · confianza {formato_pct(r['confianza'])}</div>
              <div class="sub">Riesgo <b>{nivel.lower()}</b> · resultado más probable: {r['resultado_estimado'].lower()}</div>
            </div>""", unsafe_allow_html=True)
            st.write("")
            st.markdown("**Probabilidad de cada resultado**")
            st.altair_chart(grafico_probabilidades(r))

            avisos = []
            for s in (1, 2):
                if d_eval[f"S{s} (approved)"] > d_eval[f"S{s} (enrolled)"]:
                    avisos.append(f"En el semestre {s} hay más unidades aprobadas que inscritas.")
                if d_eval[f"S{s} (approved)"] > d_eval[f"S{s} (evaluations)"]:
                    avisos.append(f"En el semestre {s} hay más unidades aprobadas que evaluaciones.")
            if avisos:
                st.warning("Revisa los datos: " + " ".join(avisos) +
                           " En el dataset de entrenamiento esto nunca ocurre.")

            senales = senales_de_riesgo(d_eval)
            st.markdown("**Señales de riesgo encontradas en el análisis exploratorio**")
            if senales:
                st.markdown("\n".join(f"- {s}" for s in senales))
            else:
                st.markdown("- Ninguna de las señales principales del EDA está presente.")

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
            plantilla = datos()[COLUMNAS].head(5)
            st.download_button("⬇️ Descargar plantilla (5 estudiantes)",
                               plantilla.to_csv(sep=";", index=False).encode("utf-8-sig"),
                               file_name="plantilla_estudiantes.csv", mime="text/csv")
            demo = st.toggle("Usar los estudiantes del conjunto de prueba (ejemplo)",
                             help="Los estudiantes que el modelo no vio al entrenar, con su resultado real.")
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
            c1.metric("Estudiantes", f"{n:,}".replace(",", "."))
            c2.metric("Riesgo alto", int(conteo.get("Alto", 0)), border=True)
            c3.metric("Riesgo medio", int(conteo.get("Medio", 0)), border=True)
            c4.metric("Clasificados como desertores", f"{(resultado['alerta'] == 'Desertor').mean():.0%}")

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Distribución de la probabilidad de deserción**")
                hist = alt.Chart(resultado[["prob_Dropout", "nivel_riesgo"]]).mark_bar().encode(
                    x=alt.X("prob_Dropout:Q", bin=alt.Bin(step=0.05), axis=alt.Axis(format="%"),
                            title="Probabilidad de deserción"),
                    y=alt.Y("count():Q", title="Estudiantes"),
                    color=alt.Color("nivel_riesgo:N", title="Riesgo",
                                    scale=alt.Scale(domain=list(COLOR_RIESGO), range=list(COLOR_RIESGO.values()))))
                st.altair_chart(hist.properties(height=280))
            with g2:
                st.markdown("**Resultado estimado**")
                est = resultado["resultado_estimado"].value_counts().rename_axis("Resultado").reset_index(name="n")
                barras = alt.Chart(est).mark_bar().encode(
                    x=alt.X("n:Q", title="Estudiantes"),
                    y=alt.Y("Resultado:N", sort=[NOMBRE_CLASE[c] for c in CLASES], title=None),
                    color=alt.Color("Resultado:N", scale=ESCALA_CLASES, legend=None), tooltip=["Resultado", "n"])
                st.altair_chart(barras.properties(height=280))

            if "Target" in resultado.columns and resultado["Target"].isin(CLASES).all():
                from sklearn.metrics import confusion_matrix, f1_score, recall_score
                real, pred = resultado["Target"], resultado["prediccion"]
                st.markdown("**Comparación con el resultado real**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Exactitud", f"{(real == pred).mean():.1%}")
                c2.metric("F1 macro", f"{f1_score(real, pred, average='macro'):.3f}")
                c3.metric("Recall de deserción",
                          f"{recall_score(real, pred, labels=['Dropout'], average='macro', zero_division=0):.1%}"
                          if (real == "Dropout").any() else "Sin desertores reales")
                st.altair_chart(matriz_confusion(confusion_matrix(real, pred, labels=CLASES).tolist(), CLASES))

            st.markdown("**Estudiantes ordenados por riesgo**")
            vista = resultado.sort_values("prob_Dropout", ascending=False)
            columnas_vista = (["prob_Dropout", "nivel_riesgo", "alerta", "resultado_estimado"]
                              + (["Target"] if "Target" in vista.columns else [])
                              + ["Course", "Tuition fees up to date", "Debtor", "Scholarship holder",
                                 "S1 (approved)", "S2 (approved)"])
            tabla = vista[columnas_vista].copy()
            tabla["Course"] = tabla["Course"].map(lambda c: etiqueta("Course", c))
            config = {
                "prob_Dropout": st.column_config.ProgressColumn("Prob. deserción", format="percent",
                                                                min_value=0.0, max_value=1.0),
                "nivel_riesgo": "Riesgo", "alerta": "Clasificación", "resultado_estimado": "Resultado estimado",
                "Target": "Resultado real", "Course": "Programa", "Tuition fees up to date": "Matrícula al día",
                "Debtor": "Deudor", "Scholarship holder": "Becado",
                "S1 (approved)": "Aprobadas S1", "S2 (approved)": "Aprobadas S2"}
            st.dataframe(tabla, hide_index=True, width="stretch", height=380,
                         column_config={k: v for k, v in config.items() if k in tabla.columns})
            st.download_button("⬇️ Descargar resultados (CSV)",
                               vista.to_csv(sep=";", index=False).encode("utf-8-sig"),
                               file_name="resultados_desercion.csv", mime="text/csv", type="primary")


# ===========================================================================
# 3. Análisis exploratorio (Entrega 1)
# ===========================================================================
@st.cache_data
def eda_basico():
    df = datos().copy()
    df["dropout_bin"] = (df["Target"] == "Dropout").astype(int)
    return df


@st.cache_data
def v_cramer_todas():
    df = eda_basico()

    def v_cramer(x, y):
        tabla = pd.crosstab(x, y)
        chi2 = chi2_contingency(tabla, correction=False)[0]
        r, k = tabla.shape
        return np.sqrt((chi2 / tabla.values.sum()) / (min(r, k) - 1))

    variables = BINARIAS + CATEGORICAS
    return (pd.Series({NOMBRES[c]: v_cramer(df[c], df["Target"]) for c in variables})
            .sort_values(ascending=False).rename_axis("Variable").reset_index(name="V"))


with tab_eda:
    if not hay_datos:
        st.info("Para ver el análisis exploratorio, incluye `data/data.csv` en el proyecto.")
    else:
        df = eda_basico()
        tasa_global = float(df["dropout_bin"].mean())
        dup = int(df.drop(columns="dropout_bin").duplicated().sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estudiantes", f"{len(df):,}".replace(",", "."))
        c2.metric("Variables originales", 36)
        c3.metric("Valores nulos", int(df.isna().sum().sum()))
        c4.metric("Filas duplicadas", dup)

        st.subheader("La variable objetivo y el desbalance de clases")
        g1, g2 = st.columns([2, 3])
        conteo = (df["Target"].value_counts().reindex(CLASES).rename_axis("Target")
                  .reset_index(name="Estudiantes"))
        conteo["Resultado"] = conteo["Target"].map(NOMBRE_CLASE)
        conteo["Proporción"] = conteo["Estudiantes"] / len(df)
        base = alt.Chart(conteo).encode(x=alt.X("Resultado:N", sort=[NOMBRE_CLASE[c] for c in CLASES], title=None),
                                        y=alt.Y("Estudiantes:Q"))
        g1.altair_chart((base.mark_bar().encode(color=alt.Color("Resultado:N", scale=ESCALA_CLASES, legend=None),
                                                tooltip=["Resultado", "Estudiantes"])
                         + base.mark_text(dy=-8).encode(text=alt.Text("Proporción:Q", format=".1%")))
                        .properties(height=300))
        g2.markdown(f"""
- **Desbalance:** la clase mayoritaria tiene {conteo['Estudiantes'].max() / conteo['Estudiantes'].min():.2f}
  veces más estudiantes que la minoritaria.
- Un "modelo" que siempre diga *Se gradúa* acertaría el {conteo['Proporción'].max():.1%} de las veces, por eso
  la **exactitud sola engaña**: se evalúa con F1 macro, balanced accuracy y recall por clase.
- La partición entrenamiento/prueba es **estratificada** y el balanceo (class_weight o SMOTE) se aplica solo
  sobre el entrenamiento.
""")

        st.subheader("Tasa de deserción por categoría")
        opciones_cat = ["Tuition fees up to date", "Scholarship holder", "Debtor", "Gender", "Displaced",
                        "Daytime/evening attendance", "Marital status", "Application mode", "Course"]
        var = st.selectbox("Variable", opciones_cat, format_func=lambda c: NOMBRES[c], key="eda_cat")
        tasas = df.groupby(var)["dropout_bin"].agg(["mean", "size"]).reset_index()
        tasas["Categoría"] = tasas[var].map(lambda k: etiqueta(var, k))
        tasas = tasas.rename(columns={"mean": "Tasa", "size": "Estudiantes"})[["Categoría", "Tasa", "Estudiantes"]]
        barras = alt.Chart(tasas).mark_bar().encode(
            x=alt.X("Tasa:Q", axis=alt.Axis(format="%"), title="Tasa de deserción"),
            y=alt.Y("Categoría:N", sort="-x", title=None),
            color=alt.condition(alt.datum.Tasa > tasa_global, alt.value(COLORES["Dropout"]), alt.value("#9ca3af")),
            tooltip=["Categoría", alt.Tooltip("Tasa:Q", format=".1%"), "Estudiantes"])
        regla = alt.Chart(pd.DataFrame({"x": [tasa_global]})).mark_rule(color="red", strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((barras + regla).properties(height=max(160, 26 * len(tasas))))
        st.caption(f"Línea roja: tasa global de deserción ({tasa_global:.1%}). En rojo, las categorías por encima.")

        st.subheader("Distribución de las variables numéricas según el resultado")
        opciones_num = ["S1 (approved)", "S2 (approved)", "S1 (grade)", "S2 (grade)",
                        "Age at enrollment", "Admission grade", "Previous qualification (grade)"]
        var_num = st.selectbox("Variable", opciones_num, format_func=lambda c: NOMBRES[c], key="eda_num")
        dist = pd.DataFrame({"valor": df[var_num].astype(float), "Resultado": df["Target"].map(NOMBRE_CLASE)})
        densidad = alt.Chart(dist).transform_density(
            "valor", groupby=["Resultado"], as_=["valor", "densidad"]).mark_area(opacity=0.45).encode(
            x=alt.X("valor:Q", title=NOMBRES[var_num]), y=alt.Y("densidad:Q", title="Densidad"),
            color=alt.Color("Resultado:N", scale=ESCALA_CLASES))
        st.altair_chart(densidad.properties(height=300))

        st.subheader("El rendimiento del 1.er semestre es la señal más fuerte")
        rangos = pd.cut(df["S1 (approved)"], bins=[-1, 0, 2, 4, 6, 30], labels=["0", "1-2", "3-4", "5-6", "7+"])
        curva = (df.assign(rango=rangos).groupby("rango", observed=True)
                 .agg(Deserta=("dropout_bin", "mean"), **{"Se gradúa": ("Target", lambda s: (s == "Graduate").mean())})
                 .reset_index().melt("rango", var_name="Resultado", value_name="Proporción"))
        linea = alt.Chart(curva).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("rango:N", sort=["0", "1-2", "3-4", "5-6", "7+"], title="Unidades aprobadas en el 1.er semestre"),
            y=alt.Y("Proporción:Q", axis=alt.Axis(format="%"), title="Proporción de estudiantes"),
            color=alt.Color("Resultado:N", scale=ESCALA_CLASES),
            tooltip=["rango", "Resultado", alt.Tooltip("Proporción:Q", format=".0%")])
        st.altair_chart(linea.properties(height=300))

        st.subheader("Qué tan asociada está cada variable con el resultado")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**V de Cramér** (categóricas y binarias; 0 = sin asociación, 1 = perfecta)")
            vc = v_cramer_todas()
            st.altair_chart(alt.Chart(vc).mark_bar(color=COLORES["Graduate"]).encode(
                x=alt.X("V:Q", title="V de Cramér"), y=alt.Y("Variable:N", sort="-x", title=None),
                tooltip=["Variable", alt.Tooltip("V:Q", format=".3f")]).properties(height=360))
        with g2:
            st.markdown("**Correlación entre variables numéricas** (y con la deserción)")
            cols_corr = ["Admission grade", "Age at enrollment", "S1 (enrolled)", "S1 (approved)", "S1 (grade)",
                         "S2 (enrolled)", "S2 (approved)", "S2 (grade)", "dropout_bin"]
            corr = df[cols_corr].corr().rename(index={"dropout_bin": "Deserción"}, columns={"dropout_bin": "Deserción"})
            largo = corr.stack().rename_axis(["fila", "columna"]).reset_index(name="r")
            orden = list(corr.columns)
            base = alt.Chart(largo).encode(x=alt.X("columna:N", sort=orden, title=None),
                                           y=alt.Y("fila:N", sort=orden, title=None))
            st.altair_chart((base.mark_rect().encode(
                color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), legend=None),
                tooltip=["fila", "columna", alt.Tooltip("r:Q", format=".2f")])
                + base.mark_text(fontSize=10).encode(text=alt.Text("r:Q", format=".2f")))
                .properties(height=360))

        st.subheader("Resumen de decisiones (Entrega 1)")
        st.markdown("""
- El conjunto está **limpio**: sin nulos, duplicados ni valores incoherentes.
- Se eliminan tres variables casi constantes: **Educational special needs**, **International** y **Nacionality**.
- Clases desbalanceadas → partición estratificada; métricas: balanced accuracy, F1 macro y recall por clase.
- El desbalance se trata en el modelado comparando **class_weight** y **SMOTE** dentro de un pipeline con
  validación cruzada, aplicado solo sobre el entrenamiento.
- El **rendimiento del 1.er semestre** es la señal más fuerte, seguido de los factores económicos
  (matrícula al día y beca).
- Las variables del 1.er y 2.º semestre están muy correlacionadas (0,84 a 0,94).
""")


# ===========================================================================
# 4. Sobre el modelo
# ===========================================================================
with tab_modelo:
    p = M["prueba"]
    st.subheader(f"{M['modelo']} con balanceo «{M['balanceo']}»")
    st.write(f"Entrenado con {M['filas_entrenamiento']:,} estudiantes y evaluado con "
             f"{M['filas_prueba']:,} que no vio durante el entrenamiento.".replace(",", "."))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 macro", f"{p['f1_macro']:.3f}", border=True)
    c2.metric("Balanced accuracy", f"{p['balanced_accuracy']:.3f}", border=True)
    c3.metric("Recall de deserción", f"{p['recall']['Dropout']:.1%}", border=True,
              help="De los estudiantes que realmente desertaron, qué porcentaje detecta el modelo.")
    c4.metric("AUC (one-vs-rest)", f"{p['auc_ovr']:.3f}", border=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Matriz de confusión (conjunto de prueba)**")
        st.altair_chart(matriz_confusion(M["matriz_confusion"], M["clases"]))
        por_clase = pd.DataFrame({"Resultado": [NOMBRE_CLASE[c] for c in CLASES],
                                  "Recall": [p["recall"][c] for c in CLASES],
                                  "Precisión": [p["precision"][c] for c in CLASES]})
        st.dataframe(por_clase, hide_index=True, width="stretch", column_config={
            "Recall": st.column_config.NumberColumn(format="percent"),
            "Precisión": st.column_config.NumberColumn(format="percent")})
    with g2:
        st.markdown("**Variables más importantes** (caída del F1 macro al desordenar cada variable)")
        imp = (pd.Series(M["importancias"]).head(15).rename_axis("Variable").reset_index(name="Importancia"))
        imp["Variable"] = imp["Variable"].map(lambda c: NOMBRES.get(c, c))
        st.altair_chart(alt.Chart(imp).mark_bar(color=COLORES["Dropout"]).encode(
            x=alt.X("Importancia:Q", title="Caída del F1 macro"), y=alt.Y("Variable:N", sort="-x", title=None),
            tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".4f")]).properties(height=420))

    st.markdown("**Comparación de modelos** (validación cruzada estratificada de 5 pliegues sobre entrenamiento)")
    comp = pd.DataFrame(M["comparacion"]).sort_values("f1_macro", ascending=False)
    st.dataframe(comp, hide_index=True, width="stretch", column_config={
        "modelo": "Modelo", "balanceo": "Balanceo", "f1_macro": st.column_config.NumberColumn("F1 macro", format="%.3f"),
        "f1_macro_std": st.column_config.NumberColumn("± desv.", format="%.3f"),
        "balanced_accuracy": st.column_config.NumberColumn("Balanced acc.", format="%.3f"),
        "recall_dropout": st.column_config.NumberColumn("Recall deserción", format="%.3f"),
        "segundos": st.column_config.NumberColumn("Tiempo (s)", format="%.1f")})
    st.caption("Preprocesamiento: one-hot para las categóricas (categorías con menos del 1 % agrupadas) y "
               "estandarización para las numéricas y binarias. Datos de una institución de educación superior "
               "de Portugal: antes de usarlo con estudiantes de otra institución hay que reentrenarlo con sus datos.")
