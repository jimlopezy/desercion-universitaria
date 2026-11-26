# 🎓 Deserción universitaria: sistema de alerta temprana

Sistema de análisis, monitoreo y alerta temprana de la deserción universitaria basado en
aprendizaje automático. Clasifica a cada estudiante como **abandono / no abandono** con un
porcentaje de confianza, para ofrecer apoyo a tiempo a quien lo necesita.

**Integrantes:** David Santiago Roa Mayor · Maira Alejandra Balanta Peña · Jimmy Lopez Yule ·
Marco Antonio Aragon Vivas · Esteban Cobo Gómez

**Datos:** [UCI #697 *Predict Students' Dropout and Academic Success*](https://archive.ics.uci.edu/dataset/697)
(4.424 estudiantes, 36 variables y 1 objetivo, licencia CC BY 4.0).

> 👉 Para ejecutarlo paso a paso, mira **[COMO_EJECUTAR.md](COMO_EJECUTAR.md)**.

---

## La app web

| Pestaña | Qué hace |
|---|---|
| 🧑‍🎓 **Evaluar estudiante** | Formulario con las 28 variables del modelo → probabilidad de abandono, clasificación *Abandono / No abandono* con su confianza, nivel de riesgo y señales encontradas en los datos |
| 📂 **Evaluar grupo (CSV)** | Sube un CSV con muchos estudiantes (formato UCI) → panel con los de mayor riesgo y descarga de resultados. Trae un ejemplo con los 813 estudiantes de prueba |
| 📊 **Análisis de los datos** | Exploración interactiva con el objetivo binario: tasa de abandono por categoría, distribuciones, V de Cramér, correlaciones y resumen de la ingeniería de características |
| 🤖 **Sobre el modelo** | Métricas en prueba, matriz de confusión, curvas ROC y precisión-recall, comparación de modelos con IC95 %, balanceo de clases e importancia de variables |

## Cómo se construyó el modelo (Entrega 2)

La app replica el notebook `Entrega2Proyecto.ipynb`:

1. **Objetivo binario:** *Dropout* = abandono (1); *Enrolled* y *Graduate* = no abandono (0).
2. **Filtro:** se quitan los estudiantes sin unidades inscritas en el 1.er o en el 2.º semestre, y los que tienen códigos fuera de las equivalencias. Quedan 4.065 de 4.424.
3. **Variables descartadas** por irrelevantes o correlacionadas: `Unemployment rate`, `Inflation rate`, `GDP`, `Nacionality`, `S2 (enrolled)`, `S2 (approved)`, `S2 (grade)` y `S2 (evaluations)`.
4. **Codificación:**
   - Las 5 nominales (estado civil, modo de postulación, programa y ocupación de la madre y del padre) pasan a texto con tipo *category*.
   - La escolaridad pasa a escala ordinal: formación previa (0-13) y madre/padre (0-23).
   - Quedan **28 variables**.
5. **Partición** estratificada 80/20 (3.252 / 813) y validación cruzada estratificada de 5 pliegues.
6. **Balanceo:** se comparan 9 métodos con regresión logística. Se adopta `class_weight="balanced"`, que mejora el recall y el F1 frente a no balancear y que el sobremuestreo no supera.
7. **Modelos** con categóricas nativas: HistGradientBoosting, LightGBM y CatBoost. Sus hiperparámetros se ajustaron con `RandomizedSearchCV` (15 combinaciones, **Average Precision**).
8. **Elección:** gana el mejor AP en validación cruzada, que es **HistGradientBoosting**, con umbral de decisión 0,5.

**Resultado del modelo desplegado** (HistGradientBoosting):

| Conjunto | AP | AUC | Recall | Precisión | F1 | Balanced acc. |
|---|---|---|---|---|---|---|
| Validación cruzada (5 pliegues) | 0,8425 [0,8295; 0,8555] | 0,8930 | — | — | 0,7481 | — |
| Prueba (813 estudiantes nunca vistos) | 0,8759 | 0,9155 | 0,8097 | 0,7605 | 0,7843 | 0,8492 |

> En el notebook, la prueba dio AP 0,8765, AUC 0,9164 y F1 0,7767. Las pequeñas diferencias vienen de
> la versión de scikit-learn: este proyecto usa la 1.9.1, que es la fijada en `uv.lock`.

## Estructura del proyecto

```
desercion-universitaria/
├── pyproject.toml              ← nombre, librerías y comandos (entrenar, app)
├── uv.lock                     ← versiones exactas de las librerías
├── .python-version             ← Python 3.12
├── requirements.txt            ← alternativa con pip
├── README.md / COMO_EJECUTAR.md
├── Dockerfile / .dockerignore  ← publicación con Docker
├── .gitignore
├── data/
│   └── data.csv                ← dataset UCI #697 (separado por ";")
├── modelo/
│   ├── modelo_desercion.joblib ← artefacto: modelo + columnas + categorías + umbral (como en el notebook)
│   └── metricas_desercion.json ← métricas, comparación, curvas y datos del formulario
└── src/desercion_universitaria/
    ├── __init__.py
    ├── core.py                 ← ingeniería de características de la Entrega 2, carga del modelo y predicción
    ├── entrenar.py             ← replica el modelado de la Entrega 2 y guarda el mejor modelo
    ├── app.py                  ← app web (Streamlit)
    └── lanzar_app.py           ← permite abrir la app con:  uv run app
```

El modelo vive en `core.py`, separado de la interfaz: la app solo arma los datos del estudiante,
llama a `predecir()` y muestra el resultado.

## Comandos

| Comando | Para qué |
|---|---|
| `uv sync` | Crea o actualiza el entorno `.venv` con las versiones de `uv.lock` |
| `uv run app` | Abre la app web en http://localhost:8501 |
| `uv run entrenar` | Vuelve a entrenar con los hiperparámetros elegidos en la Entrega 2 y guarda el modelo en `modelo/` |
| `uv run entrenar --buscar` | Repite la búsqueda aleatoria de hiperparámetros en lugar de usar los de la Entrega 2 |
| `uv add lightgbm catboost` | (Opcional) incluye LightGBM y CatBoost en la comparación de modelos (`imbalanced-learn`, para los 9 métodos de balanceo, ya está en el proyecto) |

Si instalas LightGBM o CatBoost y alguno resulta mejor, la app publicada también necesita esas
librerías. Como `uv add` las anota en `pyproject.toml` y en `uv.lock`, basta con subir esos archivos.

---

## 🌐 Poner la app en línea

Para cualquier opción, el repositorio debe incluir `data/data.csv` y la carpeta `modelo/` ya
entrenada. La app publicada no entrena: solo carga el modelo.

### Opción A · Streamlit Community Cloud (gratis, la más sencilla)

1. Sube el proyecto a un repositorio de GitHub (los comandos están en `COMO_EJECUTAR.md`).
2. Entra a **[share.streamlit.io](https://share.streamlit.io)** con GitHub y elige **Create app**.
3. Elige el repositorio y la rama `main`. En **Main file path** escribe `src/desercion_universitaria/app.py`.
4. En **Advanced settings** elige **Python 3.12** y presiona **Deploy**.
5. La app queda en una dirección como `https://<nombre>.streamlit.app`.

Community Cloud instala las librerías desde `uv.lock`, así que usa las mismas versiones que tu
computador. Eso importa porque el modelo (`.joblib`) debe abrirse con la misma versión de scikit-learn con la
que se entrenó. Cada `git push` vuelve a desplegar la app.

### Opción B · Hugging Face Spaces (gratis, con Docker)

Hugging Face ya no ofrece Streamlit como SDK propio: se publica con el SDK **Docker**, usando el
`Dockerfile` del proyecto.

1. Crea un Space en [huggingface.co/new-space](https://huggingface.co/new-space) con **SDK: Docker** (plantilla *Blank*).
2. Sube los archivos del proyecto (sin `.venv/`), por ejemplo desde *Files → Add file → Upload files*.
3. El `README.md` del Space debe empezar con este bloque:
   ```yaml
   ---
   title: Deserción universitaria
   emoji: 🎓
   colorFrom: blue
   colorTo: red
   sdk: docker
   app_port: 8501
   pinned: false
   ---
   ```

### Opción C · Render u otro servicio con Docker

En [render.com](https://render.com) crea un **New → Web Service**, conecta el repositorio de GitHub
y Render detecta el `Dockerfile`. El contenedor ya lee el puerto de la variable `PORT`.
En los planes gratuitos la app se duerme si nadie la visita y tarda unos segundos en despertar.

### Opción D · Servidor propio

```bash
docker build -t desercion-universitaria .
docker run -d -p 8501:8501 --restart unless-stopped desercion-universitaria
```

La app queda en `http://<ip-del-servidor>:8501`. Para usar un dominio con HTTPS, pon delante un
proxy inverso (Nginx o Caddy).

### Hacia la arquitectura de la propuesta (Docker + FastAPI + MySQL)

1. **API (FastAPI):** un endpoint `POST /predecir` que llame a `core.predecir()`.
2. **Base de datos (MySQL):** guardar cada estudiante evaluado con su probabilidad, clase y fecha,
   para los reportes acumulados del panel administrativo.
3. **Frontend:** esta misma app de Streamlit, llamando a la API en lugar de cargar el modelo.
4. **`docker-compose.yml`** con los servicios `api`, `db` y `web`.
