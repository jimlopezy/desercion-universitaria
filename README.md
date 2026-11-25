# 🎓 Deserción universitaria: sistema de alerta temprana

Sistema de análisis, monitoreo y alerta temprana de la deserción universitaria basado en
aprendizaje automático. Clasifica a cada estudiante como **desertor / no desertor** con un
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
| 🧑‍🎓 **Evaluar estudiante** | Formulario con las 33 variables → probabilidad de desertar, seguir matriculado o graduarse, nivel de riesgo y señales encontradas en el EDA |
| 📂 **Evaluar grupo (CSV)** | Sube un CSV con muchos estudiantes → panel con los de mayor riesgo y descarga de resultados. Trae un ejemplo con el conjunto de prueba |
| 📊 **Análisis exploratorio** | El EDA de la Entrega 1, interactivo: desbalance, tasa de deserción por categoría, distribuciones, V de Cramér y correlaciones |
| 🤖 **Sobre el modelo** | Comparación de modelos, métricas en prueba, matriz de confusión e importancia de las variables |

## Cómo se construyó el modelo (decisiones de la Entrega 1)

- Limpia los nombres de columnas y abrevia los semestres (`S1`, `S2`).
- Elimina las 3 variables casi constantes: `Educational special needs`, `International` y `Nacionality` (quedan 33).
- Conserva las 3 clases (`Dropout`, `Enrolled`, `Graduate`). La alerta es *Desertor* cuando la clase más probable es `Dropout`.
- Hace una partición estratificada 80/20 y una validación cruzada estratificada de 5 pliegues.
- Compara regresión logística, árbol de decisión, random forest y perceptrón multicapa.
- Trata el desbalance con `class_weight="balanced"` y, si se instala `imbalanced-learn`, también con **SMOTE** dentro del pipeline (solo sobre el entrenamiento).
- Elige el modelo por **F1 macro** y reporta balanced accuracy y recall por clase.
- Hace one-hot de las categóricas y agrupa las categorías con menos del 1 %. Estandariza el resto.

**Resultado actual:** Random Forest + `class_weight`.

| Conjunto | F1 macro | Balanced accuracy | Exactitud | AUC | Recall por clase (Dropout / Enrolled / Graduate) |
|---|---|---|---|---|---|
| Validación cruzada (5 pliegues) | 0,726 | 0,728 | — | — | Dropout 0,735 |
| Prueba (885 estudiantes nunca vistos) | 0,705 | 0,712 | 0,748 | 0,888 | 0,70 / 0,60 / 0,83 |

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
│   ├── modelo_desercion.pkl    ← modelo entrenado
│   └── metricas_desercion.json ← métricas, comparación y datos del formulario
└── src/desercion_universitaria/
    ├── __init__.py
    ├── core.py                 ← preparación de datos, carga del modelo y predicción
    ├── entrenar.py             ← compara los modelos y guarda el mejor
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
| `uv run entrenar` | Vuelve a entrenar y guarda el mejor modelo en `modelo/` |
| `uv add imbalanced-learn` | (Opcional) incluye SMOTE en la comparación del entrenamiento |

> 💡 **Para el modelo final (Entrega 2):** agréguenlo a `crear_modelo()` y a la lista `ALGORITMOS`
> en `entrenar.py` y ejecuten `uv run entrenar`. La app se actualiza sola porque lee `modelo/`.

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
computador. Eso importa porque el `.pkl` debe abrirse con la misma versión de scikit-learn con la
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
