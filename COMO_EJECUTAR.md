# 🎓 Cómo ejecutar el sistema de alerta temprana de deserción universitaria

El modelo **ya está entrenado** y los datos ya están en su lugar. Para abrir la app solo hacen
falta los pasos 1 a 3.

---

## 📁 Dónde está todo

Carpeta del proyecto (desde aquí se ejecutan **todos** los comandos):

```
C:\Users\jimmy\Downloads\desercion-universitaria
```

```
desercion-universitaria\
│
├── pyproject.toml              ← nombre del proyecto, librerías y comandos (entrenar, app)
├── uv.lock                     ← versiones exactas de las librerías (no se edita a mano)
├── .python-version             ← el proyecto usa Python 3.12
├── requirements.txt            ← lista de librerías por si se usa pip en lugar de uv
├── README.md                   ← descripción del proyecto, resultados y cómo publicarlo
├── COMO_EJECUTAR.md            ← esta guía
├── Dockerfile                  ← para publicar con Docker (Hugging Face, Render, servidor)
├── .dockerignore / .gitignore
├── .venv\                      ← entorno virtual (lo crea uv; no se toca ni se sube a GitHub)
│
├── data\
│   └── data.csv                ← dataset del notebook (UCI #697, 4.424 estudiantes)
│
├── modelo\
│   ├── modelo_desercion.joblib ← modelo entrenado (HistGradientBoosting de la Entrega 2, umbral 0,5)
│   └── metricas_desercion.json ← métricas, comparación de modelos, curvas y datos del formulario
│
└── src\desercion_universitaria\
    ├── __init__.py
    ├── core.py                 ← ingeniería de características de la Entrega 2, carga del modelo y predicción
    ├── entrenar.py             ← replica el modelado de la Entrega 2 y guarda el modelo en modelo\
    ├── app.py                  ← la app web (Streamlit) con sus 4 pestañas
    └── lanzar_app.py           ← permite abrir la app con:  uv run app
```

---

## Paso 1 · Abrir una terminal en la carpeta del proyecto

Elige una de estas formas:

- **Explorador de archivos:** entra a la carpeta del proyecto, haz clic derecho en un espacio vacío y elige **"Abrir en Terminal"**.
- **VS Code:** *Archivo → Abrir carpeta…*, eliges la carpeta y luego *Terminal → Nueva terminal*.
- **PowerShell:**
  ```powershell
  cd "C:\Users\jimmy\Downloads\desercion-universitaria"
  ```

Para comprobar que estás en el lugar correcto, escribe `dir`: debes ver `pyproject.toml`.

## Paso 2 · Preparar el entorno (la primera vez)

```powershell
uv --version
uv sync
```

`uv sync` crea el entorno `.venv` con exactamente las versiones de `uv.lock`.
Si `uv --version` da error, instala uv con
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`,
cierra la terminal y ábrela de nuevo.

## Paso 3 · Abrir la app

```powershell
uv run app
```

Se abre el navegador en **http://localhost:8501**. Si no se abre solo, copia esa dirección.
La terminal queda ocupada porque está funcionando como servidor: para detener la app presiona
**Ctrl + C**.

| Pestaña | Para qué sirve |
|---|---|
| 🧑‍🎓 Evaluar estudiante | Llenar el formulario y ver la probabilidad de abandono, la clasificación y el nivel de riesgo |
| 📂 Evaluar grupo (CSV) | Subir un CSV con varios estudiantes, o activar el ejemplo con el conjunto de prueba |
| 📊 Análisis de los datos | Exploración interactiva del dataset y resumen de la ingeniería de características |
| 🤖 Sobre el modelo | Métricas, curvas ROC y precisión-recall, comparación de modelos, balanceo e importancia de variables |

---

## 🔁 (Opcional) Volver a entrenar el modelo

Solo hace falta si cambian los datos (`data\data.csv`) o el código del modelo:

```powershell
uv run entrenar            # con los hiperparámetros elegidos en la Entrega 2
uv run entrenar --buscar   # repite la búsqueda aleatoria de hiperparámetros
```

Como `imbalanced-learn` ya está en el proyecto, el entrenamiento compara los 9 métodos de balanceo del
notebook. Para comparar también los 3 modelos (HistGB, LightGBM y CatBoost):

```powershell
uv add lightgbm catboost
uv run entrenar
```

El entrenamiento reemplaza `modelo\modelo_desercion.joblib` y `modelo\metricas_desercion.json`.
La próxima vez que abras la app, ya usa el modelo nuevo.

---

## 🌐 Publicarla en internet (Streamlit Community Cloud, gratis)

Necesitas una cuenta de [GitHub](https://github.com) y tener **git** instalado (`git --version`).

1. Crea un repositorio vacío en GitHub llamado `desercion-universitaria`.
2. En la terminal, dentro de la carpeta del proyecto:
   ```powershell
   git init
   git add .
   git commit -m "Sistema de alerta temprana de deserción universitaria"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/desercion-universitaria.git
   git push -u origin main
   ```
   El `.gitignore` ya excluye `.venv\`. Las carpetas `data\` y `modelo\` **sí** se suben, porque la app publicada las necesita.
3. Entra a **https://share.streamlit.io** con tu cuenta de GitHub y elige **Create app**.
4. Completa el formulario:
   - **Repository:** `<tu-usuario>/desercion-universitaria`
   - **Branch:** `main`
   - **Main file path:** `src/desercion_universitaria/app.py`
   - **Advanced settings → Python version:** `3.12`
5. Presiona **Deploy**. En unos minutos tendrás un enlace como `https://<nombre>.streamlit.app` para compartir.

Cada vez que hagas `git push`, la app publicada se actualiza sola. Las otras formas de publicar
(Hugging Face Spaces, Render, servidor propio con Docker) están en el `README.md`.

---

## 🆘 Si algo falla

| Mensaje | Solución |
|---|---|
| `uv : El término 'uv' no se reconoce...` | Instala uv (paso 2) y abre una terminal nueva |
| `error: No pyproject.toml found` | No estás en la carpeta del proyecto: repite el paso 1 |
| `Failed to spawn: app` | Ejecuta `uv sync` y revisa que `pyproject.toml` tenga `app = "desercion_universitaria.lanzar_app:main"` |
| Errores con rutas viejas o `No module named ...` después de renombrar | Borra la carpeta `.venv` y ejecuta `uv sync` |
| `No encontré el modelo. Ejecuta primero: uv run entrenar` | Falta la carpeta `modelo\`: ejecuta `uv run entrenar` |
| `Falta el dataset...` | Falta `data\data.csv`: cópialo de nuevo en la carpeta `data\` |
| El navegador dice que no puede conectar | La terminal se cerró o se detuvo con Ctrl + C: vuelve a ejecutar `uv run app` |
| `Port 8501 is already in use` | Ya hay una app abierta: ciérrala con Ctrl + C o usa `uv run app --server.port 8502` |
