# Imagen de la app web de alerta temprana de deserción universitaria.
# Sirve para Hugging Face Spaces (SDK Docker), Render, Railway o cualquier servidor con Docker.
#
#   docker build -t desercion-universitaria .
#   docker run -p 8501:8501 desercion-universitaria      ->  http://localhost:8501

FROM python:3.12-slim

# uv: el mismo gestor de paquetes que usamos en local
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Usuario sin privilegios con UID 1000 (lo recomienda Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR $HOME/app

# 1) Librerías: con uv.lock se instalan exactamente las mismas versiones que en local
#    (importante: el modelo .joblib se debe cargar con la misma versión de scikit-learn).
COPY --chown=user pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Código, modelo entrenado y datos (el análisis exploratorio usa data/data.csv)
COPY --chown=user src ./src
COPY --chown=user modelo ./modelo
COPY --chown=user data ./data
RUN uv sync --frozen --no-dev

ENV PATH="$HOME/app/.venv/bin:$PATH"
EXPOSE 8501

# Render y Railway asignan el puerto en la variable PORT; Hugging Face Spaces usa 8501.
CMD ["sh", "-c", "streamlit run src/desercion_universitaria/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
