"""
Lanzador de la app web de deserción, para poder escribir:  uv run app
(Streamlit no se ejecuta con `python archivo.py` sino con `streamlit run archivo.py`.)
"""
import sys
from pathlib import Path

from streamlit.web import cli


def main():
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(cli.main())
