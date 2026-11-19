from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import difflib
import html
import io
import json
import os
import random
import re
import sqlite3
import threading
import time
import unicodedata

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit.errors import StreamlitAPIException
import pypdf

# Requiere Streamlit >= 1.37 (st.fragment, run_every y st.rerun(scope="fragment"))

st.set_page_config(
    page_title="Gestor de Apuntes y Tests TAI", page_icon="📚", layout="wide"
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none !important; }
    .block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }
    div.stButton > button { padding: 2px 10px; min-height: 30px; margin: 0px 0px; }
    hr { margin-top: 0.8rem !important; margin-bottom: 0.8rem !important; }
    /* cajetín de cada apunte (único contenedor con borde de la app) */
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 0.4rem 0.6rem !important; }
    .st-key-fila_grupos label p { overflow: visible !important; text-overflow: clip !important; white-space: normal !important; }
    .st-key-fila_acciones [data-testid="stHorizontalBlock"] { gap: 0.2rem !important; align-items: flex-start !important; }
    /* misma altura para botones y casilla «Término» (los que llevan tooltip salían más altos) */
    .st-key-fila_acciones button { height: 2.5rem !important; min-height: 2.5rem !important; padding-top: 0 !important; padding-bottom: 0 !important; }
    .st-key-fila_acciones [class*="st-key-chk_"] button { background-color: #43a047 !important; border-color: #43a047 !important; }
    .st-key-fila_acciones [class*="st-key-chk_"] button:hover { background-color: #388e3c !important; border-color: #388e3c !important; }
    .st-key-fila_acciones [class*="st-key-chk_"] button p { color: #fff !important; }
    .st-key-fila_acciones [data-testid="stTooltipHoverTarget"] { width: 100% !important; display: block !important; }
    .st-key-fila_acciones [data-testid="stTextInput"] { margin: 0 !important; }
    .st-key-fila_acciones [data-baseweb="input"],
    .st-key-fila_acciones [data-baseweb="base-input"] { height: 2.5rem !important; min-height: 2.5rem !important; }
    [class*="st-key-inline_f"] input, [class*="st-key-ap_f"] input { text-transform: uppercase; }

    @keyframes miniConfetiLargo {
        0% { opacity: 1; transform: translateY(0) scale(0.8); }
        70% { opacity: 1; transform: translateY(-10px) scale(1.1); }
        100% { opacity: 0; transform: translateY(-30px) scale(1.3); }
    }
    .celebracion-sutil {
        font-size: 1.4rem;
        display: inline-block;
        animation: miniConfetiLargo 4.0s ease-out forwards;
        margin-bottom: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

DB_APUNTES = "apuntes.db"
DB_TESTS = "tests.db"
DB_CURSO_IA = "CursoIA.db"
DB_TEMARIO = "Temario.db"  # BD intermedia: estado del temario independiente del .txt

# IA: Gemini a través de su endpoint compatible con OpenAI.
# La clave se lee de .streamlit/secrets.toml  ->  GOOGLE_API_KEY = "..."
MODELO_IA = "gemini-3.5-flash-lite"  # la capa gratuita da más cuota a los modelos lite (mira tus límites en Google AI Studio)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_REASONING = "minimal"  # poco razonamiento = respuestas rápidas y baratas. Pon None para el valor por defecto del modelo.
GEMINI_REINTENTOS_429 = 3     # reintentos si Google responde "demasiadas peticiones" (límite de la capa gratuita)
GEMINI_RPM = 9                # peticiones por minuto que se permite la app (pon un poco menos que tu límite real)
MAX_RESULTADOS_APUNTES = 20

# Límites de tokens ajustados
TOKENS_PREGUNTA = 300
TOKENS_EVALUACION = 130
TOKENS_PISTA = 450
TOKENS_RESOLVER = 420
TOKENS_SEGUIMIENTO = 700  # las dudas de seguimiento (ejemplos, etc.) necesitan más espacio que la explicación inicial

PREF_JOBS_TUTOR = ("job_exam_", "job_corr_")

NUM_PREGUNTAS_EXAMEN = 10
NOTA_APROBADO = 9
PREF_JOBS_TEST = ("job_ia_",)


def cargar_temario_desplegable_txt():
    ruta = Path("temario_tai.txt")
    temarios = []
    if not ruta.exists():
        return temarios

    bloque_actual = ""
    grupo_actual = ""

    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea_str = linea.strip()
            if not linea_str:
                continue
            if linea_str.startswith("[") and "]" in linea_str:
                bloque_actual = linea_str.replace("[", "").replace("]", "").strip()
            elif linea_str.startswith("- Grupo") or linea_str.startswith("-Grupo"):
                grupo_actual = re.sub(r"^-\s*Grupo\s*", "Grupo ", linea_str).strip()
            elif linea_str.startswith("Tema") or linea_str.startswith("   Tema"):
                tema_nombre = re.sub(r"^Tema\s*", "", linea_str).strip()
                if tema_nombre and bloque_actual and grupo_actual:
                    temarios.append((bloque_actual, grupo_actual, tema_nombre))
    return list(set(temarios))


def agregar_tema_a_txt(bloque: str, grupo: str, tema: str):
    ruta = Path("temario_tai.txt")
    bloque_buscado = bloque if bloque.startswith("[") else f"[{bloque}]"
    grupo_buscado = grupo if grupo.startswith("-") else f"- {grupo}"
    tema_buscado = tema if tema.startswith("Tema") else f"Tema {tema}"

    with open(ruta, "a", encoding="utf-8") as f:
        f.write(f"\n{bloque_buscado}\n{grupo_buscado}\n   {tema_buscado}\n")


# ==========================================
# UTILIDADES BASE DE DATOS
# ==========================================
@contextmanager
def db(path):
    conn = sqlite3.connect(path, timeout=10)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@st.cache_resource
def inicializar_dbs():
    with db(DB_APUNTES) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bloques (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titulo_etiquetas TEXT,
                contenido_bloque TEXT
            );
        """)

    with db(DB_TESTS) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titulo TEXT,
                pdf_examen_nombre TEXT,
                pdf_respuestas_nombre TEXT
            );
        """)
        for col in ("pdf_examen_nombre", "pdf_respuestas_nombre"):
            try:
                conn.execute(f"ALTER TABLE tests ADD COLUMN {col} TEXT;")
            except sqlite3.OperationalError:
                pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS preguntas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER,
                numero INTEGER,
                enunciado TEXT,
                opcion_a TEXT,
                opcion_b TEXT,
                opcion_c TEXT,
                opcion_d TEXT,
                correcta TEXT,
                FOREIGN KEY(test_id) REFERENCES tests(id)
            );
        """)
        try:
            conn.execute("ALTER TABLE preguntas ADD COLUMN bloque TEXT;")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE preguntas ADD COLUMN omitida INTEGER DEFAULT 0;")
        except sqlite3.OperationalError:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_preg_test ON preguntas(test_id);")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS explicaciones (
                pregunta_id INTEGER PRIMARY KEY,
                texto TEXT
            );
        """)

    with db(DB_CURSO_IA) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temas_estudio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bloque TEXT DEFAULT '',
                grupo TEXT DEFAULT '',
                nombre_tema TEXT UNIQUE,
                estado TEXT DEFAULT 'Mejorar',
                aciertos_seguidoss INTEGER DEFAULT 0,
                prompt_personalizado TEXT DEFAULT '',
                pregunta_actual TEXT DEFAULT '',
                feedback_actual TEXT DEFAULT '',
                opcion_correcta TEXT DEFAULT '',
                enunciado_actual TEXT DEFAULT '',
                opciones_actual TEXT DEFAULT '',
                examen_json TEXT DEFAULT ''
            );
        """)
        for col, col_def in [
            ("bloque", "TEXT DEFAULT ''"),
            ("grupo", "TEXT DEFAULT ''"),
            ("prompt_personalizado", "TEXT DEFAULT ''"),
            ("pregunta_actual", "TEXT DEFAULT ''"),
            ("feedback_actual", "TEXT DEFAULT ''"),
            ("opcion_correcta", "TEXT DEFAULT ''"),
            ("enunciado_actual", "TEXT DEFAULT ''"),
            ("opciones_actual", "TEXT DEFAULT ''"),
            ("examen_json", "TEXT DEFAULT ''"),
            ("descripcion", "TEXT DEFAULT ''"),
            ("temario_id", "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE temas_estudio ADD COLUMN {col} {col_def};")
            except sqlite3.OperationalError:
                pass


    with db(DB_TEMARIO) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temario (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bloque TEXT DEFAULT '',
                grupo TEXT DEFAULT '',
                titulo TEXT DEFAULT '',
                descripcion TEXT DEFAULT '',
                descripcion_anterior TEXT DEFAULT '',
                estado TEXT DEFAULT 'Mejorar',
                aciertos INTEGER DEFAULT 0,
                situacion TEXT DEFAULT 'Nuevo',
                linea_original TEXT DEFAULT '',
                orden INTEGER DEFAULT 0
            );
        """)
    return True


inicializar_dbs()


_RE_PIE_TAI = re.compile(r"\s*\b\d{4}\s*[-–—]\s*TAI(?:\s*[-–—]\s*[A-Z]{1,3}\b)?")


@st.cache_resource(show_spinner=False)
def limpiar_pies_bd():
    """Una vez por arranque: quita de enunciados y opciones ya guardados el pie de página del PDF
    ('2019 – TAI - LI') que se coló al extraer las preguntas."""
    try:
        with db(DB_TESTS) as conn:
            filas = conn.execute(
                "SELECT id, enunciado, opcion_a, opcion_b, opcion_c, opcion_d FROM preguntas"
            ).fetchall()
            for pid, *campos in filas:
                nuevos = [_RE_PIE_TAI.sub("", c).strip() if c else c for c in campos]
                if nuevos != list(campos):
                    conn.execute(
                        "UPDATE preguntas SET enunciado=?, opcion_a=?, opcion_b=?, opcion_c=?, opcion_d=? WHERE id=?",
                        (*nuevos, pid),
                    )
    except Exception:
        pass
    return True


limpiar_pies_bd()


# ==========================================
# CACHÉS Y BÚSQUEDA
# ==========================================
def _sin_acentos(txt: str) -> str:
    """Minúsculas y sin tildes: 'Transporte', 'TRANSPORTE' y 'tránsporte' se buscan igual."""
    txt = unicodedata.normalize("NFD", (txt or "").lower())
    return "".join(c for c in txt if unicodedata.category(c) != "Mn")


@st.cache_data(ttl=60, show_spinner=False)
def cargar_apuntes():
    with db(DB_APUNTES) as conn:
        filas = conn.execute(
            "SELECT id, titulo_etiquetas, contenido_bloque FROM bloques"
        ).fetchall()
    resultado = []
    for bid, etq, cont in filas:
        etq = etq or ""
        cont = cont or ""
        t_low = _sin_acentos(etq)
        resultado.append(
            (bid, etq, cont, t_low, frozenset(re.findall(r"\w+", t_low)), _sin_acentos(cont))
        )
    return resultado


def buscar_apuntes(busq1: str, busq2: str):
    b1 = _sin_acentos(busq1).strip()
    palabras2 = _sin_acentos(busq2).split()
    salida = []
    for bid, etq, cont, t_low, palabras, c_low in cargar_apuntes():
        # campo 1: título exacto, palabra del título, o palabra que empieza así
        pasa1 = (not b1) or b1 == t_low or b1 in palabras or any(p.startswith(b1) for p in palabras)
        # campo 2: TODAS las palabras escritas deben estar en el contenido (en cualquier orden)
        pasa2 = all(p in c_low for p in palabras2)
        if pasa1 and pasa2:
            salida.append((bid, etq, cont))
    return salida


@st.cache_data(show_spinner=False)
def parsear_tabla(contenido_tabla: str) -> pd.DataFrame:
    df = pd.read_csv(
        io.StringIO(contenido_tabla.strip()),
        sep="\t",
        header=None,
        on_bad_lines="skip",
        engine="python",
    )
    df.columns = [str(col) for col in df.columns]
    return df


def cargar_glosario_externo():
    glosario = set()
    ruta = Path("glosario.txt")
    if ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                termino = linea.strip().lower()
                if termino and not termino.startswith("#"):
                    glosario.add(termino)
    return glosario


@st.cache_resource(show_spinner=False)
def patron_glosario(terminos: frozenset):
    if not terminos:
        return None
    alt = "|".join(re.escape(t) for t in sorted(terminos, key=len, reverse=True))
    return re.compile(rf"\b({alt})\b")


if "glosario_externo" not in st.session_state:
    st.session_state.glosario_externo = cargar_glosario_externo()


# ==========================================
# IA (Gemini) + TRABAJOS EN SEGUNDO PLANO
# ==========================================
def _obtener_api_key():
    try:
        clave = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        clave = ""
    return clave or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


class LimitadorRPM:
    """Espera lo necesario para no pasar de N peticiones por minuto (compartido entre hilos)."""

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.lock = threading.Lock()
        self.marcas = []

    def esperar(self):
        while True:
            with self.lock:
                ahora = time.monotonic()
                self.marcas = [t for t in self.marcas if ahora - t < 60]
                if len(self.marcas) < self.rpm:
                    self.marcas.append(ahora)
                    return
                espera = 60 - (ahora - self.marcas[0]) + 0.2
            time.sleep(max(espera, 0.2))


@st.cache_resource
def get_limitador():
    return LimitadorRPM(GEMINI_RPM)


@st.cache_resource
def get_client():
    clave = _obtener_api_key()
    if not clave:
        raise RuntimeError("Falta GOOGLE_API_KEY en .streamlit/secrets.toml")
    cliente = OpenAI(base_url=GEMINI_BASE_URL, api_key=clave)
    try:
        cliente._limitador = get_limitador()
    except Exception:
        pass
    return cliente


def _es_429(e) -> bool:
    msg = str(e).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg or "rate limit" in msg


def _cuota_irrecuperable(e) -> bool:
    """Cuota diaria agotada o modelo sin cuota en tu plan: esperar un minuto no sirve."""
    msg = str(e).lower()
    return "perday" in msg or "per day" in msg or "limit: 0" in msg


def _segundos_reintento(e, por_defecto: float) -> float:
    m = re.search(r"retry in ([\d\.]+)\s*s", str(e), flags=re.IGNORECASE) or \
        re.search(r"retrydelay\W+([\d\.]+)\s*s", str(e), flags=re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 1, 70)
    return por_defecto


def _msg_error_gemini(e) -> str:
    base = f"Error de conexión con Gemini: {e}"
    if not _es_429(e):
        return base
    if _cuota_irrecuperable(e):
        return (base + " — Cuota diaria agotada (o este modelo no está en tu plan gratuito). Esperar no sirve: "
                "cambia MODELO_IA, prueba mañana o activa la facturación en Google AI Studio.")
    return (base + " — Has superado el límite de peticiones por minuto. Espera un minuto y vuelve a intentarlo "
            "(o baja GEMINI_RPM).")


@st.cache_resource
def get_executor_tutor():
    return ThreadPoolExecutor(max_workers=1)


@st.cache_resource
def get_executor_test():
    return ThreadPoolExecutor(max_workers=1)


def _crear_completion(client, **kwargs):
    """chat.completions.create con reasoning_effort y reintentos ante límite de peticiones (429)."""
    kwargs["max_tokens"] = max(int(kwargs.get("max_tokens", 64)), 64)  # el razonamiento también cuenta como tokens
    usar_reasoning = bool(GEMINI_REASONING)
    limitador = getattr(client, "_limitador", None)
    for intento in range(GEMINI_REINTENTOS_429 + 1):
        extra = {"reasoning_effort": GEMINI_REASONING} if usar_reasoning else {}
        if limitador is not None:
            limitador.esperar()
        try:
            return client.chat.completions.create(**kwargs, **extra)
        except Exception as e:
            msg = str(e).lower()
            if usar_reasoning and ("reasoning" in msg or "thinking" in msg) and not _es_429(e):
                usar_reasoning = False  # el modelo no admite ese nivel: reintenta sin él
                continue
            if _es_429(e) and not _cuota_irrecuperable(e) and intento < GEMINI_REINTENTOS_429:
                time.sleep(_segundos_reintento(e, 8 * (intento + 1)))
                continue
            raise


if not _obtener_api_key():
    st.error("Falta la clave de Gemini. Crea `.streamlit/secrets.toml` junto a este script con la línea "
             "`GOOGLE_API_KEY = \"tu_clave\"` y vuelve a lanzar la app.")
    st.stop()


class Job:
    def __init__(self, tipo, tema_id=0, aplicar=False, subtipo="", p_id=None, marca=""):
        self._es_job = True
        self.tipo = tipo
        self.subtipo = subtipo
        self.tema_id = tema_id
        self.p_id = p_id
        self.marca = marca
        self.aplicar = aplicar
        self.lock = threading.Lock()
        self.listo = False
        self.done = False
        self.error = None
        self.result = None
        self.texto = ""
        self.progreso = 0
        self.total = 0
        self.explicaciones = {}


def es_job(obj):
    return getattr(obj, "_es_job", False)


def job_activo(clave):
    j = st.session_state.get(clave)
    return j if (es_job(j) and not j.done) else None


def hay_jobs(prefijos) -> bool:
    """True si hay algún job en marcha cuyo nombre empiece por alguno de los prefijos."""
    return any(
        es_job(v) and not v.done
        for k, v in list(st.session_state.items())
        if isinstance(k, str) and k.startswith(prefijos)
    )


def llamar_ia_con_reintentos(prompt: str, max_tokens: int = 300, max_reintentos: int = 1, client=None,
                             temperature: float = 0.2, stop=None):
    client = client or get_client()
    for intento in range(max_reintentos):
        try:
            kwargs = dict(
                model=MODELO_IA,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            if stop:
                kwargs["stop"] = stop
            response = _crear_completion(client, **kwargs)
            return response.choices[0].message.content or "", None
        except Exception as e:
            if intento < max_reintentos - 1:
                continue
            return None, _msg_error_gemini(e)
    return None, "Error desconocido"


def stream_ia(prompt: str, max_tokens: int = 130, client=None):
    client = client or get_client()
    stream = _crear_completion(
        client,
        model=MODELO_IA,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def stream_ia_mensajes(mensajes, max_tokens: int = 700, client=None):
    """Como stream_ia, pero con una conversación completa (system + turnos user/assistant)."""
    client = client or get_client()
    stream = _crear_completion(
        client,
        model=MODELO_IA,
        messages=mensajes,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def rerun_local():
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def formatear_pregunta(enunciado, a, b, c, d):
    lineas = [enunciado]
    for letra, op in zip("ABCD", (a, b, c, d)):
        if op:
            lineas.append(f"{letra}) {op}")
    return "\n".join(lineas)


def cargar_explicacion(pregunta_id):
    with db(DB_TESTS) as conn:
        fila = conn.execute(
            "SELECT texto FROM explicaciones WHERE pregunta_id=?", (pregunta_id,)
        ).fetchone()
    return fila[0] if fila else None


def guardar_explicacion(pregunta_id, texto):
    with db(DB_TESTS) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO explicaciones (pregunta_id, texto) VALUES (?, ?)",
            (pregunta_id, texto),
        )


def borrar_explicacion(pregunta_id):
    with db(DB_TESTS) as conn:
        conn.execute("DELETE FROM explicaciones WHERE pregunta_id=?", (pregunta_id,))


def cargar_examen(tema_id):
    with db(DB_CURSO_IA) as conn:
        row = conn.execute("SELECT examen_json FROM temas_estudio WHERE id=?", (tema_id,)).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except Exception:
            return None
    return None


def guardar_examen(tema_id, examen):
    txt = json.dumps(examen, ensure_ascii=False) if examen else ""
    with db(DB_CURSO_IA) as conn:
        conn.execute("UPDATE temas_estudio SET examen_json=? WHERE id=?", (txt, tema_id))


def guardar_feedback_tema(tema_id, feedback):
    with db(DB_CURSO_IA) as conn:
        conn.execute(
            "UPDATE temas_estudio SET feedback_actual=? WHERE id=?", (feedback, tema_id)
        )


def preparar_prompt_tutor(tema_id, evitar: str = "", extra_instr: str = "") -> str:
    with db(DB_CURSO_IA) as conn:
        row = conn.execute("SELECT nombre_tema, prompt_personalizado, descripcion FROM temas_estudio WHERE id=?", (tema_id,)).fetchone()

    tema_nombre = row[0] if row else ""
    desc_tema = (row[2] or "") if row else ""
    linea_tema = f"{tema_nombre}: {desc_tema}" if desc_tema else tema_nombre
    prompt_base = row[1] if row and row[1] else f"Eres profesor de oposiciones TAI. Redacta una pregunta de examen sobre: {linea_tema}."

    with db(DB_TESTS) as conn:
        insp = conn.execute(
            "SELECT enunciado FROM preguntas WHERE enunciado LIKE ? LIMIT 1",
            (f"%{tema_nombre[:12]}%",),
        ).fetchall()
    texto_insp = "\n".join(p[0] for p in insp) if insp else "Estilo examen oficial TAI C1."
    extra = f"\nEvita repetir o parecerte a estas preguntas ya generadas:\n{evitar}\n" if evitar else ""
    if extra_instr:
        extra += f"\n{extra_instr}\n"

    return f"""{prompt_base}
Referencia: {texto_insp}
{extra}
ESTILO DE EXAMEN TAI (obligatorio):
- Pregunta DIRECTA y corta (una o dos líneas) sobre un dato concreto: qué hace un comando, qué puerto, protocolo, capa o directorio corresponde, para qué sirve algo, diferencia entre dos conceptos básicos.
- Nada de razones de diseño, historia ni justificaciones de un estándar; evita preguntas tipo «¿cuál es la razón técnica de…?».
- No escribas títulos, encabezados ni negritas: empieza directamente por la pregunta.
- Las 4 opciones son breves, de longitud parecida y plausibles; solo una es claramente correcta.

Responde ÚNICAMENTE en el formato del ejemplo, sin texto adicional ni explicaciones.
La primera línea es un enunciado concreto y específico del tema (no genérico).
Escribe UNA SOLA pregunta con EXACTAMENTE 4 opciones (A, B, C y D), nunca una opción E.
Las opciones deben ser breves (una línea cada una) y solo una es correcta.

EJEMPLO DE FORMATO (con otro tema, no lo copies):
¿Qué puerto utiliza por defecto el protocolo HTTPS?
A) 80
B) 443
C) 21
D) 25
CORRECTA: B

Ahora redacta tu pregunta sobre el tema indicado con exactamente ese formato y termina justo después de la línea CORRECTA."""


def parsear_pregunta(txt: str):
    if not txt:
        return None
    t = re.sub(r"\[enunciado.*?\]", "", txt, flags=re.IGNORECASE)
    t = t.replace("**", "").replace("`", "").strip()

    m = re.search(
        r"(?:CORRECTA|Respuesta correcta|Respuesta)\s*:\s*\[?\s*(?:Opci[oó]n\s*)?([A-Ea-e])\b", t
    )
    if not m:
        return None
    letra_correcta = m.group(1).upper()
    bloque = t[: m.start()].strip()

    bloque = re.sub(
        r"(?m)^\s*([A-Ea-e])[\.\)]\s*", lambda mm: f"{mm.group(1).upper()}) ", bloque
    )
    bloque = re.sub(r"[ \t]+([B-E]\))", r"\n\1", bloque)

    ma = re.search(r"(?:^|\s)A\)", bloque)
    if not ma:
        return None
    enunciado = bloque[: ma.start()].strip()
    cuerpo = bloque[ma.start():].strip()

    partes = [x.strip() for x in re.split(r"\n\s*\n", enunciado) if x.strip()]
    if len(partes) > 1:
        cab = partes[0]
        if "?" not in cab and len(cab) <= 120 and re.search(r":\s*\S", cab):
            partes = partes[1:]  # encabezado con el título del tema, no forma parte de la pregunta
        enunciado = " ".join(partes)

    opciones = re.findall(r"(?m)^([A-E])\)\s*(.+?)\s*$", cuerpo)
    letras = [l for l, _ in opciones]
    if letras == list("ABCDE") and letra_correcta != "E":
        opciones = opciones[:4]
        letras = list("ABCD")
    if letras != list("ABCD") or letra_correcta not in "ABCD":
        return None

    opciones_txt = "\n".join(f"{l}) {o}" for l, o in opciones)
    if len(enunciado) < 10 or len(opciones_txt) < 10:
        return None

    return enunciado, opciones_txt, letra_correcta


def generar_pregunta(prompt: str, client=None, intentos: int = 3, temperature: float = 0.85, log=None):
    for intento in range(intentos):
        txt, err = llamar_ia_con_reintentos(
            prompt,
            max_tokens=TOKENS_PREGUNTA,
            client=client,
            temperature=temperature,
            stop=(["Explicación", "Justificación", "Explicacion", "\n¿", "\nPregunta"]
                  if intento == 0 else ["Explicación", "Justificación", "Explicacion"]),
        )
        if err:
            return None, err
        parsed = parsear_pregunta(txt)
        if parsed is not None:
            return parsed, None

        tmp = parsear_pregunta((txt or "").rstrip() + "\nCORRECTA: A")
        if tmp is not None:
            enun, ops, _ = tmp
            txt2, err2 = llamar_ia_con_reintentos(
                f"Pregunta:\n{enun}\n{ops}\n\nResponde ÚNICAMENTE con la letra (A, B, C o D) de la opción correcta.",
                max_tokens=6, client=client, temperature=0.0,
            )
            m = re.search(r"\b([A-D])\b", (txt2 or "").upper())
            if m and not err2:
                return (enun, ops, m.group(1)), None

        if log is not None:
            log.append(txt)
    return None, "Error al formatear la pregunta. Pulsa de nuevo."


def verificar_preguntas_lote(preguntas, client=None):
    """Segunda opinión en UNA sola llamada: la IA resuelve todas las preguntas sin ver la respuesta marcada.
    Devuelve los índices de las preguntas dudosas (respuesta distinta, o X = ninguna/varias/dato inventado)."""
    if not preguntas:
        return []
    bloques = "\n\n".join(
        f"[{i + 1}]\n{p['enunciado']}\n{p['opciones']}" for i, p in enumerate(preguntas)
    )
    prompt = (
        "Eres un examinador experto en informática (nivel TAI). Resuelve por tu cuenta cada pregunta tipo test.\n"
        "Responde ÚNICAMENTE con una línea por pregunta con este formato: 1:C (número y letra de la opción correcta). "
        "Si ninguna opción es correcta, si hay más de una correcta o si alguna opción contiene un dato inventado "
        "o inexistente, pon X en lugar de la letra.\n\n" + bloques
    )
    txt, err = llamar_ia_con_reintentos(
        prompt, max_tokens=40 + 10 * len(preguntas), client=client, temperature=0.0
    )
    if err or not txt:
        return []  # no se pudo verificar: se aceptan
    resp = dict(re.findall(r"(\d+)\s*[:\-\.\)]\s*([A-Da-dXx])\b", txt))
    dudosas = []
    for i, p in enumerate(preguntas):
        letra = resp.get(str(i + 1), "").upper()
        if letra and letra != p["correcta"].upper():
            dudosas.append((i, letra))
    return dudosas


def _log_debug(titulo: str, texto: str):
    try:
        with open("debug_examen.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%H:%M:%S')} · {titulo} =====\n{texto}\n")
    except Exception:
        pass


def _texto_opcion(opciones: str, letra: str) -> str:
    for linea in opciones.split("\n"):
        m = re.match(r"\s*([A-Da-d])\)\s*(.*)", linea)
        if m and m.group(1).upper() == letra.upper():
            return m.group(2).strip()
    return ""


def _norm(txt: str) -> str:
    return re.sub(r"\W+", " ", txt.lower()).strip()


ANGULOS_GENERICOS = [
    "concepto y definición", "componentes y arquitectura", "protocolos, puertos y estándares",
    "seguridad y cifrado", "ventajas e inconvenientes", "comparación con alternativas",
    "normativa y estándares aplicables", "configuración y administración", "casos de uso típicos",
    "errores o problemas habituales", "versiones y evolución", "terminología y siglas",
]


def _info_tema(tema_id):
    with db(DB_CURSO_IA) as conn:
        row = conn.execute("SELECT nombre_tema, grupo, descripcion FROM temas_estudio WHERE id=?", (tema_id,)).fetchone()
    return (row[0], row[1], row[2] or "") if row else ("", "", "")


def _generar_subtemas(tema_id, n, client):
    """Pide a la IA n subtemas distintos; si falla, completa con ángulos genéricos."""
    tema, grupo, desc = _info_tema(tema_id)
    contenido = f" Basa los subtemas en estos contenidos concretos del temario: {desc}." if desc else ""
    prompt = (
        f"Eres profesor de oposiciones TAI. Para el tema «{tema}» ({grupo}) lista exactamente {n} "
        "subtemas o aspectos técnicos DISTINTOS y concretos sobre los que se podría preguntar en un examen tipo test."
        f"{contenido} "
        "Una línea por subtema, sin numeración ni explicaciones, máximo 8 palabras cada uno."
    )
    subtemas = []
    txt, err = llamar_ia_con_reintentos(prompt, max_tokens=150, client=client, temperature=0.5)
    if txt and not err:
        for linea in txt.split("\n"):
            limpia = re.sub(r"^[\-\*•\d\.\)\s]+", "", linea).strip(" .:")
            if 4 <= len(limpia) <= 80 and limpia.lower() not in [x.lower() for x in subtemas]:
                subtemas.append(limpia)
    subtemas = subtemas[:n]
    for a in ANGULOS_GENERICOS:
        if len(subtemas) >= n + 6:
            break
        subtemas.append(f"{a} de {tema}")
    return subtemas


def _muy_parecida(enun: str, previos) -> bool:
    a = {w for w in _norm(enun).split() if len(w) > 3}
    if not a:
        return False
    for p in previos:
        b = {w for w in _norm(p).split() if len(w) > 3}
        if b and len(a & b) / len(a | b) >= 0.8:
            return True
    return False


def trabajo_examen(job, tema_id, client):
    preguntas = []
    f_formato = 0
    f_repetidas = 0
    f_dudosas = 0
    job.total = NUM_PREGUNTAS_EXAMEN
    try:
        pendientes = list(_generar_subtemas(tema_id, NUM_PREGUNTAS_EXAMEN, client))
        desde = 0  # a partir de aquí las preguntas aún no están verificadas
        for ronda in range(3):
            while len(preguntas) < NUM_PREGUNTAS_EXAMEN and pendientes:
                subtema = pendientes.pop(0)
                evitar = "\n".join(f"- {p['enunciado']}" for p in preguntas)
                extra = (
                    f"Esta pregunta debe tratar EXCLUSIVAMENTE sobre este aspecto concreto del tema: «{subtema}».\n"
                    "Basa la pregunta en hechos técnicos ampliamente conocidos y verificables; "
                    "si no estás seguro de un dato, no lo uses."
                )
                prompt = preparar_prompt_tutor(tema_id, evitar=evitar, extra_instr=extra)
                crudos = []
                parsed, err = generar_pregunta(prompt, client, intentos=2, log=crudos)
                if parsed is None:
                    if err and err.startswith("Error de conexión"):
                        job.error = err
                        break
                    f_formato += 1
                    for c in crudos:
                        _log_debug(f"FORMATO INCORRECTO ({subtema})", c)
                    continue
                enun, ops, corr = parsed
                if _muy_parecida(enun, [p["enunciado"] for p in preguntas]):
                    f_repetidas += 1
                    _log_debug(f"REPETIDA ({subtema})", f"{enun}\n{ops}\nCORRECTA: {corr}")
                    continue
                preguntas.append({"enunciado": enun, "opciones": ops, "correcta": corr})
                job.progreso = len(preguntas)

            if job.error:
                break

            nuevas = preguntas[desde:]
            dudosas = verificar_preguntas_lote(nuevas, client)
            if not dudosas:
                break
            for i, letra in sorted(dudosas, reverse=True):
                p = preguntas.pop(desde + i)
                f_dudosas += 1
                _log_debug(f"DUDOSA (marcada {p['correcta']}, verificador {letra})", f"{p['enunciado']}\n{p['opciones']}")
            job.progreso = len(preguntas)
            desde = len(preguntas)
            if ronda < 2:
                pendientes = list(_generar_subtemas(tema_id, len(dudosas) + 1, client))

        if not job.error:
            if len(preguntas) < NUM_PREGUNTAS_EXAMEN:
                job.error = (
                    f"Solo se pudieron generar {len(preguntas)} de {NUM_PREGUNTAS_EXAMEN} preguntas "
                    f"(formato incorrecto: {f_formato}, repetidas: {f_repetidas}, descartadas por dudosas: {f_dudosas}). "
                    "Vuelve a intentarlo. Detalles en debug_examen.log."
                )
            else:
                guardar_examen(tema_id, {
                    "preguntas": preguntas,
                    "respuestas": [None] * len(preguntas),
                    "idx": 0,
                    "terminado": False,
                    "nota": None,
                })
    except Exception as e:
        job.error = f"Error generando el examen: {e}"
    finally:
        job.done = True


def trabajo_corregir_examen(job, tema_id, examen, client):
    try:
        for i, p in enumerate(examen["preguntas"]):
            resp = examen["respuestas"][i]
            if resp == p["correcta"]:
                continue
            job.explicaciones[i] = ""
            txt_ok = _texto_opcion(p["opciones"], p["correcta"])
            txt_al = _texto_opcion(p["opciones"], resp) if resp else ""
            prompt = (
                f"Pregunta:\n{p['enunciado']}\n{p['opciones']}\n\n"
                f"La opción correcta según la pregunta es la {p['correcta']}) {txt_ok}. "
                f"El alumno marcó {('la ' + resp + ') ' + txt_al) if resp else 'ninguna'}.\n"
                f"Explica en un máximo de 3 líneas cortas por qué la {p['correcta']}) {txt_ok} es la correcta y por qué falla la opción del alumno. "
                "No repitas las opciones y no te contradigas. Si la opción marcada como correcta es técnicamente falsa o la pregunta es "
                "ambigua, di SOLO una línea que empiece por «⚠️ Posible error en la pregunta:» y explica el motivo."
            )
            for trozo in stream_ia(prompt, TOKENS_EVALUACION, client):
                job.explicaciones[i] += trozo
    except Exception as e:
        job.error = _msg_error_gemini(e)
    try:
        actual = cargar_examen(tema_id)
        if actual:
            actual["explicaciones"] = {str(k): v for k, v in job.explicaciones.items()}
            guardar_examen(tema_id, actual)
    except Exception as e:
        job.error = f"Error guardando la corrección: {e}"
    finally:
        job.done = True


def trabajo_ia_test(job, prompt, max_tokens, client, cachear_id=None):
    try:
        for trozo in stream_ia(prompt, max_tokens, client):
            job.texto += trozo
        if cachear_id and job.texto:
            guardar_explicacion(cachear_id, job.texto)
    except Exception as e:
        job.error = _msg_error_gemini(e)
    finally:
        job.done = True


def trabajo_ia_seguimiento(job, mensajes, max_tokens, client):
    try:
        for trozo in stream_ia_mensajes(mensajes, max_tokens, client):
            job.texto += trozo
    except Exception as e:
        job.error = _msg_error_gemini(e)
    finally:
        job.done = True


def recoger_jobs_tutor():
    hecho = False
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(PREF_JOBS_TUTOR):
            job = st.session_state.get(k)
            if es_job(job) and job.done:
                del st.session_state[k]
                hecho = True
                if job.error:
                    st.session_state[f"err_{job.tema_id}"] = job.error
                elif job.tipo == "exam":
                    st.toast("✅ Examen listo")
    return hecho


def recoger_jobs_test():
    hecho = False
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(PREF_JOBS_TEST):
            job = st.session_state.get(k)
            if es_job(job) and job.done:
                del st.session_state[k]
                hecho = True
                if job.subtipo == "seguimiento":
                    hist = st.session_state.setdefault(f"chat_ia_{job.p_id}", [])
                    if job.error or not job.texto:
                        if hist and hist[-1]["role"] == "user":
                            hist.pop()  # la pregunta quedó sin respuesta
                        st.session_state[f"chat_err_{job.p_id}"] = job.error or "La IA no devolvió respuesta. Inténtalo de nuevo."
                    else:
                        hist.append({"role": "assistant", "content": job.texto})
                    continue
                if job.error:
                    st.session_state[f"ia_{job.p_id}"] = ("error", job.error)
                elif job.texto:
                    st.session_state[f"ia_{job.p_id}"] = (job.subtipo, job.texto)
    return hecho


# ---- Palabras partidas por el PDF ("pr opiedades"): se reparan con el vocabulario de las propias preguntas ----
_RE_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_PALABRAS_CORTAS = {
    "a", "al", "de", "del", "el", "en", "la", "las", "los", "un", "una", "y", "o", "u", "e", "se", "su", "no",
    "si", "es", "lo", "le", "que", "por", "con", "sin", "para", "como", "más", "mas", "ni", "ya", "ha", "he",
    "me", "te", "mi", "tu", "son", "ser", "fue", "uno", "dos", "sus", "hay", "va", "ve", "sí", "sé", "qué",
}


def construir_vocabulario(textos):
    from collections import Counter
    vocab = Counter()
    for t in textos:
        vocab.update(w.lower() for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", t or ""))
    return vocab


def _debe_unirse(a: str, b: str, vocab) -> bool:
    union = (a + b).lower()
    n_union = vocab.get(union, 0)
    if n_union == 0 or len(union) < 4:
        return False
    na, nb = vocab.get(a.lower(), 0), vocab.get(b.lower(), 0)
    a_corta, b_corta = a.lower() in _PALABRAS_CORTAS, b.lower() in _PALABRAS_CORTAS
    if a_corta and b_corta:
        return False
    if a_corta:
        return nb <= 2
    if b_corta:
        return na <= 2
    return na < n_union or nb < n_union


def reparar_palabras_partidas(texto: str, vocab) -> str:
    """Une «pr opiedades» → «propiedades» solo si la palabra unida ya aparece en el vocabulario y alguno de
    los dos trozos es un fragmento suelto (aparece menos veces que la palabra entera)."""
    if not texto or " " not in texto:
        return texto
    for _ in range(3):  # por si una palabra quedó partida en 3 trozos
        toks = list(_RE_PALABRA.finditer(texto))
        salida, pos, i, cambio = [], 0, 0, False
        while i < len(toks):
            m = toks[i]
            if (
                i + 1 < len(toks)
                and texto[m.end():toks[i + 1].start()] == " "
                and _debe_unirse(m.group(), toks[i + 1].group(), vocab)
            ):
                salida.append(texto[pos:m.end()])       # hasta el final del primer trozo
                pos = toks[i + 1].start()               # y se salta el espacio
                cambio = True
                i += 2
            else:
                i += 1
        salida.append(texto[pos:])
        texto = "".join(salida)
        if not cambio:
            break
    return texto


def extraer_texto_pdf(uploaded_file) -> str:
    lector = pypdf.PdfReader(uploaded_file)
    return "\n\f\n".join([pagina.extract_text() or "" for pagina in lector.pages])  # \f = salto de página


# ---- Bloques de un examen: Primera Parte, Supuesto N, Reserva (de cada uno) ----
_RE_HDR_PRIMERA = re.compile(r"^\s*primera\s+parte\b", re.IGNORECASE)
_RE_HDR_SUPUESTO = re.compile(
    r"^\s*supuesto(?:\s+pr[aá]ctico)?\s*(?:n[º°o\.]*\s*)?([IVXivx]+|\d+)\b", re.IGNORECASE
)
_RE_HDR_RESERVA = re.compile(
    r"^\s*(?:preguntas?\s+de\s+)?reservas?\s*:?\s*(?:\(([^)]*)\))?\s*(?:CUERPO DE.*)?$", re.IGNORECASE
)


def _num_romano_o_arabigo(txt: str) -> int:
    if txt.isdigit():
        return int(txt)
    vals = {"I": 1, "V": 5, "X": 10}
    total, prev = 0, 0
    for ch in reversed(txt.upper()):
        v = vals.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _nombre_bloque(txt: str) -> str:
    """Normaliza «Primera Parte», «Supuesto I», «Supuesto Nº 2»… ('' si no es un bloque)."""
    if _RE_HDR_PRIMERA.match(txt):
        return "Primera Parte"
    m = _RE_HDR_SUPUESTO.match(txt)
    return f"Supuesto {_num_romano_o_arabigo(m.group(1))}" if m else ""


def _actualizar_bloque(linea: str, bloque: str, padre: str):
    """Devuelve (es_encabezado, bloque, padre). 'Reserva' pertenece al bloque indicado entre
    paréntesis o, si no lo hay, al que la precede."""
    nombre = _nombre_bloque(linea)
    if nombre:
        return True, nombre, nombre
    m = _RE_HDR_RESERVA.match(linea)
    if m:
        padre_res = _nombre_bloque(m.group(1) or "") or padre
        return True, f"Reserva {padre_res}".strip(), padre_res
    return False, bloque, padre


_RE_BASURA_PAGINA = re.compile(
    r"(CUERPO DE|P[áa]gina\s+\d+|PROMOCI[ÓO]N INTERNA|CUESTIONARIO|\b\d{4}\s*[-–—]\s*TAI\b)"
)


def _partir_opciones(bloque: str):
    """Separa el enunciado y las opciones a)-d), incluso con varias opciones en la misma línea.
    Devuelve (enunciado, {letra: texto}) o None."""
    m = re.search(r"^[ \t]*a[\.\)]\s+", bloque, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        # el PDF a veces deja «a)» pegada al final del enunciado («…redes de datos? a) BPDUs…»): se acepta
        # una «a)» a mitad de línea, la última que tenga una «b)» detrás (así no se confunde con un «a)» del enunciado)
        for c in reversed(list(re.finditer(r"(?<=\s)a\)\s+(?=\S)", bloque, flags=re.IGNORECASE))):
            if re.search(r"(?<=\s)b\)\s+", bloque[c.end():], flags=re.IGNORECASE):
                m = c
                break
        if not m:
            return None
    enunciado = bloque[: m.start()].strip()
    marcas = [("a", m.start(), m.end())]
    pos = m.end()
    for letra in "bcd":
        # se toma el marcador más cercano, esté al principio de línea o dentro de ella. Antes se preferían
        # los de inicio de línea aunque estuvieran mucho más lejos (en la pregunta siguiente), y por eso
        # «a) Un router. b) Un hub.» se tragaba la pregunta 100 entera.
        cands = [
            c for c in (
                re.search(rf"^[ \t]*{letra}[\.\)]\s+", bloque[pos:], flags=re.IGNORECASE | re.MULTILINE),
                re.search(rf"(?<=\s){letra}\)\s+", bloque[pos:], flags=re.IGNORECASE),
            ) if c
        ]
        if not cands:
            break
        mm = min(cands, key=lambda c: c.start())
        marcas.append((letra, pos + mm.start(), pos + mm.end()))
        pos += mm.end()
    opciones = {}
    for k, (letra, _ini, fin_marca) in enumerate(marcas):
        fin_txt = marcas[k + 1][1] if k + 1 < len(marcas) else len(bloque)
        txt = bloque[fin_marca:fin_txt]
        txt = _RE_BASURA_PAGINA.split(txt)[0]          # cabeceras/pies de página pegados
        opciones[letra] = " ".join(txt.split())
    return enunciado, opciones


_RE_NUM_LINEA = re.compile(r"^\s*(\d{1,3})[\.\)]")
_RE_NUM_INTERNO = re.compile(r"(?<=[\.\?:;\)!»\"])[ \t]*(\d{1,3})[\.\)](?=\s*[¿¡A-ZÁÉÍÓÚÜÑ])")


def _separar_preguntas_pegadas(texto: str) -> str:
    """Si el PDF deja el número de la pregunta siguiente en la misma línea que el final de la anterior
    («d) Un switch. 100.¿Cuál…»), le pone un salto de línea delante. Solo corta si el número es
    exactamente el siguiente al de la última pregunta vista, para no confundirlo con un «2.» del texto."""
    salida, ultimo = [], None
    for linea in texto.split("\n"):
        m0 = _RE_NUM_LINEA.match(linea)
        if m0:
            ultimo = int(m0.group(1))
        pos = m0.end() if m0 else 0
        ini, trozos = 0, []
        while ultimo is not None:
            hallado = next(
                (m for m in _RE_NUM_INTERNO.finditer(linea, pos) if int(m.group(1)) == ultimo + 1), None
            )
            if not hallado:
                break
            trozos.append(linea[ini:hallado.start()].rstrip())
            ini = hallado.start(1)
            ultimo = int(hallado.group(1))
            pos = hallado.end()
        trozos.append(linea[ini:])
        salida.extend(trozos)
    return "\n".join(salida)


def _extraer_preguntas_segmento(texto_completo: str):
    texto = re.sub(r"\r\n", "\n", texto_completo).replace("\f", "\n")
    texto = _separar_preguntas_pegadas(texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    patron_pregunta = re.compile(
        r"(?:^|\n)\s*(\d{1,3})[\.\)](?:\s+|(?=[¿¡A-ZÁÉÍÓÚÜÑ]))(.+?)"
        r"(?=(?:\n\s*\d{1,3}[\.\)](?:\s+|[¿¡A-ZÁÉÍÓÚÜÑ]))|\Z)",
        re.DOTALL,
    )
    preguntas = []
    for match in patron_pregunta.finditer(texto):
        numero = int(match.group(1))
        partido = _partir_opciones(match.group(2).strip())
        if not partido:
            continue
        enunciado, opciones = partido
        if "a" in opciones and "b" in opciones and enunciado:
            preguntas.append({
                "numero": numero,
                "enunciado": enunciado,
                "opcion_a": opciones.get("a", ""),
                "opcion_b": opciones.get("b", ""),
                "opcion_c": opciones.get("c", ""),
                "opcion_d": opciones.get("d", ""),
            })
    return preguntas


def extraer_preguntas_del_pdf(texto_completo: str):
    """Extrae las preguntas indicando a qué bloque pertenece cada una (clave 'bloque').

    Hay PDFs donde el título del bloque («PRIMERA PARTE», «SUPUESTO I») es un cuadro en lo alto de la
    página que el lector entrega como ÚLTIMA línea de esa página: en ese caso vale para toda la página."""
    texto = re.sub(r"\r\n", "\n", texto_completo)
    paginas = [pg.split("\n") for pg in texto.split("\f")]
    hay_bloques = any(_actualizar_bloque(l, "", "")[0] for pg in paginas for l in pg)

    inicial = "Primera Parte" if hay_bloques else ""
    bloque = padre = inicial
    segmentos = [[bloque, []]]  # [bloque, líneas]

    for lineas in paginas:
        no_vacias = [i for i, l in enumerate(lineas) if l.strip()]
        if no_vacias:
            i_ult = no_vacias[-1]
            nombre = _nombre_bloque(lineas[i_ult])
            if nombre:  # etiqueta al final de la página: se aplica desde el principio de la página
                if nombre != bloque:
                    bloque = padre = nombre
                    segmentos.append([bloque, []])
                lineas = lineas[:i_ult]
        for linea in lineas:
            es_hdr, nuevo_bloque, padre = _actualizar_bloque(linea, bloque, padre)
            if es_hdr:
                if nuevo_bloque != bloque:  # una cabecera repetida en cada página no parte la pregunta
                    bloque = nuevo_bloque
                    segmentos.append([bloque, []])
                continue
            segmentos[-1][1].append(linea)

    preguntas = []
    for nombre, lineas in segmentos:
        for p in _extraer_preguntas_segmento("\n".join(lineas)):
            p["bloque"] = nombre
            preguntas.append(p)
    return preguntas


def extraer_plantilla_respuestas_pdf(texto_completo: str):
    """Devuelve {(bloque, número): letra}. Sin encabezados, el bloque queda vacío ('')."""
    respuestas = {}
    bloque = padre = ""
    patron = re.compile(r"(?<!\d)(\d{1,3})\s*[\.\)\-–]\s*([A-Da-d])\b")
    for linea in texto_completo.replace("\r\n", "\n").split("\n"):
        es_hdr, bloque, padre = _actualizar_bloque(linea, bloque, padre)
        if es_hdr:
            continue
        for num_str, letra in patron.findall(linea):
            respuestas[(bloque, int(num_str))] = letra.upper()
    return respuestas


def buscar_respuesta(plantilla: dict, bloque: str, numero: int):
    for clave in ((bloque, numero), ("", numero), ("Primera Parte", numero) if not bloque else None):
        if clave and clave in plantilla:
            return plantilla[clave]
    return None


def mostrar_tabla_aggrid(contenido_tabla, key_unico, editable=False):
    try:
        df = parsear_tabla(contenido_tabla)
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            editable=editable,
            wrapText=True,
            autoHeight=True,
            resizable=True,
            sortable=True,
        )
        response = AgGrid(
            df,
            gridOptions=gb.build(),
            use_container_width=True,
            fit_columns_on_grid_load=True,
            theme="streamlit",
            update_mode=GridUpdateMode.MODEL_CHANGED if editable else GridUpdateMode.NO_UPDATE,
            key=f"aggrid_{key_unico}",
        )
        if editable:
            df_mod = pd.DataFrame(response["data"])
            output = io.StringIO()
            df_mod.to_csv(output, sep="\t", index=False, header=False)
            return output.getvalue()
    except Exception as e:
        st.error(f"Error al renderizar la tabla: {e}")
        st.text_area(
            "Contenido:",
            value=contenido_tabla,
            height=220,
            label_visibility="collapsed",
            key=f"fallback_ta_{key_unico}",
        )
    return contenido_tabla


def _con_saltos(txt: str) -> str:
    """Markdown junta las líneas sueltas: se convierten en saltos de línea reales (dos espacios + \\n)."""
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", (txt or "").replace("\r\n", "\n"))


def actualizar_apunte(bid: int, titulo: str, contenido: str):
    with db(DB_APUNTES) as conn:
        conn.execute(
            "UPDATE bloques SET titulo_etiquetas=?, contenido_bloque=? WHERE id=?",
            (titulo.strip(), formatear_apunte(contenido).strip(), bid),
        )
    cargar_apuntes.clear()


def borrar_apunte(bid: int):
    with db(DB_APUNTES) as conn:
        conn.execute("DELETE FROM bloques WHERE id=?", (bid,))
    cargar_apuntes.clear()


# ==========================================
# APUNTES: botón derecho sobre una línea -> negrita (línea entera o lo que va antes de ":")
# ==========================================
_DIR_COMP_APUNTE = Path(__file__).resolve().parent / "componente_apunte"
_HTML_COMP_APUNTE = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent; width: 100%; }
  body { font-family: "Source Sans Pro", "Source Sans 3", sans-serif; font-size: 16px; line-height: 1.55; color: #31333F; }
  #cont { padding: 2px 0; width: 100%; box-sizing: border-box; }
  .linea { padding: 3px 6px; white-space: pre-wrap; word-break: break-word; cursor: context-menu; border-radius: 4px; }
  .linea:hover { background: rgba(128,128,128,.10); }
  #cont { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; table-layout: auto; }
  td { position: relative; border: 1px solid rgba(128,128,128,.30); padding: 6px 8px; vertical-align: top; white-space: pre-wrap; overflow-wrap: break-word; cursor: context-menu; }
  .rs { position: absolute; top: 0; right: -4px; width: 8px; height: 100%; cursor: col-resize; z-index: 3; }
  .rs:hover, .rs.act { background: rgba(255,75,75,.40); }
  td:hover { background: rgba(128,128,128,.10); }
  .cod { background: rgba(128,128,128,.14); border-radius: 8px; padding: 10px 14px; margin: 4px 0;
         font-family: "Source Code Pro", Menlo, Consolas, monospace; font-size: 14px; line-height: 1.6; }
  .cod .linea { padding: 0 2px; border-radius: 3px; }
  .cod .linea:hover { background: rgba(128,128,128,.18); }
  td.celdacod { background: rgba(128,128,128,.14); font-family: "Source Code Pro", Menlo, Consolas, monospace; font-size: 14px; }
  #menu { position: fixed; display: none; z-index: 10; min-width: 230px; padding: 4px; border-radius: 8px;
          background: #fff; border: 1px solid rgba(128,128,128,.45); box-shadow: 0 4px 14px rgba(0,0,0,.25); }
  #menu button { display: block; width: 100%; text-align: left; border: 0; background: transparent; color: #31333F;
                 padding: 8px 12px; font: inherit; font-size: 15px; border-radius: 6px; cursor: pointer; }
  #menu button:hover { background: rgba(255,75,75,.12); }
</style></head>
<body>
<div id="cont"></div>
<div id="menu"></div>
<script>
  const enviar = (tipo, extra) => window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: tipo}, extra || {}), "*");
  let lineas = [], menuAbierto = false, ultimoTexto = null;
  let anchos = null;   // anchos (px) de las columnas fijados con el ratón; null = ajuste automático
  const menu = document.getElementById("menu"), cont = document.getElementById("cont");

  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const fmt = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  const esCod = (t) => /^`[^`]+`$/.test(t.trim());                    // línea en cursiva: `texto`
  const quitaCod = (t) => { const m = t.match(/^(\s*)`([^`]+)`\s*$/); return m ? m[1] + m[2] : t; };

  function posDosPuntos(t) {
    for (let i = 0; i < t.length; i++) {
      if (t[i] !== ":") continue;
      if (t.substr(i + 1, 2) === "//") continue;
      if (t.slice(0, i).trim()) return i;
    }
    return -1;
  }
  function opciones(raw, n) {
    const core = raw.trim(), res = [];
    if (n > 1) {  // varias líneas seleccionadas: todos los formatos
      res.push(["linea", "Negrita en las " + n + " líneas"]);
      res.push(["antes", "Negrita solo antes de «:» en las " + n + " líneas"]);
      res.push(["quitar", "Quitar negrita de las " + n + " líneas"]);
      res.push(["cursiva", "Cursiva en las " + n + " líneas seleccionadas"]);
      res.push(["quitar_cursiva", "Quitar cursiva de las " + n + " líneas"]);
      return res;
    }
    if (!core) return res;
    if (esCod(core)) { res.push(["quitar_cursiva", "Quitar cursiva"]); return res; }
    if (!(core.startsWith("**") && core.endsWith("**") && core.indexOf("**", 2) === core.length - 2))
      res.push(["linea", "Negrita en toda la línea"]);
    const i = posDosPuntos(core);
    if (i > 0 && !/^\*\*[^*]*:\*\*/.test(core)) res.push(["antes", "Negrita solo antes de «:»"]);
    if (raw.indexOf("**") >= 0) res.push(["quitar", "Quitar negrita"]);
    res.push(["cursiva", "Cursiva (fondo gris)"]);
    return res;
  }

  // Líneas [primera, última] cubiertas por la selección del ratón si el clic derecho cae dentro de ella
  function rangoSeleccion(l) {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const lineaDe = (n) => { const e = n && (n.nodeType === 1 ? n : n.parentElement); return e ? e.closest(".linea[data-l]") : null; };
    const r = sel.getRangeAt(0), a = lineaDe(r.startContainer), b = lineaDe(r.endContainer);
    if (!a || !b) return null;
    let lo = +a.dataset.l, hi = +b.dataset.l;
    if (hi > lo && r.endOffset === 0) hi--;   // la selección termina justo al inicio de la línea siguiente
    return (hi > lo && l >= lo && l <= hi) ? [lo, hi] : null;
  }

  // Streamlit a veces deja el iframe (y sus contenedores) más estrechos que la línea: se fuerza al 100 %
  function ajustarAncho() {
    try {
      const f = window.frameElement;
      if (!f) return;
      f.style.width = "100%";
      f.style.minWidth = "100%";
      f.removeAttribute("width");
      let n = f.parentElement;
      for (let k = 0; k < 3 && n; k++, n = n.parentElement) {
        n.style.width = "100%";
        n.style.maxWidth = "100%";
      }
    } catch (e) { /* sin acceso al marco: se queda como esté */ }
  }

  function alto() {
    let h = cont.offsetHeight + 6;
    if (menuAbierto) h = Math.max(h, menu.offsetTop + menu.offsetHeight + 8);
    enviar("streamlit:setFrameHeight", {height: h});
  }
  function aplicarAnchos() {
    const t = cont.querySelector("table");
    if (!t) return;
    const cols = t.querySelectorAll("col");
    if (!anchos) { t.style.tableLayout = ""; cols.forEach((c) => { c.style.width = ""; }); return; }
    t.style.tableLayout = "fixed";   // la última columna queda libre y se lleva el resto del ancho
    cols.forEach((c, i) => { c.style.width = i < cols.length - 1 ? anchos[i] + "px" : ""; });
  }

  // Cambiar el ancho de una columna arrastrando su borde (doble clic en el borde: vuelve al ajuste automático)
  cont.addEventListener("mousedown", (e) => {
    const rs = e.target.closest && e.target.closest(".rs");
    if (!rs || e.button !== 0) return;
    e.preventDefault(); e.stopPropagation();
    const t = cont.querySelector("table");
    if (!t) return;
    if (!anchos) anchos = Array.from(t.querySelectorAll("col")).map((c) => Math.round(c.getBoundingClientRect().width));
    const k = +rs.dataset.k, x0 = e.clientX, w0 = anchos[k];
    rs.classList.add("act");
    document.body.style.userSelect = "none";
    const mover = (ev) => {
      if (ev.buttons === 0) { soltar(); return; }
      anchos[k] = Math.max(30, Math.round(w0 + ev.clientX - x0));
      aplicarAnchos(); alto();
    };
    const soltar = () => {
      document.removeEventListener("mousemove", mover);
      document.removeEventListener("mouseup", soltar);
      document.body.style.userSelect = "";
      rs.classList.remove("act");
    };
    document.addEventListener("mousemove", mover);
    document.addEventListener("mouseup", soltar);
  });
  cont.addEventListener("dblclick", (e) => {
    if (e.target.closest && e.target.closest(".rs")) { anchos = null; aplicarAnchos(); alto(); }
  });

  function cerrarMenu() { if (menuAbierto) { menuAbierto = false; menu.style.display = "none"; alto(); } }

  function pintar(texto) {
    lineas = (texto || "").split("\n");
    const conTabs = (texto || "").indexOf("\t") >= 0;
    let h = "";
    if (conTabs) {
      const maxCols = Math.max(...lineas.map((l) => l.split("\t").length));
      if (anchos && anchos.length !== maxCols) anchos = null;   // la tabla cambió de forma: se vuelve al ajuste automático
      h = "<table><colgroup>" + "<col>".repeat(maxCols) + "</colgroup>";
      lineas.forEach((l, i) => {
        const cs = l.split("\t");
        h += "<tr>" + cs.map((c, j) => {
          // fila con menos columnas que la tabla: su última celda se estira hasta el final (ancho completo)
          const nSpan = (j === cs.length - 1 && cs.length < maxCols) ? maxCols - cs.length + 1 : 1;
          const span = nSpan > 1 ? ` colspan="${nSpan}"` : "";
          const k = j + nSpan - 1;   // última columna que cubre la celda
          const tir = k < maxCols - 1 ? `<div class="rs" data-k="${k}" title="Arrastra para cambiar el ancho (doble clic: ajuste automático)"></div>` : "";
          return esCod(c)
            ? `<td class="celdacod"${span} data-l="${i}" data-c="${j}">${esc(quitaCod(c))}${tir}</td>`
            : `<td${span} data-l="${i}" data-c="${j}">${fmt(c) || "&nbsp;"}${tir}</td>`;
        }).join("") + "</tr>";
      });
      h += "</table>";
    } else {
      let grupo = [];  // líneas en cursiva consecutivas: forman un único bloque gris
      const volcar = () => { if (grupo.length) { h += '<div class="cod">' + grupo.join("") + "</div>"; grupo = []; } };
      lineas.forEach((l, i) => {
        if (esCod(l)) { grupo.push(`<div class="linea" data-l="${i}" data-c="0">${esc(quitaCod(l))}</div>`); return; }
        volcar();
        h += `<div class="linea" data-l="${i}" data-c="0">${fmt(l) || "&nbsp;"}</div>`;
      });
      volcar();
    }
    cont.innerHTML = h;
    aplicarAnchos();
    ajustarAncho();
    cerrarMenu();
    alto();
  }

  document.addEventListener("contextmenu", (e) => {
    const el = e.target.closest("[data-l]");
    if (!el) return;
    e.preventDefault();
    const l = +el.dataset.l, c = +el.dataset.c;
    const raw = (lineas[l] || "").split("\t")[c] || "";
    const rg = el.tagName === "TD" ? null : rangoSeleccion(l);
    const ops = opciones(raw, rg ? rg[1] - rg[0] + 1 : 1);
    if (el.tagName === "TD") {   // opciones de columnas: afectan solo a la fila donde se hace clic
      const nCel = (lineas[l] || "").split("\t").length;
      const maxC = Math.max(...lineas.map((x) => x.split("\t").length));
      if (nCel > 1) ops.push(["fila_ancha", "Quitar columnas de esta fila (ancho completo)"]);
      if (nCel < maxC) ops.push(["fila_col", "Añadir columna a esta fila"]);
    }
    if (!ops.length) { cerrarMenu(); return; }
    menu.innerHTML = "";
    ops.forEach(([modo, etiqueta]) => {
      const b = document.createElement("button");
      b.textContent = etiqueta;
      b.onclick = () => {
        enviar("streamlit:setComponentValue", {value: {id: Date.now() + "-" + Math.random(), linea: rg ? rg[0] : l, hasta: rg ? rg[1] : null, celda: c, modo: modo}, dataType: "json"});
        cerrarMenu();
      };
      menu.appendChild(b);
    });
    menu.style.display = "block";
    const mw = menu.offsetWidth, mh = menu.offsetHeight;
    let top = e.clientY;
    if (top + mh + 6 > window.innerHeight && e.clientY - mh >= 4) top = e.clientY - mh;  // no cabe debajo: hacia arriba
    menu.style.left = Math.max(4, Math.min(e.clientX, window.innerWidth - mw - 6)) + "px";
    menu.style.top = Math.max(4, top) + "px";
    menuAbierto = true;
    alto();  // si aun así no cabe (apunte muy corto), el marco crece lo justo mientras el menú está abierto
  });
  document.addEventListener("click", (e) => { if (!menu.contains(e.target)) cerrarMenu(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") cerrarMenu(); });
  window.addEventListener("blur", cerrarMenu);

  window.addEventListener("message", (ev) => {
    if (!ev.data || ev.data.type !== "streamlit:render") return;
    const tema = ev.data.theme;
    if (tema) {
      document.body.style.color = tema.textColor;
      menu.style.background = tema.secondaryBackgroundColor || "#fff";
      menu.querySelectorAll("button").forEach((b) => (b.style.color = tema.textColor));
      menu.style.color = tema.textColor;
    }
    const t = (ev.data.args || {}).texto || "";
    if (t !== ultimoTexto) { ultimoTexto = t; pintar(t); } else { ajustarAncho(); }
  });
  ajustarAncho();
  window.addEventListener("resize", alto);
  enviar("streamlit:componentReady", {apiVersion: 1});
  enviar("streamlit:setFrameHeight", {height: 40});
</script>
</body></html>
'''


def _preparar_componente_apunte():
    """Crea (si hace falta) la carpeta componente_apunte/ junto al script y registra el componente."""
    try:
        _DIR_COMP_APUNTE.mkdir(exist_ok=True)
        f = _DIR_COMP_APUNTE / "index.html"
        if not f.exists() or f.read_text(encoding="utf-8") != _HTML_COMP_APUNTE:
            f.write_text(_HTML_COMP_APUNTE, encoding="utf-8")
        return components.declare_component("apunte_negrita", path=str(_DIR_COMP_APUNTE))
    except Exception:
        return None  # sin componente se usa la visualización de siempre


_COMP_APUNTE = _preparar_componente_apunte()


def _pos_dos_puntos(txt: str):
    """Posición del primer «:» con texto delante (se ignoran los de «http://»)."""
    for m in re.finditer(":", txt):
        i = m.start()
        if txt[i + 1:i + 3] == "//":
            continue
        if txt[:i].strip():
            return i
    return None


_RE_CURSIVA = re.compile(r"^`[^`]+`$")


def _es_cursiva(txt: str) -> bool:
    """Una línea en cursiva se guarda envuelta en `comillas invertidas` (se ve como un bloque gris)."""
    return bool(_RE_CURSIVA.match((txt or "").strip()))


def _cursiva_celda(txt: str, modo: str) -> str:
    m = re.match(r"^(\s*)(.*?)(\s*)$", txt, flags=re.DOTALL)
    ini, nucleo, fin = m.groups()
    if modo == "quitar_cursiva":
        return f"{ini}{nucleo[1:-1] if _es_cursiva(nucleo) else nucleo}{fin}"
    if not nucleo or _es_cursiva(nucleo):
        return txt
    limpio = nucleo.replace("**", "").replace("`", "")
    return f"{ini}`{limpio}`{fin}" if limpio.strip() else txt


def _negrita_celda(txt: str, modo: str) -> str:
    if modo in ("cursiva", "quitar_cursiva"):
        return _cursiva_celda(txt, modo)
    if modo == "quitar":
        return txt.replace("**", "")
    m = re.match(r"^(\s*)(.*?)(\s*)$", txt, flags=re.DOTALL)
    ini, nucleo, fin = m.groups()
    if not nucleo:
        return txt
    if modo == "linea":
        return f"{ini}**{nucleo.replace('**', '')}**{fin}"
    if modo == "antes":
        i = _pos_dos_puntos(nucleo)
        if i is None:
            return txt
        return f"{ini}**{nucleo[:i].replace('**', '').strip()}:**{nucleo[i + 1:]}{fin}"
    return txt


def _cambiar_columnas_fila(fila: str, modo: str) -> str:
    """fila_ancha: junta todas las celdas de la fila en una sola (ocupa el ancho completo).
    fila_col: añade una celda vacía al final de la fila."""
    if modo == "fila_col":
        return fila + "\t"
    partes, hay_cursiva = [], False
    for c in fila.split("\t"):
        c = c.strip()
        if not c:
            continue
        if _es_cursiva(c):
            hay_cursiva = True
            c = c[1:-1]
        partes.append(c)
    unido = " ".join(partes)
    if hay_cursiva and unido.strip():
        return f"`{unido.replace('`', '')}`"
    return unido


def _aplicar_negrita(cont: str, linea: int, celda: int, modo: str, hasta=None) -> str:
    lineas = (cont or "").replace("\r\n", "\n").split("\n")
    if not (0 <= linea < len(lineas)):
        return cont
    if modo in ("fila_ancha", "fila_col"):
        lineas[linea] = _cambiar_columnas_fila(lineas[linea], modo)
        return "\n".join(lineas)
    if hasta is not None:  # varias líneas seleccionadas: el formato se aplica a todas
        es_cursiva_modo = modo in ("cursiva", "quitar_cursiva")
        for i in range(linea, min(max(hasta, linea), len(lineas) - 1) + 1):
            if not es_cursiva_modo and _es_cursiva(lineas[i]):
                continue  # las líneas en cursiva no llevan negrita
            lineas[i] = _negrita_celda(lineas[i], modo)
        return "\n".join(lineas)
    celdas = lineas[linea].split("\t")
    if not (0 <= celda < len(celdas)):
        return cont
    celdas[celda] = _negrita_celda(celdas[celda], modo)
    lineas[linea] = "\t".join(celdas)
    return "\n".join(lineas)


def actualizar_contenido_apunte(bid: int, contenido: str):
    with db(DB_APUNTES) as conn:
        conn.execute("UPDATE bloques SET contenido_bloque=? WHERE id=?", (contenido, bid))
    cargar_apuntes.clear()


def mostrar_apunte_negrita(bid, cont, clave):
    """Pinta el apunte (texto o tabla con tabuladores) y deja poner negrita con el botón derecho."""
    if _COMP_APUNTE is None:
        if "\t" in cont:
            mostrar_tabla_aggrid(cont, clave)
        else:
            st.markdown(_con_saltos(cont))
        return
    res = _COMP_APUNTE(texto=cont, key=f"cmpap_{clave}", default=None)
    if isinstance(res, dict) and res.get("id") and res["id"] != st.session_state.get(f"cmpap_ult_{clave}"):
        st.session_state[f"cmpap_ult_{clave}"] = res["id"]
        try:
            hasta = res.get("hasta")
            nuevo = _aplicar_negrita(
                cont, int(res["linea"]), int(res["celda"]), str(res["modo"]),
                int(hasta) if hasta is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return
        if nuevo != cont:
            actualizar_contenido_apunte(bid, nuevo)
            rerun_local()


AYUDA_NEGRITA = (
    "💡 Botón derecho sobre una línea del apunte: negrita (toda la línea o lo que va antes de «:») o cursiva con fondo gris. "
    "En las tablas también puedes quitar o añadir columnas solo en la fila donde haces clic. "
    "Si seleccionas varias líneas con el ratón y haces clic derecho dentro de la selección, el formato se aplica a todas (la cursiva las junta en un solo bloque)."
)


def mostrar_bloque(bid, etq, cont, prefijo, nivel="###"):
    """Cada apunte va en su propio cajetín (con borde), con el título en negrita y sin línea final."""
    with st.container(border=True):
        _contenido_bloque(bid, etq, cont, prefijo)


def _contenido_bloque(bid, etq, cont, prefijo):
    key_edit = f"edit_ap_{prefijo}_{bid}"

    if st.session_state.get(key_edit, False):
        with st.form(f"form_ed_ap_{prefijo}_{bid}"):
            nuevo_tit = st.text_input("Título / términos", value=etq or "")
            nuevo_cont = st.text_area("Contenido", value=cont or "", height=200)
            c_g, c_c, _ = st.columns([1, 1, 6])
            guardar = c_g.form_submit_button("💾 Guardar")
            cancelar = c_c.form_submit_button("Cancelar")
        if guardar:
            if not nuevo_tit.strip() or not nuevo_cont.strip():
                st.error("El título y el contenido son obligatorios.")
            else:
                actualizar_apunte(bid, nuevo_tit, nuevo_cont)
                agregar_a_glosario(nuevo_tit)  # todo apunte es también sugerencia
                st.session_state[key_edit] = False
                st.toast("✅ Apunte modificado")
                rerun_local()
        elif cancelar:
            st.session_state[key_edit] = False
            rerun_local()
        return

    st.markdown(f"**📌 {(etq or '').upper()}**")
    cont = formatear_apunte(cont)
    if "```" in cont:
        # apunte con bloque de código (XML, SQL, etc.): se pinta como Markdown aunque lleve tabuladores
        st.markdown(_con_saltos(cont.replace("\t", "    ")))
    else:
        mostrar_apunte_negrita(bid, cont, f"{prefijo}_{bid}")

    c_mod, c_del, _ = st.columns([1.2, 1.2, 6])
    with c_mod:
        if st.button("✏️ Modificar", key=f"mod_{prefijo}_{bid}", use_container_width=True):
            st.session_state[key_edit] = True
            rerun_local()
    with c_del:
        with st.popover("🗑️ Borrar", use_container_width=True):
            st.warning("Se borrará este apunte de apuntes.db.")
            if st.button("Confirmar borrado", key=f"del_{prefijo}_{bid}"):
                borrar_apunte(bid)
                st.toast("Apunte eliminado")
                rerun_local()


# ==========================================
# PREGUNTAS REPETIDAS (la misma pregunta en varios exámenes)
# ==========================================
def _compacto(txt: str) -> str:
    """Minúsculas, sin tildes ni espacios/puntuación: tolera el ruido típico de los PDF ('al me nos')."""
    t = unicodedata.normalize("NFD", txt or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _misma_pregunta(opts_a, opts_b, umbral: float = 0.8) -> bool:
    """Mismo enunciado + opciones equivalentes (sin importar el orden ni pequeñas diferencias de texto).
    Con enunciados genéricos («Señale la respuesta correcta:») las opciones distintas las diferencian."""
    A = [_compacto(o) for o in opts_a if o]
    B = [_compacto(o) for o in opts_b if o]
    if not A or not B:
        return False

    def cubre(X, Y):
        return all(max(difflib.SequenceMatcher(None, x, y).ratio() for y in Y) >= umbral for x in X)

    return cubre(A, B) and cubre(B, A)


def quitar_preguntas_repetidas(filas):
    """filas: (id, numero, enunciado, op_a, op_b, op_c, op_d, correcta), ya barajadas.
    Devuelve (filas_sin_repetidas, cuántas_se_omitieron)."""
    vistas = {}  # enunciado compacto -> [opciones de las ya aceptadas]
    unicas, omitidas = [], 0
    for f in filas:
        clave = _compacto(f[2])
        opciones = f[3:7]
        if any(_misma_pregunta(opciones, previa) for previa in vistas.get(clave, [])):
            omitidas += 1
            continue
        vistas.setdefault(clave, []).append(opciones)
        unicas.append(f)
    return unicas, omitidas


@st.cache_data(show_spinner=False)
def contar_repetidas_guardadas(n_preguntas: int, max_id: int) -> int:
    """Cuántas preguntas guardadas están repetidas entre exámenes (n_preguntas y max_id solo invalidan la caché)."""
    with db(DB_TESTS) as conn:
        filas = conn.execute(
            "SELECT id, numero, enunciado, opcion_a, opcion_b, opcion_c, opcion_d, correcta FROM preguntas"
        ).fetchall()
    return quitar_preguntas_repetidas(filas)[1]


def contar_ya_existentes(preguntas_nuevas):
    """Cuántas de las preguntas recién leídas ya están guardadas (en cualquier examen)."""
    with db(DB_TESTS) as conn:
        existentes = conn.execute(
            "SELECT enunciado, opcion_a, opcion_b, opcion_c, opcion_d FROM preguntas"
        ).fetchall()
    por_enunciado = {}
    for e in existentes:
        por_enunciado.setdefault(_compacto(e[0]), []).append(e[1:5])
    return sum(
        1 for p in preguntas_nuevas
        if any(
            _misma_pregunta((p["opcion_a"], p["opcion_b"], p["opcion_c"], p["opcion_d"]), op)
            for op in por_enunciado.get(_compacto(p["enunciado"]), [])
        )
    )


# ==========================================
# ALTA DE APUNTES Y GESTIÓN DEL GLOSARIO
# ==========================================
_RE_LINEA_XML = re.compile(r"^\s*<[?!/]?[A-Za-z][^\n]*>\s*$|^\s*<\?xml[^\n]*\?>\s*$")
_RE_ETIQUETA = re.compile(r"</?[A-Za-z?][^<>\n`]*>")
_RE_ATRIBUTO = re.compile(r"(?<![\w`])[A-Za-z_][\w-]*=\"[^\"\n]*\"")
_RE_ROTULO = re.compile(r"^(\s*)([^\n:<>`*]{2,40}):(\s+\S.*)$")


# --- Texto pegado desde Gemini/ChatGPT: que se vea igual que en el chat ---------------------
_RE_TITULO_NUM = re.compile(
    r"^(\s*\d+[.)])\s+([^\n*`]{1,60}?)\s+((?:Es|Son|Sirve|Permite)\s+(?:un|una|unos|unas|el|la|los|las|lo|para)\b.*)$"
)
_RE_TITULO_DOSPUNTOS = re.compile(r"^(\s*\d+[.)])\s+([^\n:*`<>]{2,60}):(\s+\S.*)$")
_RE_ARTE_LINEA = re.compile(r"^\s*[+|].*[+|]\s*$")
_RE_ARTE_BORDE = re.compile(r"^\s*\+[-=+]{3,}\+\s*$")
_RE_PY_INICIO = re.compile(
    r"^(?:(?:def|class|if|elif|for|while|with|try|except)\b.*:|else:|finally:|try:"
    r"|import\s+\w[\w.]*|from\s+[\w.]+\s+import\s+.+|return\b.*|print\(.*\)"
    r"|@\w[\w.]*(?:\(.*\))?|[A-Za-z_][\w.\[\]]*\s*(?:[-+*/]?=)(?!=).*|[A-Za-z_][\w.]*\(.*\)\s*|#.*)\s*$"
)
_RE_PY_FUERTE = re.compile(r"\b(?:def|while|elif|import|return|self)\b|\w\.\w+\(|\w+\s=\s|:\s*$")


def _negrita_titulos_numerados(txt: str) -> str:
    """Pone en negrita el rótulo de las líneas tipo «9. Iterator Es un patrón…» (fuera de bloques de código).
    Es idempotente: las líneas que ya llevan ** no se tocan."""
    salida, en_codigo = [], False
    for l in txt.split("\n"):
        if l.lstrip().startswith("```"):
            en_codigo = not en_codigo
        elif not en_codigo and "**" not in l:
            m = _RE_TITULO_NUM.match(l)
            if m and len(m.group(2).split()) <= 6:
                l = f"**{m.group(1).strip()} {m.group(2).strip()}** {m.group(3)}"
            else:
                m2 = _RE_TITULO_DOSPUNTOS.match(l)
                if m2 and len(m2.group(2).split()) <= 6:
                    l = f"**{m2.group(1).strip()} {m2.group(2).strip()}:**{m2.group(3)}"
        salida.append(l)
    return "\n".join(salida)


def _etiquetar_bloques_python(txt: str) -> str:
    """Los bloques ``` sin lenguaje que parecen Python pasan a ```python (así se colorea como en Gemini)."""
    lineas, salida, i = txt.split("\n"), [], 0
    while i < len(lineas):
        l = lineas[i]
        if l.strip().startswith("```"):
            fin = next((j for j in range(i + 1, len(lineas)) if lineas[j].strip() == "```"), None)
            if fin is not None:
                cuerpo = lineas[i + 1:fin]
                if (l.strip() == "```" and not any(_RE_LINEA_XML.match(c) for c in cuerpo)
                        and _RE_PY_FUERTE.search("\n".join(cuerpo))):
                    l = l.replace("```", "```python", 1)
                salida.append(l)
                salida.extend(cuerpo)
                salida.append(lineas[fin])
                i = fin + 1
                continue
        salida.append(l)
        i += 1
    return "\n".join(salida)


def _python_suelto_a_bloques(txt: str) -> str:
    """Si se pegó código Python sin ``` (el portapapeles pierde las vallas), lo agrupa en un bloque ```python."""
    lineas = txt.split("\n")
    es_cod = lambda l: bool(l.strip()) and bool(_RE_PY_INICIO.match(l.strip()) if not l[:1].isspace() else True)
    salida, bloque, n = [], [], len(lineas)
    i = 0

    def cerrar():
        while bloque and not bloque[-1].strip():
            bloque.pop()
        if bloque:
            if _RE_PY_FUERTE.search("\n".join(bloque)):
                if salida and salida[-1].strip():
                    salida.append("")
                salida.append("```python")
                salida.extend(x.replace("\t", "    ").rstrip() for x in bloque)
                salida.append("```")
                salida.append("")
            else:
                salida.extend(bloque)
        bloque.clear()

    while i < n:
        l = lineas[i]
        inicio = bool(l.strip()) and not l[:1].isspace() and _RE_PY_INICIO.match(l.strip())
        sangrada = bool(bloque) and bool(l.strip()) and l[:1].isspace()
        if inicio or sangrada:
            bloque.append(l)
        elif bloque and not l.strip() and i + 1 < n and lineas[i + 1][:1].isspace() and lineas[i + 1].strip():
            bloque.append(l)  # hueco dentro de un bloque sangrado
        else:
            cerrar()
            salida.append(l)
        i += 1
    cerrar()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(salida)).strip("\n")


_RE_DIRECTIVA = re.compile(r"^([A-Za-z][\w.-]*):\s+(\S.*)$")
_RE_LINK_MD = re.compile(r"\[([^\]\n]+)\]\((?:https?://)[^)\s]+\)")


def _directivas_a_lista(txt: str) -> str:
    """Texto copiado de un chat (se pierde el formato): «Directiva: explicación» seguida de su línea de
    ejemplo («Directiva valor») -> lista numerada con `Directiva` en código y el ejemplo en un bloque."""
    txt = txt.replace("\r\n", "\n")
    lineas = txt.split("\n")
    idx = [i for i, l in enumerate(lineas) if _RE_DIRECTIVA.match(l)]
    if len(idx) < 2:
        return txt

    def bloque_de(k):
        ini = idx[k] + 1
        fin = idx[k + 1] if k + 1 < len(idx) else len(lineas)
        b = lineas[ini:fin]
        while b and not b[0].strip():
            b.pop(0)
        while b and not b[-1].strip():
            b.pop()
        return b

    validos = 0
    for k, i in enumerate(idx):
        tok = _RE_DIRECTIVA.match(lineas[i]).group(1)
        b = bloque_de(k)
        if b and (b[0].lstrip().startswith(tok + " ") or b[0].lstrip().startswith("<")):
            validos += 1
    if validos < 2:
        return txt

    salida = lineas[:idx[0]]
    if salida and salida[-1].strip():
        salida.append("")
    for k, i in enumerate(idx):
        m = _RE_DIRECTIVA.match(lineas[i])
        salida.append(f"{k + 1}. **{m.group(1)}:** {m.group(2)}")
        for x in bloque_de(k):
            x = _RE_LINK_MD.sub(r"\1", x).replace("\t", "    ").replace("`", "").rstrip()
            if x.strip():  # cada línea va en `cursiva`: el componente las junta en un bloque gris
                salida.append(f"{x[:len(x) - len(x.lstrip())]}`{x.strip()}`")
        salida.append("")
    return "\n".join(salida).strip("\n")


_LENGUAJES_COMPONENTE = {"", "xml", "html", "apache", "apacheconf", "conf", "ini", "text", "txt"}


def _vallas_a_componente(txt: str) -> str:
    """Apuntes con bloques ``` de XML/config (sin Python): se pasan al formato del componente (cada línea de
    código entre `acentos`, que se pinta como bloque gris) para que funcione el botón derecho.
    Las líneas sangradas que quedaron fuera de las vallas (el portapapeles las deja sueltas) también son código."""
    if "```" not in txt or "\t" in txt:
        return txt
    lineas = txt.split("\n")
    langs = {m.group(1).lower() for l in lineas for m in [re.match(r"^\s*```(\w*)", l)] if m}
    if not langs <= _LENGUAJES_COMPONENTE:
        return txt
    salida, dentro = [], False
    for l in lineas:
        if l.strip().startswith("```"):
            dentro = not dentro
            continue
        if dentro or (l.strip() and (l.startswith("    ") or l.startswith("\t"))):
            x = _RE_LINK_MD.sub(r"\1", l).replace("`", "").rstrip()
            if x.strip():
                salida.append(f"{x[:len(x) - len(x.lstrip())]}`{x.strip()}`")
            continue
        salida.append(l.replace("`", ""))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(salida)).strip("\n")


def _arte_ascii_a_componente(txt: str) -> str:
    """Esquemas de cajas ASCII (+----+ / | texto |) pegados sin vallas: cada línea va entre `acentos` para que se
    pinten en monoespaciado, con sus espacios, en un bloque gris (y se pueda seguir usando el botón derecho)."""
    if "```" in txt or "\t" in txt:
        return txt
    lineas = txt.split("\n")
    salida, i, n = [], 0, len(lineas)
    while i < n:
        if not _RE_ARTE_LINEA.match(lineas[i]):
            salida.append(lineas[i])
            i += 1
            continue
        j = i
        while j < n and _RE_ARTE_LINEA.match(lineas[j]):
            j += 1
        grupo = lineas[i:j]
        if len(grupo) >= 3 and any(_RE_ARTE_BORDE.match(g) for g in grupo):
            for g in grupo:
                g = g.replace("`", "").rstrip()
                salida.append(f"{g[:len(g) - len(g.lstrip())]}`{g.strip()}`")
        else:
            salida.extend(grupo)
        i = j
    return "\n".join(salida)


_RE_ITEM_FENCED = re.compile(r"^(\d+)\. `([\w.-]+)`: (.*)$")


def _desfenciar_directivas(txt: str) -> str:
    """Apuntes guardados con la versión anterior (lista con ``` sangrado): se pasan al formato del
    componente (negrita + líneas `cursiva`) para que vuelva a funcionar el botón derecho."""
    if "\n    ```" not in txt or not any(_RE_ITEM_FENCED.match(l) for l in txt.split("\n")):
        return txt
    salida, dentro = [], False
    for l in txt.split("\n"):
        if not dentro and l.startswith("    ```"):
            dentro = True
            continue
        if dentro:
            if l.strip() == "```":
                dentro = False
            elif l.strip():
                sangria = l[4:] if l.startswith("    ") else l
                sangria = sangria.rstrip()
                salida.append(f"{sangria[:len(sangria) - len(sangria.lstrip())]}`{sangria.strip()}`")
            continue
        m = _RE_ITEM_FENCED.match(l)
        salida.append(f"{m.group(1)}. **{m.group(2)}:** {m.group(3)}" if m else l)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(salida)).strip("\n")


def formatear_apunte(txt: str) -> str:
    """Pone formato Markdown a un apunte pegado como texto plano: las líneas de código XML/HTML
    se agrupan en un bloque ```xml (conservando la sangría), las etiquetas y atributos sueltos van
    en `código`, y las líneas 'Rótulo: valor' consecutivas se convierten en lista numerada con
    el rótulo en negrita. Si el apunte ya trae bloques ``` o no tiene código, no se toca."""
    txt = (txt or "").replace("\r\n", "\n")
    txt = _negrita_titulos_numerados(txt)  # «8. Strategy Es un…» -> «**8. Strategy** Es un…»
    txt = _desfenciar_directivas(txt)
    txt = _vallas_a_componente(txt)
    txt = _arte_ascii_a_componente(txt)
    if "```" not in txt:
        txt = _directivas_a_lista(txt)  # «ServerName: …» + «ServerName valor» -> lista con bloques de código
    if "```" in txt:
        return _etiquetar_bloques_python(txt)
    lineas = txt.split("\n")
    if not any(_RE_LINEA_XML.match(l) for l in lineas):
        return _python_suelto_a_bloques(txt)

    # 1) bloques de código
    salida, bloque = [], []

    def cerrar():
        if bloque:
            if salida and salida[-1].strip():
                salida.append("")
            salida.append("```xml")
            salida.extend(l.replace("\t", "    ").rstrip() for l in bloque)
            salida.append("```")
            salida.append("")
            bloque.clear()

    for l in lineas:
        if _RE_LINEA_XML.match(l):
            bloque.append(l)
        elif bloque and not l.strip():
            bloque.append(l)  # línea en blanco dentro del bloque: se conserva si sigue más código
        else:
            if bloque:
                while bloque and not bloque[-1].strip():
                    bloque.pop()
                cerrar()
            salida.append(l)
    while bloque and not bloque[-1].strip():
        bloque.pop()
    cerrar()

    # 2) fuera de los bloques: etiquetas/atributos en `código`, rótulos en negrita y lista numerada
    resultado, en_codigo, grupo = [], False, []

    def volcar_grupo():
        if len(grupo) >= 2:
            resultado.extend(f"{i}. **{r.strip()}:**{resto}" for i, (r, resto) in enumerate(grupo, 1))
        else:
            resultado.extend(f"{r}:{resto}" for r, resto in grupo)
        grupo.clear()

    for l in salida:
        if l.startswith("```"):
            volcar_grupo()
            en_codigo = not en_codigo
            resultado.append(l)
            continue
        if en_codigo:
            resultado.append(l)
            continue
        l2 = _RE_ETIQUETA.sub(lambda m: f"`{m.group(0)}`", l)
        if "=" in l2:  # atributos sueltos (id="U001"), solo fuera de los `códigos` ya marcados
            trozos = l2.split("`")
            for k in range(0, len(trozos), 2):
                trozos[k] = _RE_ATRIBUTO.sub(lambda m: f"`{m.group(0)}`", trozos[k])
            l2 = "`".join(trozos)
        m = _RE_ROTULO.match(l2)
        if m and "`" in m.group(3):
            grupo.append((m.group(2), m.group(3)))
            continue
        volcar_grupo()
        resultado.append(l2)
    volcar_grupo()
    return "\n".join(resultado).strip("\n")


def guardar_apunte(titulo: str, contenido: str):
    with db(DB_APUNTES) as conn:
        conn.execute(
            "INSERT INTO bloques (titulo_etiquetas, contenido_bloque) VALUES (?, ?)",
            (titulo.strip(), formatear_apunte(contenido).strip()),
        )
    cargar_apuntes.clear()  # la caché de búsqueda dura 60 s: se invalida para ver el apunte al instante


def agregar_a_glosario(termino: str) -> bool:
    """Añade el término a glosario.txt. Devuelve False si ya existía o está vacío."""
    termino = termino.strip().lower()
    if not termino or termino in st.session_state.glosario_externo:
        return False
    ruta = Path("glosario.txt")
    prefijo = ""
    if ruta.exists() and ruta.stat().st_size > 0:
        with open(ruta, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefijo = "\n"
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(f"{prefijo}{termino}\n")
    st.session_state.glosario_externo = cargar_glosario_externo()
    return True


def _anadir_termino_cb(p_id):
    """Botón ➕ del test: guarda el término en el glosario, lo deja como sugerencia en esta pregunta
    aunque no aparezca literalmente en su texto, y vacía la casilla."""
    t = (st.session_state.get(f"input_{p_id}") or "").strip().lower()
    if not t:
        return
    nuevo = agregar_a_glosario(t)
    extras = st.session_state.setdefault(f"glos_extra_{p_id}", [])
    if t not in extras:
        extras.append(t)
    st.session_state[f"input_{p_id}"] = ""
    st.toast(f"«{t}» añadido al glosario" if nuevo else f"«{t}» ya estaba en el glosario")


def quitar_del_glosario(termino: str) -> int:
    """Quita el término de glosario.txt y borra de apuntes.db los apuntes titulados exactamente así.
    Devuelve cuántos apuntes se borraron."""
    t = termino.strip().lower()
    ruta = Path("glosario.txt")
    if ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            lineas = f.read().splitlines()
        restantes = [l for l in lineas if l.strip().lower() != t]
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(restantes) + ("\n" if restantes else ""))
    st.session_state.glosario_externo = cargar_glosario_externo()

    with db(DB_APUNTES) as conn:
        ids = [
            bid for bid, tit in conn.execute("SELECT id, titulo_etiquetas FROM bloques").fetchall()
            if (tit or "").strip().lower() == t
        ]
        conn.executemany("DELETE FROM bloques WHERE id=?", [(i,) for i in ids])
    cargar_apuntes.clear()
    return len(ids)


def formulario_nuevo_apunte(clave: str, titulo_inicial: str = ""):
    """Formulario de alta de apunte (título + descripción) con el título ya rellenado."""
    sufijo = f"{clave}_{titulo_inicial}"
    with st.form(f"form_nuevo_apunte_{clave}", clear_on_submit=True):
        titulo = st.text_input("Título / término", value=titulo_inicial, key=f"nuevo_tit_{sufijo}")
        contenido = st.text_area(
            "Descripción / contenido (admite Markdown; con tabuladores se muestra como tabla)",
            height=140, key=f"nuevo_cont_{sufijo}",
        )
        guardar = st.form_submit_button("💾 Guardar apunte")

    if guardar:
        if not titulo.strip() or not contenido.strip():
            st.error("El título y la descripción son obligatorios.")
            return
        guardar_apunte(titulo, contenido)
        agregar_a_glosario(titulo)  # solo lo añade si aún no está en glosario.txt
        st.toast("✅ Apunte guardado")
        rerun_local()


tab_hacer_test, tab_estudia, tab_apuntes, tab_agregar_test, tab_temario = st.tabs([
    "🎯 Hacer Test Global",
    "🧠 Estudia con la IA",
    "🔍 Apuntes y Tablas",
    "➕ Subir y Gestionar Exámenes",
    "📋 Temario TAI",
])


# ==========================================
# PESTAÑA 1 - HACER TEST
# ==========================================
SUGERENCIAS_SEGUIMIENTO = (
    "Ponme un ejemplo de cada opción",
    "Explícamelo más sencillo",
    "Dame una regla para recordarlo",
)


def _mensajes_seguimiento(texto_pregunta, correcta, explicacion, historial):
    sistema = (
        "Eres profesor de oposiciones de TAI (Técnico Auxiliar de Informática). Estás ayudando a un alumno "
        "a entender una pregunta de test. Responde SOLO a lo que te pide ahora, en español, claro y concreto, "
        "en Markdown y sin rodeos. Si pide ejemplos, ponlos reales (código, comandos, casos prácticos) y "
        "breves, uno por opción si dice «de cada». No repitas la explicación anterior salvo que te lo pidan "
        "y no te contradigas. Si la pregunta es ambigua o la respuesta marcada como correcta parece falsa, "
        "dilo.\n\n"
        f"Pregunta:\n{texto_pregunta}\n\nOpción correcta según el examen: {correcta}.\n\n"
        f"Explicación que ya le diste al alumno:\n{explicacion}"
    )
    return [{"role": "system", "content": sistema}] + list(historial)[-10:]


def seccion_seguimiento(p_id, texto_pregunta, correcta, explicacion, job_ia):
    """Chat bajo la explicación para repreguntar a Gemini (ejemplos, versión más simple, etc.)."""
    clave_chat = f"chat_ia_{p_id}"
    historial = st.session_state.setdefault(clave_chat, [])

    st.markdown("#### 💬 Preguntar más sobre esta explicación")
    for m in historial:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    if job_ia is not None and job_ia.subtipo == "seguimiento":
        with st.chat_message("assistant"):
            st.markdown(job_ia.texto or "⏳ Pensando...")
        return

    err = st.session_state.pop(f"chat_err_{p_id}", None)
    if err:
        st.error(err)

    nueva = None
    cols = st.columns(len(SUGERENCIAS_SEGUIMIENTO))
    for i, (col, sg) in enumerate(zip(cols, SUGERENCIAS_SEGUIMIENTO)):
        if col.button(sg, key=f"sg_{p_id}_{i}", use_container_width=True):
            nueva = sg

    with st.form(f"form_seg_{p_id}", clear_on_submit=True):
        txt = st.text_input(
            "Tu pregunta", placeholder="Ej.: ponme un ejemplo de cada una", label_visibility="collapsed"
        )
        c_env, c_borrar, _ = st.columns([1, 1.4, 5])
        enviar = c_env.form_submit_button("Preguntar")
        limpiar = c_borrar.form_submit_button("🧹 Nueva conversación") if historial else False
    if limpiar:
        st.session_state[clave_chat] = []
        rerun_local()
    if enviar and txt.strip():
        nueva = txt.strip()

    if nueva:
        historial.append({"role": "user", "content": nueva})
        mensajes = _mensajes_seguimiento(texto_pregunta, correcta, explicacion, historial)
        job = Job("ia", subtipo="seguimiento", p_id=p_id)
        st.session_state[f"job_ia_{p_id}"] = job
        get_executor_test().submit(trabajo_ia_seguimiento, job, mensajes, TOKENS_SEGUIMIENTO, get_client())
        st.rerun()


def _enunciado_md(enun: str) -> str:
    """Enunciado en negrita grande SIN que se corte: los saltos de línea del PDF se unen en una sola línea
    (un encabezado markdown solo abarca una línea) y los apartados tipo lista van en su propia línea."""
    txt = (enun or "").replace("\r\n", "\n").replace("**", "")
    bloques, actual = [], []
    for linea in txt.split("\n"):
        l = linea.strip()
        if not l:
            if actual:
                bloques.append(" ".join(actual))
                actual = []
        elif actual and re.match(r"^(?:\d+[.)]|[-•–]|[IVX]+[.)]|[a-eA-E][.)])\s", l):
            bloques.append(" ".join(actual))
            actual = [l]
        else:
            actual.append(l)
    if actual:
        bloques.append(" ".join(actual))
    # un salto de párrafo en mitad de una frase (el PDF partió la página) se vuelve a unir
    unidos = []
    for b in bloques:
        if unidos and not re.search(r"[.:;?!)»\"]$", unidos[-1]) and b[:1].islower():
            unidos[-1] += " " + b
        else:
            unidos.append(b)
    return "\n\n".join("### " + b for b in unidos)


GRUPO_SIN_BLOQUE = "Muestreo IA"  # las preguntas sin bloque (las generadas por la IA) se agrupan con este nombre
GRUPO_OMITIDAS = "Omitidas"       # preguntas eliminadas con el botón 🗑️: no salen en los demás grupos
GRUPO_LEYES = "Leyes"             # grupo fijo (sale siempre, aunque esté vacío); recibe las preguntas del botón ⏩


def _grupo_test(bloque, omitida=0) -> str:
    """Grupo del test al que pertenece una pregunta: Primera Parte, Supuesto N, Reserva (todas),
    Muestreo IA (sin bloque) u Omitidas (eliminadas por el usuario)."""
    if omitida:
        return GRUPO_OMITIDAS
    b = (bloque or "").strip()
    if not b:
        return GRUPO_SIN_BLOQUE
    return "Reserva" if b.lower().startswith("reserva") else b


def _orden_grupo_test(nombre: str):
    if nombre == "Primera Parte":
        return (0, 0, nombre)
    m = re.match(r"Supuesto\s+(\d+)", nombre)
    if m:
        return (1, int(m.group(1)), nombre)
    if nombre == "Reserva":
        return (2, 0, nombre)
    if nombre == GRUPO_SIN_BLOQUE:
        return (4, 0, nombre)
    if nombre == GRUPO_OMITIDAS:
        return (5, 0, nombre)
    if nombre == GRUPO_LEYES:
        return (6, 0, nombre)
    return (3, 0, nombre)


def _ids_equivalentes(conn, p_id: int):
    """La pregunta y sus copias idénticas de otros exámenes (se mueven siempre juntas)."""
    base = conn.execute(
        "SELECT enunciado, opcion_a, opcion_b, opcion_c, opcion_d FROM preguntas WHERE id=?", (p_id,)
    ).fetchone()
    ids = [p_id]
    if base:
        clave = _compacto(base[0])
        for f in conn.execute(
            "SELECT id, enunciado, opcion_a, opcion_b, opcion_c, opcion_d FROM preguntas WHERE id<>?", (p_id,)
        ).fetchall():
            if _compacto(f[1]) == clave and _misma_pregunta(base[1:5], f[2:6]):
                ids.append(f[0])
    return ids


def marcar_omitida(p_id: int, omitida: bool):
    """Marca (o desmarca) la pregunta y sus copias idénticas de otros exámenes. Devuelve los ids afectados."""
    with db(DB_TESTS) as conn:
        ids = _ids_equivalentes(conn, p_id)
        conn.executemany(
            "UPDATE preguntas SET omitida=? WHERE id=?", [(1 if omitida else 0, i) for i in ids]
        )
    return ids


def mover_pregunta(p_id: int, destino: str):
    """Pasa la pregunta (y sus copias idénticas) a otro grupo. Devuelve los ids afectados."""
    with db(DB_TESTS) as conn:
        ids = _ids_equivalentes(conn, p_id)
        if destino == GRUPO_OMITIDAS:
            conn.executemany("UPDATE preguntas SET omitida=1 WHERE id=?", [(i,) for i in ids])
        else:
            bloque = "" if destino == GRUPO_SIN_BLOQUE else destino
            conn.executemany(
                "UPDATE preguntas SET bloque=?, omitida=0 WHERE id=?", [(bloque, i) for i in ids]
            )
    return ids


GRUPOS_TEST_POR_DEFECTO = ("Primera Parte", "Supuesto 2")


def _puntuacion_sesion(aciertos: int, fallos: int):
    """Acierto +3, fallo -2. Las preguntas omitidas con ⏩ no suman ni restan. El % compara los puntos
    netos con los 3 puntos de cada acierto (6 bien, 2 mal: 18-4 = 14 → 14/18 = 77 %)."""
    brutos = 3 * aciertos
    netos = brutos - 2 * fallos
    pct = 0 if brutos <= 0 else max(0, min(100, int(100 * netos / brutos)))
    return netos, pct


def _color_pct(pct: int) -> str:
    if pct <= 20:
        return "#e53935"   # rojo
    if pct <= 40:
        return "#fb8c00"   # naranja
    if pct <= 60:
        return "#fdd835"   # amarillo
    if pct <= 80:
        return "#1e88e5"   # azul
    return "#43a047"       # verde


def mostrar_estadisticas_sesion():
    a = st.session_state.get("aciertos_global", 0)
    f = st.session_state.get("fallos_global", 0)
    o = st.session_state.get("omitidas_global", 0)
    netos, pct = _puntuacion_sesion(a, f)
    color = _color_pct(pct)
    st.markdown(
        f"<div style='font-size:0.9rem;margin:2px 0 4px 0'>"
        f"📊 <b>Hechas:</b> {a + f + o} &nbsp;·&nbsp; ✅ <b>{a}</b> acertadas &nbsp;·&nbsp; "
        f"❌ <b>{f}</b> fallidas &nbsp;·&nbsp; ⏭️ <b>{o}</b> omitidas &nbsp;·&nbsp; "
        f"<b>{netos}</b> pts &nbsp;·&nbsp; <b style='color:{color}'>{pct}%</b></div>"
        f"<div style='background:rgba(128,128,128,0.25);border-radius:6px;height:10px;width:100%;overflow:hidden'>"
        f"<div style='background:{color};width:{pct}%;height:100%;border-radius:6px;"
        f"transition:width .3s'></div></div>",
        unsafe_allow_html=True,
    )


def _md(txt: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", txt or "")


def _negrita_parentesis(texto: str) -> str:
    """Pone en negrita lo que va entre paréntesis (sin las comillas invertidas de dentro).
    Soporta paréntesis anidados: «(`Secuencia de comandos en sitios cruzados (XSS)`)» queda entero
    en negrita, no a medias. No toca los paréntesis que están dentro de un `código`, como `printf(x)`."""
    texto = texto or ""
    salida, i, n, en_codigo = [], 0, len(texto), False
    while i < n:
        ch = texto[i]
        if ch == "`":
            en_codigo = not en_codigo
        elif ch == "(" and not en_codigo:
            prof, j, fin = 0, i, -1
            while j < n and texto[j] != "\n":  # el paréntesis que cierra, en la misma línea
                if texto[j] == "(":
                    prof += 1
                elif texto[j] == ")":
                    prof -= 1
                    if prof == 0:
                        fin = j
                        break
                j += 1
            if fin > 0:
                dentro = texto[i + 1:fin].replace("`", "").replace("**", "").strip()
                if dentro:
                    salida.append(f"(**{dentro}**)")
                    i = fin + 1
                    continue
        salida.append(ch)
        i += 1
    return "".join(salida)


def _filas_omitidas():
    """Preguntas omitidas; las copias idénticas de otros exámenes cuentan como una sola."""
    with db(DB_TESTS) as conn:
        filas = conn.execute(
            "SELECT id, numero, enunciado, opcion_a, opcion_b, opcion_c, opcion_d, correcta "
            "FROM preguntas WHERE COALESCE(omitida, 0)=1 ORDER BY id"
        ).fetchall()
    return quitar_preguntas_repetidas(filas)[0]


def _filas_grupo(grupo: str):
    with db(DB_TESTS) as conn:
        filas = conn.execute(
            "SELECT id, numero, enunciado, opcion_a, opcion_b, opcion_c, opcion_d, correcta, bloque, "
            "COALESCE(omitida, 0) FROM preguntas ORDER BY id"
        ).fetchall()
    return [f for f in filas if _grupo_test(f[8], f[9]) == grupo]


def _ver_grupo_cb(g: str):
    """Al marcar un grupo, el panel de abajo pasa a mostrar sus preguntas (por defecto: las omitidas)."""
    if st.session_state.get(f"grp_test_{g}"):
        st.session_state.vista_grupo_test = g
    elif st.session_state.get("vista_grupo_test") == g:
        st.session_state.vista_grupo_test = GRUPO_OMITIDAS


def _mover_cb(p_id: int, key: str):
    destino = st.session_state.get(key)
    if destino:
        mover_pregunta(p_id, destino)
        st.toast(f"➡️ Pregunta movida a «{destino}»")


def mostrar_panel_grupo(grupo: str, destinos):
    """Enunciados del grupo elegido, cada uno con un desplegable para moverlo a otro grupo."""
    es_omit = grupo == GRUPO_OMITIDAS
    filas = _filas_omitidas() if es_omit else _filas_grupo(grupo)
    if not filas and es_omit:
        return
    st.markdown("---")
    icono = "🗑️" if es_omit else "📋"
    with st.expander(f"{icono} {grupo} ({len(filas)})", expanded=True):
        if not filas:
            st.caption("Este grupo está vacío.")
            return
        opciones = [d for d in destinos if d != grupo]
        with st.container(height=520, border=False):
            for f in filas:
                pesos = [6, 1.6, 2.4] if es_omit else [8, 2.4]
                cols = st.columns(pesos, gap="small", vertical_alignment="center")
                with cols[0]:
                    st.markdown(_md(" ".join((f[2] or "").split())))
                with cols[-1]:
                    key_mv = f"mv_{grupo}_{f[0]}"
                    st.selectbox(
                        "Mover a", opciones, index=None, placeholder="Mover a…", key=key_mv,
                        label_visibility="collapsed", on_change=_mover_cb, args=(f[0], key_mv),
                    )
                if es_omit:
                    with cols[1]:
                        if st.button("♻️ Recuperar", key=f"recup_{f[0]}", use_container_width=True,
                                     help="Vuelve a su grupo original"):
                            marcar_omitida(f[0], False)
                            st.toast("♻️ Pregunta recuperada")
                            rerun_local()


def zona_test():
    if recoger_jobs_test():
        st.rerun()

    st.header("Practica tus conocimientos haciendo TESTs")

    with db(DB_TESTS) as conn:
        total_preguntas_bd = conn.execute("SELECT COUNT(*) FROM preguntas").fetchone()[0]

    if total_preguntas_bd == 0:
        st.info("No hay preguntas guardadas todavía. Sube algún examen en la pestaña de Subir.")
        return

    if "test_global_activo" not in st.session_state:
        with db(DB_TESTS) as conn:
            filas_bloque = conn.execute(
                "SELECT bloque, COALESCE(omitida, 0), COUNT(*) FROM preguntas GROUP BY bloque, COALESCE(omitida, 0)"
            ).fetchall()
        conteo = {}
        for bl, om, n in filas_bloque:
            g = _grupo_test(bl, om)
            conteo[g] = conteo.get(g, 0) + n
        if GRUPO_OMITIDAS in conteo:
            conteo[GRUPO_OMITIDAS] = len(_filas_omitidas())  # sin contar las copias repetidas
        conteo.setdefault(GRUPO_LEYES, 0)  # Leyes sale siempre, aunque esté vacío
        grupos = sorted(conteo, key=_orden_grupo_test)
        destinos = sorted(set(grupos) | {GRUPO_OMITIDAS, GRUPO_LEYES}, key=_orden_grupo_test)
        st.session_state.setdefault("vista_grupo_test", GRUPO_OMITIDAS)

        # por defecto solo Primera Parte y Supuesto 2 (si no existe ninguno, todos)
        defecto = [g for g in grupos if g in GRUPOS_TEST_POR_DEFECTO] or grupos
        for g in grupos:
            st.session_state.setdefault(f"grp_test_{g}", g in defecto)

        st.markdown("**Elige qué grupos quieres practicar:**")
        etiquetas = {g: f"{g} ({conteo[g]})" for g in grupos}
        elegidos = []
        with st.container(key="fila_grupos"):
            # cada columna mide según lo que tarda su texto, para que no se corten los números
            cols = st.columns([len(etiquetas[g]) + 6 for g in grupos] or [1], gap="small")
        for i, g in enumerate(grupos):
            with cols[i]:
                if st.checkbox(etiquetas[g], key=f"grp_test_{g}", on_change=_ver_grupo_cb, args=(g,)):
                    elegidos.append(g)

        if st.button("Iniciar Test", disabled=not elegidos):
            with db(DB_TESTS) as conn:
                todas_preg = conn.execute(
                    "SELECT id, numero, enunciado, opcion_a, opcion_b, opcion_c, opcion_d, correcta, bloque, "
                    "COALESCE(omitida, 0) FROM preguntas"
                ).fetchall()
            todas_preg = [f[:8] for f in todas_preg if _grupo_test(f[8], f[9]) in elegidos]
            random.shuffle(todas_preg)
            todas_preg, _repes = quitar_preguntas_repetidas(todas_preg)
            st.session_state.grupos_test_sel = elegidos
            st.session_state.test_global_activo = todas_preg
            st.session_state.idx_global = 0
            st.session_state.aciertos_global = 0
            st.session_state.fallos_global = 0
            st.session_state.omitidas_global = 0
            st.session_state.respondido_global = False
            st.session_state.elegido_global = None
            rerun_local()
        if not elegidos:
            st.caption("Marca al menos un grupo para empezar.")
        mostrar_panel_grupo(st.session_state.vista_grupo_test, destinos)
        return

    c_grp, c_fin = st.columns([5, 1.6])
    with c_grp:
        st.caption("🗂️ Grupos: " + " · ".join(st.session_state.get("grupos_test_sel", [])))
    with c_fin:
        if st.button("⏹ Cambiar grupos", key="cambiar_grupos_test", use_container_width=True):
            del st.session_state.test_global_activo
            rerun_local()

    preguntas = st.session_state.test_global_activo
    idx = st.session_state.idx_global

    if idx >= len(preguntas):
        st.balloons()
        st.success("¡Has terminado el test!")
        st.metric("Aciertos", st.session_state.aciertos_global)
        st.metric("Fallos", st.session_state.fallos_global)
        st.metric("Omitidas", st.session_state.get("omitidas_global", 0))
        mostrar_estadisticas_sesion()
        if st.button("Reiniciar Test"):
            del st.session_state.test_global_activo
            rerun_local()
        return

    p_id, p_num, enunciado, op_a, op_b, op_c, op_d, correcta = preguntas[idx]

    st.progress(idx / len(preguntas), text=f"Pregunta {idx+1} de {len(preguntas)}")

    with db(DB_TESTS) as conn:
        fila_origen = conn.execute(
            "SELECT t.titulo, p.bloque, COALESCE(p.omitida, 0) FROM preguntas p "
            "JOIN tests t ON t.id = p.test_id WHERE p.id=?", (p_id,)
        ).fetchone()
    origen = fila_origen[0] if fila_origen and fila_origen[0] else "Examen desconocido"
    bloque_preg = fila_origen[1] if fila_origen and fila_origen[1] else ""
    es_omitida = bool(fila_origen and fila_origen[2])
    partes_origen = [origen] + ([bloque_preg] if bloque_preg else []) + [f"Pregunta nº {p_num}"]
    if es_omitida:
        partes_origen.append("🗑️ Omitida")
    st.caption("📚 " + " · ".join(partes_origen))
    mostrar_estadisticas_sesion()

    st.markdown(_enunciado_md(enunciado))

    opciones_dict = {}
    if op_a:
        opciones_dict["A"] = f"a) {op_a}"
    if op_b:
        opciones_dict["B"] = f"b) {op_b}"
    if op_c:
        opciones_dict["C"] = f"c) {op_c}"
    if op_d:
        opciones_dict["D"] = f"d) {op_d}"

    key_eleccion = f"elec_{p_id}"
    if key_eleccion not in st.session_state:
        st.session_state[key_eleccion] = None

    for letra, texto in opciones_dict.items():
        c1, c2 = st.columns([0.08, 0.92])
        with c1:
            elegido = st.session_state[key_eleccion] == letra
            label = f"🔴 {letra}" if elegido else f"⚪ {letra}"
            if st.button(label, key=f"btn_{p_id}_{letra}", use_container_width=True):
                st.session_state[key_eleccion] = letra
                rerun_local()
        with c2:
            st.markdown(
                f"<div style='padding-top:6px'><b>{html.escape(texto)}</b></div>",
                unsafe_allow_html=True,
            )

    eleccion = st.session_state[key_eleccion]
    st.markdown("")

    if st.session_state.respondido_global:
        if st.session_state.elegido_global == correcta:
            st.success(f"¡Correcto! Era la {correcta}")
        else:
            st.error(f"Incorrecto. Marcaste {st.session_state.elegido_global}, era la {correcta}")

    with st.container(key="fila_acciones"):
        (
            col_omitir, col_chk, col_input, col_add, col_edit, col_elim, col_resolver,
        ) = st.columns([1, 4, 3, 1, 1, 1, 2], gap="small")  # ⏩1 · Responder4 · Término3 · ➕1 · ✏️1 · 🗑️1 · Resolver2

    def pasar_a_siguiente():
        st.session_state.idx_global += 1
        st.session_state.respondido_global = False
        st.session_state.elegido_global = None
        st.session_state[key_eleccion] = None
        rerun_local()

    with col_omitir:
        if st.button("⏩", key=f"omit_{p_id}", use_container_width=True,
                     help=f"Omitir y mandar la pregunta al grupo «{GRUPO_LEYES}»"):
            if not st.session_state.respondido_global:  # solo cuenta como omitida si no la has respondido
                st.session_state.omitidas_global = st.session_state.get("omitidas_global", 0) + 1
            if not es_omitida:  # las eliminadas se quedan en Omitidas
                mover_pregunta(p_id, GRUPO_LEYES)
                st.toast(f"⚖️ Pregunta enviada a «{GRUPO_LEYES}»")
            pasar_a_siguiente()

    with col_chk:
        if not st.session_state.respondido_global:
            if st.button("Responder", key=f"chk_{p_id}", use_container_width=True):
                if eleccion is None:
                    st.warning("Selecciona una opción primero.")
                else:
                    st.session_state.respondido_global = True
                    st.session_state.elegido_global = eleccion
                    if eleccion == correcta:
                        st.session_state.aciertos_global += 1
                    else:
                        st.session_state.fallos_global += 1
                    rerun_local()
        else:
            if st.button("Siguiente ➡️", key=f"next_{p_id}", use_container_width=True):
                pasar_a_siguiente()

    with col_input:
        nuevo_termino = st.text_input(
            "Añadir término",
            key=f"input_{p_id}",
            placeholder="Término",
            label_visibility="collapsed",
        ).strip().lower()

    with col_add:
        st.button(
            "➕", key=f"add_{p_id}", use_container_width=True, help="Añadir al glosario",
            on_click=_anadir_termino_cb, args=(p_id,),
        )

    with col_edit:
        key_edit = f"edit_{p_id}"
        if key_edit not in st.session_state:
            st.session_state[key_edit] = False
        icon = "❌" if st.session_state[key_edit] else "✏️"
        if st.button(icon, key=f"btn_edit_{p_id}", use_container_width=True):
            st.session_state[key_edit] = not st.session_state[key_edit]
            rerun_local()

    with col_elim:
        etiqueta_elim = "♻️" if es_omitida else "🗑️"
        ayuda_elim = (
            "Devuelve la pregunta a su grupo original" if es_omitida
            else f"No volverá a aparecer: pasa al grupo «{GRUPO_OMITIDAS}»"
        )
        if st.button(etiqueta_elim, key=f"elim_{p_id}", use_container_width=True, help=ayuda_elim):
            ids_afectados = marcar_omitida(p_id, not es_omitida)
            lista = st.session_state.test_global_activo
            antes = sum(1 for f in lista[:idx] if f[0] in ids_afectados)
            st.session_state.test_global_activo = [f for f in lista if f[0] not in ids_afectados]
            st.session_state.idx_global = idx - antes  # ya apunta a la siguiente pregunta
            st.session_state.respondido_global = False
            st.session_state.elegido_global = None
            st.session_state[key_eleccion] = None
            st.toast(
                "♻️ Pregunta restaurada en su grupo" if es_omitida
                else f"🗑️ Pregunta eliminada: pasa al grupo «{GRUPO_OMITIDAS}»"
            )
            rerun_local()

    texto_pregunta = formatear_pregunta(enunciado, op_a, op_b, op_c, op_d)
    job_ia = job_activo(f"job_ia_{p_id}")

    with col_resolver:
        if st.button("✅ Resolver", key=f"resolver_{p_id}", use_container_width=True, disabled=job_ia is not None):
            cacheada = cargar_explicacion(p_id)
            if cacheada and "\n- " in cacheada:
                st.session_state[f"ia_{p_id}"] = ("resolver", cacheada)
                rerun_local()
            else:
                prompt = (
                    f"Pregunta:\n{texto_pregunta}\n\n"
                    f"La opción correcta es la {correcta}. Responde EXACTAMENTE con este formato:\n"
                    f"**La opción {correcta}** (`texto breve de la opción`) es la correcta porque ... "
                    "(1-2 frases).\n\n"
                    "Las demás fallan:\n"
                    "- **X** (`texto breve`): motivo en una frase.\n"
                    "(una viñeta por CADA opción incorrecta, cada una en su propia línea, sin juntarlas "
                    "en un párrafo).\n"
                    "Pon entre `comillas invertidas` los términos técnicos, comandos y nombres. "
                    "Sin introducción ni conclusión."
                )
                job = Job("ia", subtipo="resolver", p_id=p_id)
                st.session_state[f"job_ia_{p_id}"] = job
                get_executor_test().submit(trabajo_ia_test, job, prompt, TOKENS_RESOLVER, get_client(), p_id)
                st.rerun()

    job_ia = job_activo(f"job_ia_{p_id}")
    seguimiento_activo = job_ia is not None and job_ia.subtipo == "seguimiento"
    if job_ia is not None and not seguimiento_activo:
        st.markdown("---")
        if job_ia.subtipo == "pista":
            st.markdown("### 💡 Pistas")
            st.markdown(job_ia.texto or "⏳ Generando...")
        else:
            with st.expander("✅ Explicación", expanded=True):
                st.markdown(_negrita_parentesis(job_ia.texto) or "⏳ Generando...")
    elif f"ia_{p_id}" in st.session_state:
        tipo, texto = st.session_state[f"ia_{p_id}"]
        st.markdown("---")
        if tipo == "error":
            st.error(texto)
        elif tipo == "pista":
            st.markdown("### 💡 Pistas")
            st.info(texto)
        else:
            with st.expander("✅ Explicación", expanded=True):
                st.success(_negrita_parentesis(texto))
                seccion_seguimiento(p_id, texto_pregunta, correcta, texto, job_ia)

    if st.session_state.get(f"edit_{p_id}", False):
        st.markdown("---")
        with st.form(key=f"form_edit_{p_id}"):
            nuevo_enun = st.text_area("Enunciado", value=enunciado, height=100)
            nueva_a = st.text_input("Opción A", value=op_a or "")
            nueva_b = st.text_input("Opción B", value=op_b or "")
            nueva_c = st.text_input("Opción C", value=op_c or "")
            nueva_d = st.text_input("Opción D", value=op_d or "")
            ops = ["A", "B"] + (["C"] if nueva_c else []) + (["D"] if nueva_d else [])
            try:
                idx_c = ops.index((correcta or "A").upper())
            except ValueError:
                idx_c = 0
            nueva_corr = st.selectbox("Respuesta correcta", ops, index=idx_c)
            if st.form_submit_button("💾 Guardar cambios"):
                with db(DB_TESTS) as conn:
                    conn.execute(
                        """UPDATE preguntas SET enunciado=?, opcion_a=?, opcion_b=?,
                           opcion_c=?, opcion_d=?, correcta=? WHERE id=?""",
                        (nuevo_enun, nueva_a, nueva_b, nueva_c, nueva_d, nueva_corr, p_id),
                    )
                borrar_explicacion(p_id)
                st.session_state.pop(f"ia_{p_id}", None)
                st.session_state.pop(f"chat_ia_{p_id}", None)
                st.session_state.test_global_activo[idx] = (
                    p_id, p_num, nuevo_enun, nueva_a, nueva_b, nueva_c, nueva_d, nueva_corr,
                )
                st.session_state[f"edit_{p_id}"] = False
                st.toast("Pregunta actualizada")
                rerun_local()

    texto_total = f"{enunciado} {op_a or ''} {op_b or ''} {op_c or ''} {op_d or ''}".lower()
    patron = patron_glosario(frozenset(st.session_state.glosario_externo))
    palabras_encontradas = []
    if patron:
        palabras_encontradas = list(dict.fromkeys(patron.findall(texto_total)))
    # los términos que acabas de añadir en esta pregunta salen siempre (primero), estén o no en su texto
    extras = [t for t in st.session_state.get(f"glos_extra_{p_id}", []) if t in st.session_state.glosario_externo]
    palabras_encontradas = list(dict.fromkeys(extras + palabras_encontradas))[:8]

    if palabras_encontradas:
        st.caption("💡 Términos del glosario encontrados:")
        cols = st.columns(min(len(palabras_encontradas), 6))
        for i, palabra in enumerate(palabras_encontradas):
            with cols[i % len(cols)]:
                c_chip, c_x = st.columns([4, 1], gap="small")
                with c_chip:
                    if st.button(f"🔍 {palabra.upper()}", key=f"glos_{p_id}_{i}", use_container_width=True):
                        st.session_state[f"inline_f1_{p_id}"] = palabra
                        st.session_state[f"inline_f2_{p_id}"] = ""
                        rerun_local()
                with c_x:
                    with st.popover("✖", help=f"Quitar «{palabra.upper()}» del glosario", use_container_width=True):
                        st.warning(
                            f"Se quitará «{palabra}» de glosario.txt y se borrarán de apuntes.db "
                            "los apuntes cuyo título sea exactamente ese término."
                        )
                        if st.button("Confirmar borrado", key=f"glos_del_{p_id}_{i}"):
                            n_borrados = quitar_del_glosario(palabra)
                            st.session_state[f"glos_extra_{p_id}"] = [
                                t for t in st.session_state.get(f"glos_extra_{p_id}", []) if t != palabra
                            ]
                            if st.session_state.get(f"inline_f1_{p_id}", "").strip().lower() == palabra:
                                st.session_state[f"inline_f1_{p_id}"] = ""
                            st.toast(f"«{palabra}» eliminado ({n_borrados} apunte(s) borrado(s))")
                            rerun_local()

    if f"inline_f1_{p_id}" not in st.session_state:
        st.session_state[f"inline_f1_{p_id}"] = ""
    if f"inline_f2_{p_id}" not in st.session_state:
        st.session_state[f"inline_f2_{p_id}"] = ""

    st.markdown("---")
    st.markdown("🔍 **Buscador de Apuntes**")
    st.caption(AYUDA_NEGRITA)
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Campo 1 (Título)", key=f"inline_f1_{p_id}")
    with c2:
        st.text_input("Campo 2 (Contenido)", key=f"inline_f2_{p_id}")

    busq1 = st.session_state[f"inline_f1_{p_id}"].strip().lower()
    busq2 = st.session_state[f"inline_f2_{p_id}"].strip().lower()

    if busq1 or busq2:
        encontrados = buscar_apuntes(busq1, busq2)
        if not encontrados:
            st.warning("No se encontraron apuntes.")
        else:
            if len(encontrados) > MAX_RESULTADOS_APUNTES:
                st.caption(f"Mostrando {MAX_RESULTADOS_APUNTES} de {len(encontrados)} resultados. Afina la búsqueda.")
            for bid, etq, cont in encontrados[:MAX_RESULTADOS_APUNTES]:
                mostrar_bloque(bid, etq, cont, f"ap_{p_id}", nivel="**")

        titulo_nuevo = st.session_state[f"inline_f1_{p_id}"].strip()
        with st.expander("➕ Añadir apunte nuevo", expanded=not encontrados):
            formulario_nuevo_apunte(f"test_{p_id}", titulo_nuevo)


def render_test():
    st.fragment(run_every=1 if hay_jobs(PREF_JOBS_TEST) else None)(zona_test)()


with tab_hacer_test:
    render_test()


# ==========================================
# PESTAÑA 2 - ESTUDIA CON LA IA
# ==========================================
TITULO_TEST_INTERESANTES = "⭐ Preguntas interesantes (IA)"


def agregar_pregunta_interesante(enunciado: str, opciones: str, correcta: str) -> bool:
    letras = {"A": "", "B": "", "C": "", "D": ""}
    for linea in opciones.split("\n"):
        m = re.match(r"\s*([A-Da-d])\)\s*(.*)", linea)
        if m:
            letras[m.group(1).upper()] = m.group(2).strip()

    with db(DB_TESTS) as conn:
        fila = conn.execute(
            "SELECT id FROM tests WHERE titulo=?", (TITULO_TEST_INTERESANTES,)
        ).fetchone()
        if fila:
            test_id = fila[0]
        else:
            test_id = conn.execute(
                "INSERT INTO tests (titulo, pdf_examen_nombre, pdf_respuestas_nombre) VALUES (?, ?, ?)",
                (TITULO_TEST_INTERESANTES, "generado por IA", None),
            ).lastrowid

        if conn.execute(
            "SELECT 1 FROM preguntas WHERE test_id=? AND enunciado=? AND opcion_a=? AND opcion_b=?",
            (test_id, enunciado, letras["A"], letras["B"]),
        ).fetchone():
            return False

        numero = conn.execute(
            "SELECT COALESCE(MAX(numero), 0) + 1 FROM preguntas WHERE test_id=?", (test_id,)
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO preguntas (test_id, numero, enunciado, opcion_a, opcion_b, opcion_c, opcion_d, correcta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (test_id, numero, enunciado, letras["A"], letras["B"], letras["C"], letras["D"], correcta),
        )
    return True


def _html_texto(txt: str) -> str:
    t = html.escape(txt or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t.replace("\n", "<br>")


def tarjeta_pregunta_html(num, p, resp, explicacion, pendiente):
    acertada = resp == p["correcta"]
    if acertada:
        bg, borde, color, titulo = "#d4edda", "#c3e6cb", "#155724", f"✅ Pregunta {num} · Correcta"
    else:
        bg, borde, color, titulo = "#f8d7da", "#f5c6cb", "#721c24", f"❌ Pregunta {num} · Incorrecta"

    texto_resp = _texto_opcion(p["opciones"], resp) if resp else ""
    mi_resp = f"{resp}) {texto_resp}" if resp else "en blanco"

    cuerpo = f"<div style='font-weight:600; margin-bottom:6px;'>{_html_texto(p['enunciado'])}</div>"
    if acertada:
        cuerpo += f"<div>Tu respuesta: <b>{_html_texto(mi_resp)}</b> — ¡muy bien!</div>"
    else:
        cuerpo += f"<div style='margin-bottom:6px;'>{_html_texto(p['opciones'])}</div>"
        cuerpo += (
            f"<div>Tu respuesta: <b>{_html_texto(mi_resp)}</b><br>"
            f"Correcta: <b>{_html_texto(p['correcta'] + ') ' + _texto_opcion(p['opciones'], p['correcta']))}</b></div>"
        )
        if explicacion:
            cuerpo += (
                f"<div style='margin-top:8px; padding-top:8px; border-top:1px solid {borde};'>"
                f"{_html_texto(explicacion)}</div>"
            )
        elif pendiente:
            cuerpo += "<div style='margin-top:8px;'>⏳ Pendiente de corrección...</div>"

    return (
        f"<div style='background-color:{bg}; border:1px solid {borde}; color:{color}; "
        f"padding:14px 16px; border-radius:8px; margin-bottom:18px;'>"
        f"<h4 style='margin:0 0 8px 0; color:{color};'>{titulo}</h4>{cuerpo}</div>"
    )


def finalizar_examen(t_id, examen):
    preguntas = examen["preguntas"]
    nota = sum(1 for p, r in zip(preguntas, examen["respuestas"]) if r == p["correcta"])
    examen["terminado"] = True
    examen["nota"] = nota
    guardar_examen(t_id, examen)

    estado = "Aprobado" if nota >= NOTA_APROBADO else "Mejorar"
    with db(DB_CURSO_IA) as conn:
        conn.execute(
            "UPDATE temas_estudio SET estado=?, aciertos_seguidoss=?, feedback_actual='' WHERE id=?",
            (estado, nota, t_id),
        )
        fila_t = conn.execute("SELECT temario_id FROM temas_estudio WHERE id=?", (t_id,)).fetchone()
    if fila_t and fila_t[0]:
        try:
            with db(DB_TEMARIO) as conn_t:
                conn_t.execute("UPDATE temario SET estado=?, aciertos=? WHERE id=?", (estado, nota, fila_t[0]))
        except sqlite3.Error:
            pass

    if nota < len(preguntas):
        job = Job("corr", tema_id=t_id)
        st.session_state[f"job_corr_{t_id}"] = job
        get_executor_tutor().submit(trabajo_corregir_examen, job, t_id, examen, get_client())


def _num_bloque(nombre):
    m = re.search(r"BLOQUE\s*(\d+)", nombre, flags=re.IGNORECASE)
    return (1, int(m.group(1))) if m else (0, 0)


def _reintentar_tema(t_id, bloque, grupo, nombre):
    """Devuelve un tema aprobado a 'Mejorar' y lo deja seleccionado en los desplegables."""
    with db(DB_CURSO_IA) as conn:
        conn.execute("UPDATE temas_estudio SET estado='Mejorar' WHERE id=?", (t_id,))
        fila = conn.execute("SELECT temario_id FROM temas_estudio WHERE id=?", (t_id,)).fetchone()
    if fila and fila[0]:
        try:
            with db(DB_TEMARIO) as conn_t:
                conn_t.execute("UPDATE temario SET estado='Mejorar' WHERE id=?", (fila[0],))
        except sqlite3.Error:
            pass
    st.session_state["sel_bloque_tutor"] = bloque
    st.session_state["sel_grupo_tutor"] = grupo
    st.session_state["sel_tema_tutor"] = nombre


def zona_estudio():
    if recoger_jobs_tutor():
        st.rerun()

    st.header("🧠 Tutor Personal - Oposición TAI (C1)")

    with db(DB_CURSO_IA) as conn:
        temas_db = conn.execute(
            "SELECT id, bloque, grupo, nombre_tema, estado, aciertos_seguidoss, prompt_personalizado FROM temas_estudio"
        ).fetchall()
        pendientes = {
            r[0] for r in conn.execute(
                "SELECT id FROM temas_estudio WHERE examen_json LIKE '%\"terminado\": true%'"
            ).fetchall()
        }

    if not temas_db:
        st.warning("No hay temas todavía. Ve a la pestaña «📋 Temario TAI» y pulsa «Recargar temario» para cargar tu `temario_tai.txt`.")
        return

    aprobados = [t for t in temas_db if t[4] == 'Aprobado' and t[0] not in pendientes]
    with st.expander(f"✅ Aprobados ({len(aprobados)})", expanded=False):
        if not aprobados:
            st.caption("Todavía no hay temas aprobados.")
        arbol_ap = {}
        for tid, blq, grp, nom, est, acs, _p in aprobados:
            arbol_ap.setdefault(blq or "Sin bloque", {}).setdefault(grp or "Sin grupo", []).append((tid, nom, acs))
        for blq in sorted(arbol_ap, key=_num_bloque, reverse=True):
            st.markdown(f"#### 📂 {_md(blq)}")
            for grp, temas_ap in arbol_ap[blq].items():
                st.markdown(f"**📁 {_md(grp)}**")
                for tid, nom, acs in temas_ap:
                    c_tema, c_btn = st.columns([5, 1.6])
                    with c_tema:
                        nota_txt = f" · nota {acs}/{NUM_PREGUNTAS_EXAMEN}" if acs else ""
                        st.markdown(f"📄 {_md(nom)}{nota_txt}")
                    with c_btn:
                        st.button(
                            "🔁 Volver a intentar", key=f"retry_{tid}",
                            on_click=_reintentar_tema, args=(tid, blq, grp, nom),
                        )

    mejorar = [t for t in temas_db if t[4] == 'Mejorar' or t[0] in pendientes]

    st.markdown("### 🔴 Temas para mejorar")

    if not mejorar:
        st.success("¡Enhorabuena! Todos los temas están aprobados.")
        return

    arbol = {}
    for tid, blq, grp, nom, est, acs, prompt_p in mejorar:
        if blq and grp and nom:
            arbol.setdefault(blq, {}).setdefault(grp, []).append((tid, nom, acs, prompt_p))

    bloques_disp = sorted(arbol.keys(), key=_num_bloque, reverse=True)
    if not bloques_disp:
        st.info("No hay bloques válidos para mostrar.")
        return

    bloque_sel = st.selectbox("📂 Selecciona el Bloque:", bloques_disp, key="sel_bloque_tutor")

    grupos_disp = list(arbol[bloque_sel].keys())
    grupo_sel = st.selectbox("📁 Selecciona el Grupo:", grupos_disp, key="sel_grupo_tutor")

    temas_en_grupo = arbol[bloque_sel][grupo_sel]
    temas_dict = {t[1]: t for t in temas_en_grupo}
    tema_nombre_sel = st.selectbox("📄 Selecciona el Tema:", list(temas_dict.keys()), key="sel_tema_tutor")

    tema_seleccionado = temas_dict[tema_nombre_sel]
    t_id, t_nombre, t_aciertos, t_prompt = tema_seleccionado

    st.markdown("---")
    st.subheader(f"🎯 Tema Activo: **{t_nombre}**")
    st.write(f"Última nota: **{t_aciertos}/{NUM_PREGUNTAS_EXAMEN}** · Para aprobar: **{NOTA_APROBADO}/{NUM_PREGUNTAS_EXAMEN}** o más")

    err_tema = st.session_state.pop(f"err_{t_id}", None)
    if err_tema:
        st.error(err_tema)

    with db(DB_CURSO_IA) as conn:
        fila_fb = conn.execute("SELECT feedback_actual FROM temas_estudio WHERE id=?", (t_id,)).fetchone()
    f_act = fila_fb[0] if fila_fb and fila_fb[0] else ""

    examen = cargar_examen(t_id)
    key_job_exam = f"job_exam_{t_id}"
    key_job_corr = f"job_corr_{t_id}"
    job_x = job_activo(key_job_exam)
    job_c = job_activo(key_job_corr)

    if job_x is not None:
        total = job_x.total or NUM_PREGUNTAS_EXAMEN
        st.progress(
            min(job_x.progreso / total, 1.0),
            text=f"⏳ Generando examen... {job_x.progreso}/{total} preguntas (puede tardar un poco)",
        )

    elif examen is None:
        if st.button(f"📝 Hacer Examen ({NUM_PREGUNTAS_EXAMEN} preguntas)", key=f"btn_exam_{t_id}"):
            job = Job("exam", tema_id=t_id)
            job.total = NUM_PREGUNTAS_EXAMEN
            st.session_state[key_job_exam] = job
            get_executor_tutor().submit(trabajo_examen, job, t_id, get_client())
            st.rerun()

    elif not examen.get("terminado"):
        preguntas = examen["preguntas"]
        n = len(preguntas)
        idx = examen["idx"]
        p = preguntas[idx]

        st.progress(idx / n, text=f"Pregunta {idx + 1} de {n}")
        st.markdown("\n\n".join(f"**{x.strip()}**" for x in re.split(r"\n\s*\n", p["enunciado"]) if x.strip()))

        ops_lista = [op.strip() for op in p["opciones"].split("\n") if op.strip()]
        eleccion = st.radio(
            "Selecciona tu respuesta:", ops_lista, index=None, key=f"radio_exam_{t_id}_{idx}"
        )

        col_r, col_i, col_c, _ = st.columns([1.2, 1.4, 1.2, 2.2])
        with col_r:
            etiqueta = "🏁 Responder y terminar" if idx + 1 >= n else "➡️ Responder"
            if st.button(etiqueta, key=f"btn_resp_{t_id}_{idx}", use_container_width=True):
                if eleccion is None:
                    st.warning("Selecciona una opción primero.")
                else:
                    examen["respuestas"][idx] = eleccion[0].upper()
                    if idx + 1 >= n:
                        finalizar_examen(t_id, examen)
                    else:
                        examen["idx"] = idx + 1
                        guardar_examen(t_id, examen)
                    st.rerun()
        with col_i:
            key_fav = f"fav_{t_id}_{idx}"
            ya_fav = st.session_state.get(key_fav, False)
            if st.button(
                "✅ Añadida al Test" if ya_fav else "⭐ Pregunta interesante",
                key=f"btn_fav_{t_id}_{idx}",
                use_container_width=True,
                disabled=ya_fav,
                help="Guarda esta pregunta en tu banco de Test Global",
            ):
                nueva = agregar_pregunta_interesante(p["enunciado"], p["opciones"], p["correcta"])
                st.session_state[key_fav] = True
                st.toast("⭐ Pregunta añadida al Test" if nueva else "Esa pregunta ya estaba en el Test")
                rerun_local()
        with col_c:
            if st.button("❌ Abandonar examen", key=f"btn_aband_{t_id}", use_container_width=True):
                guardar_examen(t_id, None)
                st.rerun()

    else:
        n = len(examen["preguntas"])
        nota = examen["nota"]
        st.metric("Nota", f"{nota}/{n}")
        if nota >= NOTA_APROBADO:
            st.success(f"🎉 ¡Aprobado! Has sacado {nota}/{n}. ¡Tema superado!")
        else:
            st.error(f"❌ Has sacado {nota}/{n}. Necesitas {NOTA_APROBADO} o más; el tema se queda en Mejorar.")

        if job_c is not None:
            st.info("⚡ La IA está corrigiendo tus errores...")
            explicaciones = {str(k): v for k, v in job_c.explicaciones.items()}
        else:
            explicaciones = examen.get("explicaciones", {})

        st.markdown("### 📋 Corrección del examen")
        for i, p in enumerate(examen["preguntas"]):
            st.markdown(
                tarjeta_pregunta_html(
                    i + 1, p, examen["respuestas"][i],
                    explicaciones.get(str(i)), pendiente=job_c is not None,
                ),
                unsafe_allow_html=True,
            )

        if st.button("✔️ Cerrar resultado", key=f"btn_cerrar_{t_id}", disabled=job_c is not None):
            guardar_examen(t_id, None)
            guardar_feedback_tema(t_id, "")
            st.rerun()

    st.markdown("---")
    with st.expander("➕ Gestionar Temario (Añadir o Modificar Bloques, Grupos y Temas)"):
        opcion_gestion = st.radio("Acción:", ["Añadir Nuevo Tema / Bloque / Grupo", "Modificar / Eliminar Tema Existente"], horizontal=True)

        if opcion_gestion == "Añadir Nuevo Tema / Bloque / Grupo":
            with st.form("form_add_temario"):
                st.markdown("**Añadir elemento al temario:**")

                with db(DB_CURSO_IA) as conn:
                    filas_bg = conn.execute("SELECT DISTINCT bloque, grupo FROM temas_estudio").fetchall()

                lista_bloques_existentes = sorted(list(set([f[0] for f in filas_bg if f[0]])))
                lista_grupos_existentes = sorted(list(set([f[1] for f in filas_bg if f[1]])))

                nuevo_bloque_sel = st.selectbox("Bloque:", ["-- Escribir nuevo bloque --"] + lista_bloques_existentes)
                custom_bloque = st.text_input("O escribe nuevo Bloque si seleccionaste '-- Escribir nuevo bloque --'")
                bloque_final = custom_bloque.strip() if nuevo_bloque_sel == "-- Escribir nuevo bloque --" else nuevo_bloque_sel

                nuevo_grupo_sel = st.selectbox("Grupo:", ["-- Escribir nuevo grupo --"] + lista_grupos_existentes)
                custom_grupo = st.text_input("O escribe nuevo Grupo si seleccionaste '-- Escribir nuevo grupo --'")
                grupo_final = custom_grupo.strip() if nuevo_grupo_sel == '-- Escribir nuevo grupo --' else nuevo_grupo_sel

                nombre_nuevo_tema = st.text_input("Nombre del Tema:")

                btn_guardar_nuevo = st.form_submit_button("📥 Guardar en el Temario")

                if btn_guardar_nuevo:
                    if not bloque_final or not grupo_final or not nombre_nuevo_tema:
                        st.error("El Bloque, el Grupo y el Nombre del Tema son obligatorios.")
                    else:
                        tema_limpio = nombre_nuevo_tema.strip()
                        prompt_def = f"Eres profesor de oposiciones TAI. Redacta una pregunta de examen técnica y rigurosa sobre el tema: {tema_limpio} ({grupo_final})."

                        with db(DB_CURSO_IA) as conn:
                            try:
                                conn.execute(
                                    """INSERT INTO temas_estudio (bloque, grupo, nombre_tema, estado, aciertos_seguidoss, prompt_personalizado)
                                       VALUES (?, ?, ?, 'Mejorar', 0, ?)""",
                                    (bloque_final, grupo_final, tema_limpio, prompt_def)
                                )
                                agregar_tema_a_txt(bloque_final, grupo_final, tema_limpio)

                                st.success(f"¡Añadido correctamente: {tema_limpio}!")
                                time.sleep(0.5)
                                rerun_local()
                            except sqlite3.IntegrityError:
                                st.warning("Ese tema ya existe en la base de datos.")

        else:
            with st.form("form_edit_temario"):
                st.markdown("**Modificar o Eliminar Tema:**")

                with db(DB_CURSO_IA) as conn:
                    todos_los_temas = conn.execute("SELECT id, bloque, grupo, nombre_tema FROM temas_estudio").fetchall()

                dict_temas_edit = {f"{t[3]} ({t[1]} > {t[2]})": t[0] for t in todos_los_temas}

                if not dict_temas_edit:
                    st.info("No hay temas para editar.")
                    tema_a_editar_id = None
                    st.form_submit_button("—", disabled=True)
                else:
                    tema_sel_str = st.selectbox("Selecciona el Tema a modificar/borrar:", list(dict_temas_edit.keys()))
                    tema_a_editar_id = dict_temas_edit[tema_sel_str]

                    t_actual_info = next((t for t in todos_los_temas if t[0] == tema_a_editar_id), None)

                    nuevo_nombre_tema = st.text_input("Nuevo Nombre del Tema:", value=t_actual_info[3] if t_actual_info else "")
                    nuevo_bloque_tema = st.text_input("Bloque:", value=t_actual_info[1] if t_actual_info else "")
                    nuevo_grupo_tema = st.text_input("Grupo:", value=t_actual_info[2] if t_actual_info else "")

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        btn_actualizar = st.form_submit_button("💾 Guardar Cambios")
                    with col_m2:
                        btn_borrar = st.form_submit_button("🗑️ Eliminar Tema")

                    if btn_actualizar and tema_a_editar_id:
                        with db(DB_CURSO_IA) as conn:
                            conn.execute(
                                "UPDATE temas_estudio SET nombre_tema=?, bloque=?, grupo=? WHERE id=?",
                                (nuevo_nombre_tema.strip(), nuevo_bloque_tema.strip(), nuevo_grupo_tema.strip(), tema_a_editar_id)
                            )
                        st.success("¡Tema actualizado con éxito!")
                        time.sleep(0.5)
                        rerun_local()

                    if btn_borrar and tema_a_editar_id:
                        with db(DB_CURSO_IA) as conn:
                            conn.execute("DELETE FROM temas_estudio WHERE id=?", (tema_a_editar_id,))
                        st.success("¡Tema eliminado con éxito!")
                        time.sleep(0.5)
                        rerun_local()

    st.markdown("---")
    with st.expander("⚙️ Modificar Prompt Personalizado para este Tema"):
        nuevo_prompt = st.text_area("Prompt de la IA:", value=t_prompt, height=120, key=f"prompt_txt_{t_id}")
        if st.button("💾 Guardar Prompt", key=f"save_prompt_{t_id}"):
            with db(DB_CURSO_IA) as conn:
                conn.execute("UPDATE temas_estudio SET prompt_personalizado=? WHERE id=?", (nuevo_prompt, t_id))
            st.success("¡Prompt actualizado correctamente!")
            st.rerun()


def render_estudio():
    st.fragment(run_every=1 if hay_jobs(PREF_JOBS_TUTOR) else None)(zona_estudio)()


with tab_estudia:
    render_estudio()


# ==========================================
# PESTAÑA 3 - APUNTES
# ==========================================
@st.fragment
def zona_apuntes():
    st.header("Buscador de Apuntes")
    st.caption(AYUDA_NEGRITA)
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.text_input("Campo 1 (Título)", key="ap_f1").strip().lower()
    with c2:
        f2 = st.text_input("Campo 2 (Contenido)", key="ap_f2").strip().lower()

    encontrados = buscar_apuntes(f1, f2)
    st.write(f"**Resultados:** {len(encontrados)}")
    if len(encontrados) > MAX_RESULTADOS_APUNTES:
        st.caption(f"Mostrando los primeros {MAX_RESULTADOS_APUNTES} resultados.")

    with st.expander("➕ Añadir término nuevo", expanded=bool((f1 or f2) and not encontrados)):
        formulario_nuevo_apunte("apuntes", st.session_state.get("ap_f1", "").strip())

    for bid, etq, cont in encontrados[:MAX_RESULTADOS_APUNTES]:
        mostrar_bloque(bid, etq, cont, "ap2")


with tab_apuntes:
    zona_apuntes()


# ==========================================
# PESTAÑA 4 - SUBIR Y GESTIONAR EXÁMENES
# ==========================================
@st.fragment
def zona_subir_examenes():
    st.header("➕ Subir y Gestionar Exámenes Oficiales")
    st.markdown("Sube exámenes en PDF y opcionalmente su plantilla de respuestas oficiales.")

    resumen = st.session_state.pop("resumen_subida", None)
    if resumen:
        st.success(resumen["ok"])
        if resumen["sin"]:
            st.warning(
                f"No se encontró respuesta en la plantilla para {len(resumen['sin'])} pregunta(s) "
                f"(se han dejado como «A», revísalas): " + ", ".join(resumen["sin"][:15])
                + ("…" if len(resumen["sin"]) > 15 else "")
            )

    with st.form("form_subir_examen"):
        titulo_examen = st.text_input("Título del Examen (ej. TAI 2023 - Promoción Interna)")
        archivo_examen = st.file_uploader("PDF del Examen", type=["pdf"], key="pdf_ex")
        archivo_respuestas = st.file_uploader("PDF de Plantilla de Respuestas (Opcional)", type=["pdf"], key="pdf_res")
        subido = st.form_submit_button("📥 Procesar y Guardar Examen")

    if subido:
        if not titulo_examen or not archivo_examen:
            st.warning("Debes indicar un título y adjuntar al menos el PDF del examen.")
            return

        with st.spinner("Procesando PDFs..."):
            texto_examen = extraer_texto_pdf(archivo_examen)
            preguntas_extraidas = extraer_preguntas_del_pdf(texto_examen)

            plantilla_respuestas = {}
            if archivo_respuestas:
                texto_respuestas = extraer_texto_pdf(archivo_respuestas)
                plantilla_respuestas = extraer_plantilla_respuestas_pdf(texto_respuestas)

            if not preguntas_extraidas:
                st.error("No se han podido extraer preguntas estructuradas del PDF del examen. Revisa el formato.")
                return

            with db(DB_TESTS) as conn:
                previos = [" ".join(f) for f in conn.execute(
                    "SELECT enunciado, opcion_a, opcion_b, opcion_c, opcion_d FROM preguntas").fetchall()]
            vocab = construir_vocabulario(previos + [
                " ".join([p["enunciado"], p["opcion_a"], p["opcion_b"], p["opcion_c"], p["opcion_d"]])
                for p in preguntas_extraidas
            ])
            for p in preguntas_extraidas:
                for campo in ("enunciado", "opcion_a", "opcion_b", "opcion_c", "opcion_d"):
                    p[campo] = reparar_palabras_partidas(p[campo], vocab)

            ya_existian = contar_ya_existentes(preguntas_extraidas)  # antes de guardar este examen

            with db(DB_TESTS) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO tests (titulo, pdf_examen_nombre, pdf_respuestas_nombre) VALUES (?, ?, ?)",
                    (titulo_examen, archivo_examen.name, archivo_respuestas.name if archivo_respuestas else None),
                )
                test_id = cursor.lastrowid

                datos_preguntas = []
                sin_respuesta = []
                por_bloque = {}
                for p in preguntas_extraidas:
                    num = p["numero"]
                    bl = p.get("bloque", "")
                    corr = buscar_respuesta(plantilla_respuestas, bl, num)
                    if corr is None:
                        corr = "A"
                        if plantilla_respuestas:
                            sin_respuesta.append(f"{bl or 'Sin bloque'} nº {num}")
                    por_bloque[bl or "Sin bloque"] = por_bloque.get(bl or "Sin bloque", 0) + 1
                    datos_preguntas.append((
                        test_id, num, p["enunciado"],
                        p["opcion_a"], p["opcion_b"], p["opcion_c"], p["opcion_d"], corr, bl
                    ))

                cursor.executemany(
                    """INSERT INTO preguntas (test_id, numero, enunciado, opcion_a, opcion_b, opcion_c, opcion_d, correcta, bloque)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    datos_preguntas,
                )

        desglose = " · ".join(f"{b}: {n}" for b, n in por_bloque.items())
        n_repes = ya_existian
        aviso_repes = (
            f" · {n_repes} ya estaban en otros exámenes (se guardan en este, pero no saldrán repetidas en el test)"
            if n_repes else ""
        )
        st.session_state["resumen_subida"] = {
            "ok": f"¡Examen guardado! {len(preguntas_extraidas)} preguntas importadas → {desglose}{aviso_repes}",
            "sin": sin_respuesta,
        }
        rerun_local()

    st.markdown("---")
    if st.button("🔧 Reparar palabras partidas en las preguntas ya guardadas", key="btn_reparar_palabras"):
        with db(DB_TESTS) as conn:
            filas = conn.execute(
                "SELECT id, enunciado, opcion_a, opcion_b, opcion_c, opcion_d FROM preguntas").fetchall()
            vocab = construir_vocabulario([" ".join(x or "" for x in f[1:]) for f in filas])
            cambiadas = 0
            for f in filas:
                nuevos = [reparar_palabras_partidas(x or "", vocab) if x else x for x in f[1:]]
                if list(f[1:]) != nuevos:
                    conn.execute(
                        "UPDATE preguntas SET enunciado=?, opcion_a=?, opcion_b=?, opcion_c=?, opcion_d=? WHERE id=?",
                        (*nuevos, f[0]),
                    )
                    cambiadas += 1
        st.success(f"Revisadas {len(filas)} preguntas; {cambiadas} corregidas.")
        st.session_state.pop("test_global_activo", None)  # el test en curso tenía el texto antiguo

    with db(DB_TESTS) as conn:
        n_pregs, max_id = conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM preguntas").fetchone()
    if n_pregs:
        n_rep = contar_repetidas_guardadas(n_pregs, max_id)
        if n_rep:
            st.caption(f"ℹ️ {n_rep} pregunta(s) repetida(s) entre exámenes (no saldrán repetidas en el test).")
    st.subheader("Exámenes guardados en el sistema")

    with db(DB_TESTS) as conn:
        tests_bd = conn.execute("SELECT id, titulo, pdf_examen_nombre, pdf_respuestas_nombre FROM tests").fetchall()

    if not tests_bd:
        st.info("No hay exámenes subidos todavía.")
    else:
        for tid, tit, p_ex, p_res in tests_bd:
            col_info, col_del = st.columns([0.85, 0.15])
            with col_info:
                st.markdown(f"**📚 {tit}** (Examen: `{p_ex or 'N/D'}` | Respuestas: `{p_res or 'Sin plantilla'}`)")
            with col_del:
                if st.button("🗑️ Borrar", key=f"del_test_{tid}2"):
                    with db(DB_TESTS) as conn:
                        conn.execute("DELETE FROM preguntas WHERE test_id=?", (tid,))
                        conn.execute("DELETE FROM tests WHERE id=?", (tid,))
                    st.toast("Examen eliminado.")
                    rerun_local()


with tab_agregar_test:
    zona_subir_examenes()


# ==========================================
# PESTAÑA 5 - TEMARIO TAI (BD intermedia Temario.db)
# ==========================================
# --- TEMARIO LOGIC START ---
ESTADOS_TEMA = ("Mejorar", "Aprobado")
SITUACIONES = ("Erróneo", "Nuevo", "Actualizado", "Sin cambio", "Huérfano")
ICONOS_SITUACION = {
    "Erróneo": "⚠️", "Nuevo": "🆕", "Actualizado": "✏️", "Sin cambio": "➖", "Huérfano": "👻",
}


def _clave(txt) -> str:
    return " ".join(str(txt or "").lower().split())


def _prompt_defecto_v1(titulo, descripcion, grupo):
    """Versión anterior del prompt por defecto (solo para reconocerlo y regenerarlo)."""
    linea = f"{titulo}: {descripcion}" if descripcion else titulo
    prompt = (
        "Eres profesor de oposiciones TAI. Redacta una pregunta de examen técnica y rigurosa. "
        f"TEMA GENERAL (el tema al que te refieres): {titulo} ({grupo}). "
        f"Línea completa del temario: «{linea}». "
    )
    if descripcion:
        prompt += f"La pregunta debe basarse en un aspecto concreto de estos contenidos: {descripcion}."
    return prompt


def prompt_tema_por_defecto(titulo, descripcion, grupo):
    """Prompt: línea completa, título como tema general y la descripción como base de las preguntas."""
    linea = f"{titulo}: {descripcion}" if descripcion else titulo
    prompt = (
        "Eres profesor de oposiciones TAI (nivel C1, Administración). Redacta una pregunta de examen tipo test "
        "clara y directa. "
        f"TEMA GENERAL (el tema al que te refieres): {titulo} ({grupo}). "
        f"Línea completa del temario: «{linea}». "
    )
    if descripcion:
        prompt += f"La pregunta debe tratar un dato o concepto concreto de esta lista: {descripcion}."
    return prompt


def _es_prompt_por_defecto(prompt, nombre_tema, descripcion, grupo):
    """True si el prompt es el autogenerado (no personalizado a mano) y por tanto se puede regenerar."""
    if not (prompt or "").strip():
        return True
    candidatos = {
        prompt_tema_por_defecto(nombre_tema, descripcion or "", grupo),
        _prompt_defecto_v1(nombre_tema, descripcion or "", grupo),
        f"Eres profesor de oposiciones TAI. Redacta una pregunta de examen técnica y rigurosa sobre el tema: {nombre_tema} ({grupo}).",
    }
    return prompt.strip() in candidatos


def parsear_temario_txt(texto: str):
    """Devuelve (validos, erroneos). Bloque = [..], grupo = '- Grupo ..', tema = 'Titulo: descripcion'."""
    validos, erroneos = [], []
    bloque = grupo = ""
    vistos = set()
    orden = 0
    for linea in texto.splitlines():
        s = linea.strip()
        if not s:
            continue
        if s.startswith("[") and "]" in s:
            bloque = s.replace("[", "").replace("]", "").strip()
            grupo = ""
            continue
        if re.match(r"^-\s*Grupo\b", s, flags=re.IGNORECASE):
            grupo = re.sub(r"^-\s*Grupo\s*", "Grupo ", s, flags=re.IGNORECASE).strip()
            continue
        if not bloque or not grupo:
            continue
        orden += 1
        cuerpo = re.sub(r"^Tema\b\s*", "", s)
        titulo, sep, desc = cuerpo.partition(":")
        titulo, desc = titulo.strip(), desc.strip()
        clave = (_clave(bloque), _clave(grupo), _clave(titulo))
        if sep and titulo and clave not in vistos:
            vistos.add(clave)
            validos.append({"bloque": bloque, "grupo": grupo, "titulo": titulo, "descripcion": desc, "orden": orden})
        else:
            erroneos.append({"bloque": bloque, "grupo": grupo, "linea": cuerpo, "orden": orden})
    return validos, erroneos


def _indice_legado():
    """Temas de CursoIA.db aún sin enlazar con Temario.db (formato antiguo: nombre = línea completa)."""
    with db(DB_CURSO_IA) as conn:
        filas = conn.execute(
            "SELECT id, bloque, grupo, nombre_tema, estado, aciertos_seguidoss, prompt_personalizado, descripcion "
            "FROM temas_estudio WHERE COALESCE(temario_id,0)=0"
        ).fetchall()
    idx = {}
    for fila in filas:
        clave = (_clave(fila[1]), _clave(fila[2]), _clave((fila[3] or "").split(":", 1)[0]))
        idx.setdefault(clave, fila)
    return idx


def recargar_temario(texto: str) -> dict:
    """Compara el txt con Temario.db, clasifica cada tema y sincroniza con CursoIA.db."""
    validos, erroneos = parsear_temario_txt(texto)
    legado = _indice_legado()

    with db(DB_TEMARIO) as conn:
        filas = conn.execute(
            "SELECT id, bloque, grupo, titulo, descripcion, linea_original FROM temario"
        ).fetchall()
        por_clave = {(_clave(b), _clave(g), _clave(t)): (i, d) for i, b, g, t, d, _ in filas if t}
        por_linea = {(_clave(b), _clave(g), _clave(lo)): i for i, b, g, t, d, lo in filas if lo}
        vistos = set()

        for v in validos:
            clave = (_clave(v["bloque"]), _clave(v["grupo"]), _clave(v["titulo"]))
            if clave in por_clave:
                i, desc_ant = por_clave[clave]
                vistos.add(i)
                if _clave(desc_ant) != _clave(v["descripcion"]):
                    conn.execute(
                        "UPDATE temario SET descripcion=?, descripcion_anterior=?, situacion='Actualizado', orden=? WHERE id=?",
                        (v["descripcion"], desc_ant, v["orden"], i),
                    )
                else:
                    conn.execute(
                        "UPDATE temario SET situacion='Sin cambio', descripcion_anterior='', orden=? WHERE id=?",
                        (v["orden"], i),
                    )
            else:
                leg = legado.get(clave)
                estado, aciertos = (leg[4], leg[5]) if leg else ("Mejorar", 0)
                cur = conn.execute(
                    """INSERT INTO temario (bloque, grupo, titulo, descripcion, estado, aciertos, situacion, orden)
                       VALUES (?, ?, ?, ?, ?, ?, 'Nuevo', ?)""",
                    (v["bloque"], v["grupo"], v["titulo"], v["descripcion"], estado or "Mejorar", aciertos or 0, v["orden"]),
                )
                vistos.add(cur.lastrowid)

        for e in erroneos:
            i = por_linea.get((_clave(e["bloque"]), _clave(e["grupo"]), _clave(e["linea"])))
            if i:
                vistos.add(i)
                conn.execute(
                    "UPDATE temario SET situacion=CASE WHEN titulo<>'' THEN 'Sin cambio' ELSE 'Erróneo' END, orden=? WHERE id=?",
                    (e["orden"], i),
                )
            else:
                cur = conn.execute(
                    """INSERT INTO temario (bloque, grupo, titulo, descripcion, situacion, linea_original, orden)
                       VALUES (?, ?, '', '', 'Erróneo', ?, ?)""",
                    (e["bloque"], e["grupo"], e["linea"], e["orden"]),
                )
                vistos.add(cur.lastrowid)

        for i, b, g, t, d, lo in filas:
            if i in vistos:
                continue
            if not t:  # erróneo sin resolver que ya no está en el txt
                conn.execute("DELETE FROM temario WHERE id=?", (i,))
            else:
                conn.execute("UPDATE temario SET situacion='Huérfano', descripcion_anterior='' WHERE id=?", (i,))

        resumen = dict(conn.execute("SELECT situacion, COUNT(*) FROM temario GROUP BY situacion").fetchall())

    sincronizar_temario_con_estudio()
    return resumen


def sincronizar_temario_con_estudio():
    """Vuelca Temario.db en CursoIA.db (temas_estudio) conservando el estado/aciertos ya guardados."""
    with db(DB_TEMARIO) as conn_t:
        filas = conn_t.execute(
            "SELECT id, bloque, grupo, titulo, descripcion, estado FROM temario WHERE titulo<>''"
        ).fetchall()
    legado = _indice_legado()

    with db(DB_CURSO_IA) as conn:
        for tid, bloque, grupo, titulo, desc, estado in filas:
            row = conn.execute(
                "SELECT id, nombre_tema, prompt_personalizado, descripcion, grupo FROM temas_estudio WHERE temario_id=?",
                (tid,),
            ).fetchone()
            if row is None:
                leg = legado.pop((_clave(bloque), _clave(grupo), _clave(titulo)), None)
                if leg is not None:
                    row = (leg[0], leg[3], leg[6], leg[7], leg[2])
                    conn.execute("UPDATE temas_estudio SET temario_id=? WHERE id=?", (tid, leg[0]))

            if row is None:
                prompt = prompt_tema_por_defecto(titulo, desc, grupo)
                for nombre in (titulo, f"{titulo} ({grupo})"):
                    try:
                        conn.execute(
                            """INSERT INTO temas_estudio (bloque, grupo, nombre_tema, estado, aciertos_seguidoss,
                                                          prompt_personalizado, descripcion, temario_id)
                               VALUES (?, ?, ?, ?, 0, ?, ?, ?)""",
                            (bloque, grupo, nombre, estado, prompt, desc, tid),
                        )
                        break
                    except sqlite3.IntegrityError:
                        continue
                continue

            id_est, nombre_old, prompt_old, desc_old, grupo_old = row
            if _es_prompt_por_defecto(prompt_old, nombre_old, desc_old, grupo_old):
                prompt_nuevo = prompt_tema_por_defecto(titulo, desc, grupo)
            else:
                prompt_nuevo = prompt_old
            for nombre in dict.fromkeys((titulo, f"{titulo} ({grupo})", nombre_old)):
                try:
                    conn.execute(
                        """UPDATE temas_estudio SET bloque=?, grupo=?, nombre_tema=?, descripcion=?, estado=?,
                                                    prompt_personalizado=? WHERE id=?""",
                        (bloque, grupo, nombre, desc, estado, prompt_nuevo, id_est),
                    )
                    break
                except sqlite3.IntegrityError:
                    continue


def cambiar_estado_tema(tid: int, estado: str):
    if estado not in ESTADOS_TEMA:
        return
    with db(DB_TEMARIO) as conn:
        conn.execute("UPDATE temario SET estado=? WHERE id=?", (estado, tid))
    with db(DB_CURSO_IA) as conn:
        conn.execute("UPDATE temas_estudio SET estado=? WHERE temario_id=?", (estado, tid))


def borrar_tema_temario(tid: int):
    with db(DB_CURSO_IA) as conn:
        conn.execute("DELETE FROM temas_estudio WHERE temario_id=?", (tid,))
    with db(DB_TEMARIO) as conn:
        conn.execute("DELETE FROM temario WHERE id=?", (tid,))


def corregir_tema_erroneo(tid: int, linea_nueva: str):
    """Convierte una línea errónea en tema válido. Devuelve (ok, mensaje)."""
    titulo, sep, desc = linea_nueva.partition(":")
    titulo, desc = titulo.strip(), desc.strip()
    if not sep or not titulo:
        return False, "Falta el carácter «:» tras el título."
    with db(DB_TEMARIO) as conn:
        fila = conn.execute("SELECT bloque, grupo FROM temario WHERE id=?", (tid,)).fetchone()
        if not fila:
            return False, "El tema ya no existe."
        bloque, grupo = fila
        otros = conn.execute(
            "SELECT titulo FROM temario WHERE id<>? AND bloque=? AND grupo=? AND titulo<>''", (tid, bloque, grupo)
        ).fetchall()
        if any(_clave(o[0]) == _clave(titulo) for o in otros):
            return False, "Ya existe un tema con ese título en este grupo."
        leg = _indice_legado().get((_clave(bloque), _clave(grupo), _clave(titulo)))
        estado, aciertos = (leg[4], leg[5]) if leg else ("Mejorar", 0)
        conn.execute(
            """UPDATE temario SET titulo=?, descripcion=?, estado=?, aciertos=?, situacion='Nuevo' WHERE id=?""",
            (titulo, desc, estado or "Mejorar", aciertos or 0, tid),
        )
    sincronizar_temario_con_estudio()
    return True, "Tema corregido."
# --- TEMARIO LOGIC END ---


def _on_cambio_estado(tid: int, key: str):
    cambiar_estado_tema(tid, st.session_state.get(key, "Mejorar"))


def _fila_temario(f):
    tid, bloque, grupo, titulo, desc, desc_ant, estado, aciertos, situacion, linea_orig = f
    c_txt, c_est, c_del = st.columns([6, 2, 1])

    with c_txt:
        if situacion == "Erróneo":
            motivo = "título duplicado o vacío" if ":" in (linea_orig or "") else "falta «:»"
            st.caption(f"⚠️ {motivo}")
            nueva = st.text_input(
                "Línea (pon «Título: descripción»)", value=linea_orig, key=f"temario_err_{tid}",
                label_visibility="collapsed",
            )
        else:
            icono = "✅" if estado == "Aprobado" else "🔴"
            st.markdown(f"{icono} **{_md(titulo)}**: {_md(desc)}")
            if situacion == "Actualizado" and desc_ant:
                st.caption(f"Antes: {desc_ant}")
            if situacion == "Huérfano":
                st.caption("No aparece en el último txt cargado. Si es un tema renombrado, borra este.")

    with c_est:
        if situacion == "Erróneo":
            if st.button("💾 Guardar", key=f"temario_save_{tid}"):
                ok, msg = corregir_tema_erroneo(tid, nueva)
                if ok:
                    st.toast(msg)
                    rerun_local()
                else:
                    st.error(msg)
        else:
            key = f"temario_est_{tid}_{estado}"
            st.selectbox(
                "Estado", ESTADOS_TEMA, index=ESTADOS_TEMA.index(estado) if estado in ESTADOS_TEMA else 0,
                key=key, label_visibility="collapsed", on_change=_on_cambio_estado, args=(tid, key),
            )

    with c_del:
        with st.popover("🗑️"):
            st.warning("Se borra de Temario.db y de «Estudia con la IA» (incluida su puntuación).")
            if st.button("Confirmar borrado", key=f"temario_del_{tid}"):
                borrar_tema_temario(tid)
                st.toast("Tema eliminado.")
                rerun_local()


@st.fragment
def zona_temario():
    st.header("📋 Temario TAI")
    st.caption(
        "Temario.db guarda el estado de cada tema, independiente del txt. "
        "Al recargar un txt nuevo se detectan cambios sin perder lo aprobado."
    )

    c1, c2 = st.columns([4, 1])
    with c1:
        archivo = st.file_uploader("Selecciona temario_tai.txt", type=["txt"], key="temario_uploader")
    with c2:
        st.write("")
        st.write("")
        pulsado = st.button("🔄 Recargar temario", disabled=archivo is None, key="btn_recargar_temario")

    if pulsado and archivo is not None:
        raw = archivo.getvalue()
        try:
            texto = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = raw.decode("latin-1")
        resumen = recargar_temario(texto)
        st.session_state["temario_resumen"] = resumen
        rerun_local()

    with db(DB_TEMARIO) as conn:
        filas = conn.execute(
            """SELECT id, bloque, grupo, titulo, descripcion, descripcion_anterior, estado, aciertos, situacion, linea_original
               FROM temario ORDER BY orden, id"""
        ).fetchall()

    if not filas:
        st.info("El temario está vacío. Sube tu `temario_tai.txt` y pulsa «Recargar temario».")
        return

    validos = [f for f in filas if f[8] != "Erróneo"]
    aprobados = sum(1 for f in validos if f[6] == "Aprobado")
    st.markdown(f"**{aprobados} aprobados de {len(validos)} temas**")

    for situacion in SITUACIONES:
        items = [f for f in filas if f[8] == situacion]
        if not items:
            continue
        abierto = situacion in ("Erróneo", "Nuevo", "Actualizado")
        with st.expander(f"{ICONOS_SITUACION[situacion]} {situacion} ({len(items)})", expanded=abierto):
            arbol = {}
            for f in items:
                arbol.setdefault(f[1], {}).setdefault(f[2], []).append(f)
            for blq, grupos in arbol.items():
                st.markdown(f"#### 📂 {_md(blq)}")
                for grp, temas in grupos.items():
                    st.markdown(f"**📁 {_md(grp)}**")
                    for f in temas:
                        _fila_temario(f)
                    st.divider()


with tab_temario:
    zona_temario()
