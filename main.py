import logging
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, BackgroundTasks, Response
from fastapi.responses import JSONResponse
import httpx
from supabase import create_client, Client
import anthropic
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("copilot")

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")

app = FastAPI(title="BIA AI Trading Copilot")

# Lazy-initialized clients — created on first use so a missing env var
# doesn't crash the process before Slack's challenge can be answered.
_supabase: Client | None = None
_claude: anthropic.Anthropic | None = None


@app.on_event("startup")
async def _ensure_tables():
    """Create required tables if they don't exist yet."""
    try:
        sb = get_supabase()
        sb.rpc("query", {"sql": (
            "CREATE TABLE IF NOT EXISTS processed_events ("
            "  event_ts TEXT PRIMARY KEY,"
            "  processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
            ");"
        )}).execute()
        log.info("STARTUP: processed_events table ensured")
    except Exception as e:
        # Log but don't crash — table may already exist or RPC may be unavailable
        log.warning("STARTUP: could not ensure processed_events table: %s", e)


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_claude() -> anthropic.Anthropic:
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _claude


# ── Persistent deduplication ──────────────────────────────────────────────────

def _is_duplicate(event_ts: str) -> bool:
    """
    Returns True if this event_ts was already processed OR if the DB is unavailable.
    Two-phase check:
      1. SELECT first — fast path for retries (row already exists → True immediately)
      2. Upsert with ignore_duplicates — atomic insert; empty result = conflict → True
    On ANY DB error we return True (block) to prevent triplication from Slack retries.
    """
    sb = get_supabase()
    try:
        # Phase 1: cheap read — catches Slack retries without a write
        exists = (
            sb.table("processed_events")
            .select("event_ts")
            .eq("event_ts", event_ts)
            .limit(1)
            .execute()
            .data or []
        )
        if exists:
            log.warning("DUPLICATE BLOCKED (select) | event_ts=%s", event_ts)
            return True

        # Phase 2: atomic insert — winner processes, losers are blocked
        result = (
            sb.table("processed_events")
            .upsert({"event_ts": event_ts}, on_conflict="event_ts", ignore_duplicates=True)
            .execute()
        )
        if not result.data:
            log.warning("DUPLICATE BLOCKED (upsert) | event_ts=%s", event_ts)
            return True

        log.info("DEDUP INSERT OK | event_ts=%s", event_ts)
        # Cleanup stale records — best-effort
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            sb.table("processed_events").delete().lt("processed_at", cutoff).execute()
        except Exception:
            pass
        return False  # new event — process it

    except Exception as e:
        # DB unavailable — let through so the bot responds. Triplication is less bad than silence.
        log.error("DEDUP ERROR (letting through) | event_ts=%s err=%.200s", event_ts, e)
        return False

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Eres el AI Trading Copilot de BIA Energy, un asistente especializado en pricing energético, gestión de posición y estrategia de cobertura para el mercado eléctrico colombiano.

Tu rol es apoyar al equipo de Compra de Energía de BIA respondiendo preguntas en lenguaje natural, detectando riesgos, y generando recomendaciones estratégicas basadas en datos reales de simulaciones tarifarias.


IDENTIDAD Y TONO
- Respondes siempre en español.
- Tu tono es directo, analítico y concreto — como un trader senior con experiencia en el mercado colombiano.
- No usas lenguaje corporativo vacío. Vas al grano.
- Cuando hay un riesgo, lo dices claramente. Cuando hay una oportunidad, la señalas.
- Siempre explicas el razonamiento detrás de cada recomendación — qué variables activaron la conclusión.
- Nunca inventas datos. Si no tienes información suficiente, lo dices y explicas qué dato falta.
- No tomas decisiones autónomas — presentas análisis, escenarios y recomendaciones para que el equipo decida.
- Usa formato Slack (*negrita*, listas con -) y sé conciso.
- En TODA respuesta sobre tarifas incluye trazabilidad: Corrida #ID | agente | base_period | fecha | por quién | oficial: sí/no


CONOCIMIENTO DEL NEGOCIO

Mercado eléctrico colombiano:
- El mercado opera en Colombia regulado por la CREG.
- Los precios de bolsa (PB) son volátiles y dependen principalmente de hidrología, despacho y restricciones del sistema.
- La hidrología baja implica mayor uso de térmica y por lo tanto precios de bolsa más altos — es el principal factor de riesgo estacional.
- Los escenarios de PB son: LOW, MEDIUM y HIGH. El equipo trabaja con los tres escenarios simultáneamente.

Componentes de la tarifa CU:
- G — Generación (el componente más volátil, sensible a PB e hidrología)
- T — Transmisión
- D — Distribución (varía por Operador de Red)
- C — Comercialización
- P — Pérdidas
- R — Restricciones
- Aj — Ajuste tarifario. Si Aj es negativo, BIA está financiando tarifa al cliente. El saldo acumulado de todos los Aj se llama AD. Un AD negativo y creciente es la señal de alerta más crítica del sistema.

Agentes del mercado — BIA Energy compite como comercializador en 21 mercados/departamentos de Colombia.

Comercializadores competidores:
- EXEC = ENEL X COLOMBIA
- ENBC = ENERBIT
- NEUC = NEU ENERGY
- GNCC = VATIA
- DLRC = DICELER
- ETTC = ENERTOTAL
- QIEC = QI ENERGY
- RTQC = RUITOQUE
- SCEC = SOL & CIELO ENERGÍA

Nombres alternativos: VATIA=GNCC | ENEL X=EXEC | ENERBIT=ENBC | NEU=NEUC | Air-e=AIRE | Afinia=AFINIA | Celsia=CELSIA TOLIMA o CELSIA VALLE según contexto

Operadores de Red (OR) por mercado:
- EPM = Antioquia
- ENEL = Bogotá, Cundinamarca
- EBSA = Boyacá
- CHEC = Caldas
- EMCALI = Cali, Yumbo
- AFINIA = Caribe Mar
- AIRE = Caribe Sol
- EEP = Cartago, Pereira, Risaralda
- ENERCA = Casanare
- CEO = Cauca
- ELECTROHUILA = Huila
- EMSA = Meta
- CEDENAR = Nariño
- CENS = Norte de Santander
- EDEQ = Quindío
- ESSA = Santander
- CELSIA TOLIMA = Tolima
- CETSA = Tuluá
- CELSIA VALLE = Valle


REGLA DE TÁNDEM — LA MÁS IMPORTANTE
BIA debe mantener un nivel de cobertura aproximado al promedio simple de cobertura de los 5 OR de referencia: Enel, Emcali, Air-e (AIRE), Afinia y Celsia.

Nota: Los OR de referencia para el tándem pueden cambiar a futuro según decisión del equipo. La lista vigente debe confirmarse antes de cada análisis.

Lógica de recomendación de cobertura:
1. Calcular el promedio de cobertura de los 5 OR de referencia.
2. Determinar la banda objetivo de BIA: [promedio - 5%, promedio + 5%].
3. Si cobertura BIA < banda → recomendación de aumentar contratación.
4. Si está dentro de la banda → posición adecuada, monitorear.
5. Si está por encima de la banda → posición sobrecontratada, evaluar.

Variables adicionales que afectan la recomendación:
- PB proyectada: Si PB alta es el escenario dominante, estar en el límite inferior de la banda es más riesgoso.
- Aj y Saldo Acumulado (AD): Ver sección completa abajo.
- Exposición spot: Meses con alta exposición + PB alta = mayor riesgo económico.


LÓGICA DE AJ Y SALDO ACUMULADO (AD) — CRÍTICO
- Aj positivo → BIA está recuperando saldo acumulado. Señal favorable.
- Aj negativo → BIA está financiando tarifa al cliente. Acumula deuda tarifaria.

El Saldo Acumulado (AD) es el valor acumulado histórico de todos los Aj:
- AD positivo → BIA ha recuperado más de lo que ha financiado. Posición sana.
- AD negativo → BIA tiene un saldo pendiente de recuperar. Mientras más negativo, mayor el riesgo.

Alertas escalonadas por magnitud del AD acumulado:
- 🟢 OK: AD ≥ 0 — Posición sana
- 🟡 ATENCIÓN: AD entre -$10 y -$20/kWh — Monitorear tendencia
- 🟠 PRECAUCIÓN: AD entre -$20 y -$100/kWh — Revisar estrategia de cobertura
- 🔴 CRÍTICO: AD entre -$100 y -$300/kWh — Acción urgente requerida
- 🚨 EMERGENCIA: AD < -$300/kWh — Escalamiento inmediato a dirección

Análisis que debes hacer sobre Aj y AD:
- Tendencia del Aj: ¿Viene mejorando o deteriorándose mes a mes? ¿Cuántos meses consecutivos lleva negativo?
- Proyección del AD: ¿Cuántos meses más viene negativo? ¿Cuándo se proyecta recuperación?

Combinación de señales de mayor riesgo (los cuatro a la vez → recomendación urgente):
- Aj negativo y creciendo en magnitud
- AD acumulado negativo y profundizándose
- PB proyectada al alza (escenario HIGH dominante)
- Cobertura por debajo de la banda tándem

Señal de recuperación: Cuando Aj vuelve a positivo después de meses negativos, indicarlo como señal favorable y proyectar en cuántos meses se recuperaría el AD acumulado.


TIPOS DE ANÁLISIS QUE PUEDES HACER
1. Pricing y Competitividad: CU proyectada de BIA por mes y mercado, comparación vs competidores, competidor más agresivo, meses donde BIA pierde competitividad, componente que explica variaciones, anomalías.
2. Variaciones entre corridas: Comparar corrida actual vs anterior, qué cambió y cuánto, qué componente explica el mayor cambio, meses con mayor variación.
3. Escenarios de PB: Impacto de LOW/MEDIUM/HIGH sobre CU, sensibilidad de la posición, meses de mayor riesgo de bolsa.
4. Posición energética: Cobertura actual vs banda tándem, exposición spot, estado del Aj, meses críticos.
5. Recomendaciones de trading: Contratar ahora vs esperar, meses prioritarios, volumen sugerido, estrategia por escenario de PB.
6. Preguntas ejecutivas: Resumen de posición actual, principal riesgo próximos 6 meses, recomendación ejecutiva del trimestre, acción más urgente esta semana.


FORMATO DE RESPUESTAS
- Preguntas simples: Respuesta directa en 2-4 líneas con el dato y contexto mínimo.
- Análisis de variaciones: Qué cambió (número concreto) → Por qué (componente o variable) → Implicación (qué significa para la posición).
- Recomendaciones: Recomendación clara → Fundamento (datos y reglas) → Escenarios (LOW/MEDIUM/HIGH) → Urgencia (inmediata / próximas 2 semanas / próximo mes).
- Alertas: 🔴 ALERTA CRÍTICA — acción inmediata | 🟡 ATENCIÓN — monitorear | 🟢 OK — dentro de parámetros.


LO QUE NO HACES
- No inventas datos ni proyecciones sin respaldo en los datos disponibles.
- No tomas decisiones de contratación por cuenta propia — recomiendas, el equipo decide.
- No hablas de clientes individuales ni consumos por usuario final.
- No das recomendaciones de inversión financiera — tu dominio es gestión de energía y cobertura.
- No usas el modo OR como competidor — OR es el benchmark de referencia para el tándem.


CONTEXTO DE DATOS DISPONIBLES
Tienes acceso a las siguientes fuentes en tiempo real:
- simulation_runs — historial de corridas (agente, fecha, periodo base, estado oficial)
- simulation_results — resultados detallados por corrida: CU, G, T, D, C, P, R, Aj por mercado, periodo y escenario PB
- pb_rates — escenarios oficiales de precio de bolsa (LOW, MEDIUM, HIGH) por mes
- cu_comparison — CU de todos los agentes en un solo lugar
- spread_vs_competitors — spread de BIA vs cada competidor
- latest_official_runs — corrida oficial más reciente por agente

Cuando respondas, indica siempre qué corrida estás usando (ID y fecha) para trazabilidad.
"""

# ── Constants ─────────────────────────────────────────────────────────────────

_SCENARIO_MAP = {"BAJO": "LOW", "MEDIO": "MEDIUM", "ALTO": "HIGH"}

_MARKETS = (
    "ANTIOQUIA|BOGOTA|BOYACA|CALDAS|CALI|CARIBE MAR|CARIBE SOL|CARTAGO|CASANARE|"
    "CUNDINAMARCA|MEDELLIN|NARINO|NARIÑO|SANTANDER|TOLIMA|VALLE|COSTA|LLANOS|SUROCCIDENTE"
)

_MONTH_NAMES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}

# ── Intent detection ──────────────────────────────────────────────────────────

_CU_KEYWORDS = re.compile(
    r"\b(cu|costo unitario|tarifa|componente|g\b|c\b|t\b|d\b|p\b|r\b|g_base|desglose|"
    r"riesgo|exposici[oó]n|escenario|qu[eé] pasa|pr[oó]ximos meses|aj\b|saldo)\b",
    re.IGNORECASE,
)
_SPREAD_KEYWORDS = re.compile(
    r"\b(spread|competidor|competencia|competitiv\w*|rival|vs\b|versus|comparar|diferencia|ventaja|"
    r"epm|emcali|afinia|codensa|ebsa|chec|essa|eep|electrohuila|emsa|cedenar|cens|edeq|enerca|cetsa)\b"
    r"|celsia\s+(tolima|valle)|enel\s+(bogot[aá]|or\b)|caribe\s+sol",
    re.IGNORECASE,
)
# Detect questions about trends / full period horizons
_TREND_KEYWORDS = re.compile(
    r"\b(tendencia|variaci[oó]n\w*|evoluci[oó]n|todos los (meses|periodos)|horizonte|"
    r"a lo largo|semestre|trimestre|cuatrimestre|meses disponibles|pr[oó]ximos \d+ meses)\b",
    re.IGNORECASE,
)
_RUNS_KEYWORDS = re.compile(
    r"\b(corrida|simulaci[oó]n|run|reciente|[uú]ltim[ao]|ejecut|qui[eé]n corri[oó]|hoy|ayer)\b",
    re.IGNORECASE,
)
_OR_RANKING_KEYWORDS = re.compile(
    r"\b(qu[eé]\s+or|cu[aá]l\s+or|or\s+m[aá]s\s+barat\w*|or\s+m[aá]s\s+car\w*|"
    r"m[aá]s\s+barat\w*.*\bor\b|\bor\b.*m[aá]s\s+barat\w*|"
    r"tarifa\s+m[aá]s\s+baj\w*.*\bor\b|\bor\b.*tarifa\s+m[aá]s\s+baj\w*|"
    r"menor\s+tarifa.*\bor\b|\bor\b.*menor\s+tarifa|"
    r"cu\s+m[aá]s\s+baj\w*.*\bor\b|\bor\b.*cu\s+m[aá]s\s+baj\w*|"
    r"ranking.*\bor\b|\bor\b.*ranking)\b",
    re.IGNORECASE,
)
_TANDEM_KEYWORDS = re.compile(
    r"\b(t[aá]ndem|tandem|cobertura|banda|posici[oó]n|qc|contratar|contrataci[oó]n|"
    r"priorizar|priorit|debo comprar|cu[aá]nto comprar|riesgo|exposici[oó]n|conviene|"
    r"deber[ií]a|pr[oó]ximos\s+\d+\s+meses|qu[eé]\s+hago)\b",
    re.IGNORECASE,
)

# OR de referencia para el tándem (or_code tal como aparece en simulation_results)
_TANDEM_OR_REFS = ["ENEL", "EMCALI", "AIRE", "AFINIA", "CELSIA TOLIMA", "CELSIA VALLE", "EPM"]

# ── Run resolution ────────────────────────────────────────────────────────────

def _fmt_run(run: dict) -> str:
    """One-line trazabilidad string shown in every response."""
    run_id = run.get("id")
    created = (run.get("created_at") or "")[:16].replace("T", " ")
    who = run.get("triggered_by") or "desconocido"
    official = "si" if run.get("is_official") else "no"
    agent = run.get("agent_code", "")
    bp = run.get("base_period", "")
    return f"Corrida #{run_id} | {agent} | base {bp} | {created} UTC | por {who} | oficial: {official}"


_AGENT_CODES = {"neuc", "bia", "exec", "gncc", "or"}
# Common Spanish words that match the "de <word>" person-name regex but are never names
_EXCLUDED_PERSON_WORDS = _AGENT_CODES | {
    "tandem", "tándem", "regla", "spread", "precio", "tarifa", "mercado",
    "corrida", "simulacion", "simulación", "energia", "energía", "contrato",
    "riesgo", "cobertura", "banda", "posicion", "posición", "escenario",
    "componente", "costo", "ahorro", "periodo", "período", "dato", "datos",
    "acuerdo", "resultado", "resultados", "hoy", "ayer", "mes", "año",
}


def _resolve_run(text: str, agent_code: str) -> dict | None:
    """
    Resolve which simulation_run to use based on natural language cues.

    Priority (each falls through to the next if no results):
      1. "la que corrió <nombre>"  → triggered_by ILIKE '<nombre>%'
      2. "la de hoy"               → created_at >= today 00:00 UTC
      3. "la de ayer"              → created_at between yesterday and today
      4. "la última"               → max created_at regardless of official
      5. DEFAULT                   → most recent is_official=true
      6. FINAL FALLBACK            → most recent completed run
    """
    sb = get_supabase()
    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    text_lower = text.lower()

    # Person detection — capture patterns like:
    #   "corrió Allison", "ejecutó Juliana", "la corrida de Allison", "la de Juliana", "de Allison"
    # Exclude agent codes (BIA, OR, EXEC…) so "de BIA" doesn't resolve as a person name.
    person_match = re.search(
        r"(?:corri[oó]|ejecut[oó]|corrida\s+de|la\s+de)\s+([a-záéíóúñ]{3,})",
        text_lower,
    )
    if not person_match:
        # Fallback: bare "de <name>" where name is a known first name (≥4 chars, not an agent code)
        person_match = re.search(r"\bde\s+([a-záéíóúñ]{4,})\b", text_lower)
    if person_match and person_match.group(1).lower() in _EXCLUDED_PERSON_WORDS:
        person_match = None

    wants_today     = bool(re.search(r"\bhoy\b", text_lower))
    wants_yesterday = bool(re.search(r"\bayer\b", text_lower))
    wants_latest    = bool(re.search(r"\b[uú]ltim[ao]\b", text_lower)) and not wants_today and not wants_yesterday

    # Build a fresh query each time — avoids QueryBuilder mutation across calls
    def q(**extra_filters):
        builder = (
            sb.table("simulation_runs")
            .select("id,agent_code,base_period,is_official,triggered_by,created_at,status")
            .eq("agent_code", agent_code)
            .eq("status", "COMPLETED")
        )
        for k, v in extra_filters.items():
            builder = builder.eq(k, v)
        return builder

    # Strategy 1 — person name: search any run (official or not) with partial match
    if person_match:
        name = person_match.group(1).strip()
        # Build fresh query without is_official filter so comparisons across runs work
        rows = (
            sb.table("simulation_runs")
            .select("id,agent_code,base_period,is_official,triggered_by,created_at,status")
            .eq("agent_code", agent_code)
            .eq("status", "COMPLETED")
            .ilike("triggered_by", f"%{name}%")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
        if rows:
            return rows[0]

    # Strategy 2 — today (falls through if empty)
    if wants_today:
        rows = q().gte("created_at", today_start.isoformat()).order("created_at", desc=True).limit(1).execute().data or []
        if rows:
            return rows[0]

    # Strategy 3 — yesterday (falls through if empty)
    if wants_yesterday:
        rows = (
            q()
            .gte("created_at", yesterday_start.isoformat())
            .lt("created_at", today_start.isoformat())
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
        if rows:
            return rows[0]

    # Strategy 4 — latest regardless of official
    if wants_latest:
        rows = q().order("created_at", desc=True).limit(1).execute().data or []
        if rows:
            return rows[0]

    # Default — most recent official run
    rows = q(**{"is_official": True}).order("created_at", desc=True).limit(1).execute().data or []
    if rows:
        return rows[0]

    # Final fallback — any completed run
    rows = q().order("created_at", desc=True).limit(1).execute().data or []
    return rows[0] if rows else None


# ── Context builders ──────────────────────────────────────────────────────────

def _ctx_simulation_runs(text: str) -> str:
    sb = get_supabase()

    text_lower = text.lower()
    wants_compare = bool(re.search(r"\bcompar\w*\b", text_lower))

    # Find ALL person names mentioned (e.g. "corrida de Allison y ... corrió Juliana")
    raw_names = re.findall(
        r"(?:corri[oó]|ejecut[oó]|corrida\s+de|la\s+de|de)\s+([a-záéíóúñ]{3,})",
        text_lower,
    )
    person_names = [n for n in dict.fromkeys(raw_names) if n not in _EXCLUDED_PERSON_WORDS]

    wants_today     = bool(re.search(r"\bhoy\b", text_lower))
    wants_yesterday = bool(re.search(r"\bayer\b", text_lower))

    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    def _base_q():
        return (
            sb.table("simulation_runs")
            .select("id,agent_code,base_period,is_official,triggered_by,created_at,status")
            .eq("status", "COMPLETED")
        )

    rows: list[dict] = []

    if person_names:
        # Fetch the most recent run per person (contains search, not prefix)
        seen_ids: set = set()
        for name in person_names:
            person_rows = (
                _base_q()
                .ilike("triggered_by", f"%{name}%")
                .order("created_at", desc=True)
                .limit(5)
                .execute()
                .data or []
            )
            for r in person_rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    rows.append(r)
        log.info("RUNS person_names=%s found=%d", person_names, len(rows))
    elif wants_today:
        rows = _base_q().gte("created_at", today_start.isoformat()).order("created_at", desc=True).limit(20).execute().data or []
    elif wants_yesterday:
        rows = _base_q().gte("created_at", yesterday_start.isoformat()).lt("created_at", today_start.isoformat()).order("created_at", desc=True).limit(20).execute().data or []
    else:
        rows = _base_q().order("created_at", desc=True).limit(20).execute().data or []

    if not rows:
        return "No se encontraron corridas con esos criterios."

    lines = ["=== Corridas de simulacion ==="]
    for r in rows:
        official = "OFICIAL" if r.get("is_official") else "no oficial"
        created = (r.get("created_at") or "")[:16].replace("T", " ")
        lines.append(
            f"  #{r['id']} | {r.get('agent_code')} | base {r.get('base_period')} | "
            f"{created} UTC | {r.get('triggered_by')} | {official}"
        )

    if wants_compare:
        official_rows   = [r for r in rows if r.get("is_official")]
        unofficial_rows = [r for r in rows if not r.get("is_official")]
        if official_rows and unofficial_rows:
            lines.append("\n--- Para comparacion ---")
            lines.append(f"  Oficial mas reciente:    {_fmt_run(official_rows[0])}")
            lines.append(f"  No oficial mas reciente: {_fmt_run(unofficial_rows[0])}")

    return "\n".join(lines)


# ── Period range parsing (Fix 2) ─────────────────────────────────────────────

_PERIOD_RANGE_RE = re.compile(
    r"\b(primer[ao]?|segundo[ao]?|tercer[ao]?|cuarto[ao]?|"
    r"1\s*[ero.]*|2\s*[do.]*|3\s*[ero.]*|4\s*[to.]*)"
    r"\s+(semestre|trimestre|cuatrimestre)\s+(?:de\s+)?(\d{4})\b",
    re.IGNORECASE,
)
_ORDINAL_IDX = {
    "primer": 0, "primera": 0, "primero": 0,
    "segundo": 1, "segunda": 1,
    "tercer": 2, "tercera": 2, "tercero": 2,
    "cuarto": 3, "cuarta": 3,
    "1": 0, "2": 1, "3": 2, "4": 3,
}
_PERIOD_BLOCKS = {
    "semestre":     [(1, 6),  (7, 12)],
    "trimestre":    [(1, 3),  (4, 6),  (7, 9),  (10, 12)],
    "cuatrimestre": [(1, 4),  (5, 8),  (9, 12)],
}


def _parse_period_range(text: str) -> list[str]:
    """Parse 'segundo semestre 2026' → ['07-2026',...,'12-2026'], etc."""
    m = _PERIOD_RANGE_RE.search(text)
    if not m:
        return []
    raw = re.sub(r"[^a-z]", "", m.group(1).lower())
    idx = _ORDINAL_IDX.get(raw)
    if idx is None:
        return []
    period_type = m.group(2).lower()
    year = m.group(3)
    blocks = _PERIOD_BLOCKS.get(period_type, [])
    if idx >= len(blocks):
        return []
    start, end = blocks[idx]
    return [f"{str(mn).zfill(2)}-{year}" for mn in range(start, end + 1)]


def _parse_period(text: str) -> str | None:
    m = re.search(r"\b(\d{2})[-/](\d{4})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(
        r"\b(" + "|".join(_MONTH_NAMES) + r")\b\s+(?:de\s+)?(\d{4})\b",
        text, re.IGNORECASE,
    )
    if m:
        return f"{_MONTH_NAMES[m.group(1).lower()]}-{m.group(2)}"
    return None


def _next_n_periods(sb, run_id: int, n: int = 3) -> list[str]:
    rows = (
        sb.table("simulation_results")
        .select("period")
        .eq("run_id", run_id)
        .limit(2000)
        .execute()
        .data or []
    )
    return sorted({r["period"] for r in rows})[:n]


def _ctx_cu_components(text: str) -> str:
    sb = get_supabase()

    agent_match = re.search(r"\b(NEUC|BIA|EXEC|GNCC|OR)\b", text, re.IGNORECASE)
    scen_match  = re.search(r"\b(LOW|MEDIUM|HIGH|BAJO|MEDIO|ALTO)\b", text, re.IGNORECASE)

    text_norm = unicodedata.normalize("NFD", text)
    text_norm = "".join(c for c in text_norm if unicodedata.category(c) != "Mn")
    mkt_match = re.search(_MARKETS, text_norm, re.IGNORECASE)

    period_asked   = _parse_period(text_norm)
    agent_code     = agent_match.group(1).upper() if agent_match else "BIA"
    scenario_asked = (
        _SCENARIO_MAP.get(scen_match.group(1).upper(), scen_match.group(1).upper())
        if scen_match else None
    )
    market_filter  = mkt_match.group(0).upper() if mkt_match else None

    # ── Resolve which run to use ──────────────────────────────────────────────
    run = _resolve_run(text, agent_code)
    if not run:
        return f"No se encontro ninguna corrida completada para {agent_code}."

    base_period = run["base_period"]
    run_trace   = _fmt_run(run)

    # ── Discover available pb_scenario (max 20 rows — only 3 distinct values) ──
    run_id = run["id"]
    avail_rows = (
        sb.table("simulation_results")
        .select("pb_scenario")
        .eq("run_id", run_id)
        .limit(20)
        .execute()
        .data or []
    )
    available = sorted({r["pb_scenario"] for r in avail_rows})

    if not available:
        return f"No hay datos de resultados para {run_trace}."

    # ── Resolve scenario ──────────────────────────────────────────────────────
    scenario_warning: str | None = None
    if scenario_asked:
        if scenario_asked in available:
            scenarios_to_fetch = [scenario_asked]
        else:
            scenarios_to_fetch = available
            scenario_warning = (
                f"No encontre el escenario *{scenario_asked}*. "
                f"Escenarios disponibles: *{', '.join(available)}*."
            )
    else:
        scenarios_to_fetch = available

    # ── Resolve periods to show ───────────────────────────────────────────────
    wants_trend  = bool(_TREND_KEYWORDS.search(text))
    period_range = _parse_period_range(text)

    if period_asked:
        periods_to_fetch = [period_asked]
        period_label = f"period={period_asked}"
    elif period_range:
        periods_to_fetch = period_range
        period_label = f"periodos={', '.join(period_range)}"
    elif wants_trend:
        periods_to_fetch = _next_n_periods(sb, run_id, n=12)
        period_label = f"todos los periodos disponibles ({', '.join(periods_to_fetch)})"
    else:
        periods_to_fetch = _next_n_periods(sb, run_id, n=3)
        period_label = f"proximos 3 periodos ({', '.join(periods_to_fetch)})"

    # ── Main data query — scoped to run_id to avoid cross-run contamination ──
    query = (
        sb.table("simulation_results")
        .select(
            "agent_code,base_period,market,period,pb_scenario,"
            "cu,g,c,t,d,p,r,g_base,g_transitorio,aj,alpha"
        )
        .eq("run_id", run_id)
        .eq("tension_level", 2)
        .eq("rate_type", "USER")
        .in_("pb_scenario", scenarios_to_fetch)
    )
    # For trend queries skip the period filter — we want all periods for trend analysis.
    if not wants_trend:
        query = query.in_("period", periods_to_fetch)
    if market_filter:
        query = query.ilike("market", f"%{market_filter}%")

    rows = query.order("period").order("market").order("pb_scenario").limit(200).execute().data or []

    if not rows:
        return (
            f"No se encontraron registros para {agent_code}, {period_label}, "
            f"escenarios={scenarios_to_fetch}"
            + (f", mercado={market_filter}" if market_filter else "") + "."
        )

    # ── Group and format ──────────────────────────────────────────────────────
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["market"], r["period"], r["pb_scenario"])].append(r)

    lines: list[str] = []
    if scenario_warning:
        lines.append(scenario_warning)

    lines += [
        f"[Trazabilidad] {run_trace}",
        f"=== CU y componentes — {agent_code} | {period_label}"
        + (f" | mercado={market_filter}" if market_filter else "") + " ===",
        f"Escenarios disponibles: {', '.join(available)} | Mostrando: {', '.join(scenarios_to_fetch)}",
        f"(tension_level=2, rate_type=USER | {len(groups)} combinaciones)",
    ]

    if wants_trend and not market_filter:
        # Compact trend view: G (and CU) per period × scenario, averaged across all markets.
        # Replaces the per-market detail entirely — avoids [:90] truncation hiding later periods.
        period_scen: dict = defaultdict(list)
        for r in rows:
            period_scen[(r["period"], r["pb_scenario"])].append(r)

        # Sort chronologically: periods are MM-YYYY; sort by (year, month)
        def _period_sort_key(ps):
            p = ps[0]
            parts = p.split("-")
            return (int(parts[1]), int(parts[0]))

        lines.append("--- Tendencia G y CU — promedio de todos los mercados por periodo ---")
        lines.append(f"{'Periodo':<10} {'Escenario':<8} {'G_avg':>8} {'CU_avg':>8} {'G_min':>8} {'G_max':>8}")
        lines.append("-" * 58)
        for (period, scenario) in sorted(period_scen.keys(), key=_period_sort_key):
            entries = period_scen[(period, scenario)]
            g_vals  = [float(e["g"]  or 0) for e in entries if e.get("g")  is not None]
            cu_vals = [float(e["cu"] or 0) for e in entries if e.get("cu") is not None]
            if not g_vals:
                continue
            g_avg  = round(sum(g_vals)  / len(g_vals),  2)
            cu_avg = round(sum(cu_vals) / len(cu_vals), 2) if cu_vals else 0
            g_min  = round(min(g_vals), 2)
            g_max  = round(max(g_vals), 2)
            lines.append(f"  {period:<10} {scenario:<8} {g_avg:>8} {cu_avg:>8} {g_min:>8} {g_max:>8}")
        lines.append(f"  (basado en {len(rows)} registros | {len({r['market'] for r in rows})} mercados)")
    else:
        for (market, period, scenario), entries in list(groups.items())[:90]:
            def avg(col: str) -> float:
                return round(sum(e[col] or 0 for e in entries) / len(entries), 2)
            lines.append(
                f"  {market} | {period} | {scenario}: "
                f"CU={avg('cu')} G={avg('g')} C={avg('c')} "
                f"T={avg('t')} D={avg('d')} P={avg('p')} R={avg('r')}"
            )
        if len(groups) > 90:
            lines.append(f"  ... y {len(groups) - 90} combinaciones mas.")
    return "\n".join(lines)


def _normalize_market(name: str) -> str:
    """Remove accents and uppercase for consistent market name comparison."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode().upper().strip()

_MARKET_ALIASES: dict[str, list[str]] = {
    "BOGOTA":    ["BOGOTA", "BOGOTÁ", "BOGOTA D.C.", "BOGOTÁ D.C."],
    "MEDELLIN":  ["MEDELLIN", "MEDELLÍN"],
    "NARINO":    ["NARINO", "NARIÑO"],
    "QUINDIO":   ["QUINDIO", "QUINDÍO"],
    "CALI":      ["CALI", "SANTIAGO DE CALI"],
}

# OR operators as competitors: text aliases → or_code in simulation_results
_OR_COMP_ALIASES: dict[str, list[str]] = {
    "EPM":           ["epm", "empresas publicas", "empresas públicas"],
    "ENEL":          ["enel bogota", "enel bogotá", "codensa", "enel or"],
    "EMCALI":        ["emcali"],
    "AFINIA":        ["afinia"],
    "AIRE":          ["air-e", "aire caribe", "caribe sol", "aire"],
    "CELSIA TOLIMA": ["celsia tolima"],
    "CELSIA VALLE":  ["celsia valle"],
    "EBSA":          ["ebsa"],
    "CHEC":          ["chec"],
    "ESSA":          ["essa"],
    "EEP":           ["eep"],
    "ELECTROHUILA":  ["electrohuila"],
    "EMSA":          ["emsa"],
    "CEO":           ["ceo cauca"],
    "CEDENAR":       ["cedenar"],
    "CENS":          ["cens"],
    "EDEQ":          ["edeq"],
    "ENERCA":        ["enerca"],
    "CETSA":         ["cetsa"],
}

# Map or_code → BIA market name (for filtering BIA results)
_OR_CODE_TO_BIA_MARKET: dict[str, str] = {
    "EPM":           "ANTIOQUIA",
    "ENEL":          "BOGOTA",
    "EMCALI":        "CALI",
    "AFINIA":        "CARIBE MAR",
    "AIRE":          "CARIBE SOL",
    "CELSIA TOLIMA": "TOLIMA",
    "CELSIA VALLE":  "VALLE",
    "EBSA":          "BOYACA",
    "CHEC":          "CALDAS",
    "ESSA":          "SANTANDER",
    "EEP":           "PEREIRA",
    "ELECTROHUILA":  "HUILA",
    "EMSA":          "META",
    "CEO":           "CAUCA",
    "CEDENAR":       "NARINO",
    "CENS":          "NORTE SANTANDER",
    "EDEQ":          "QUINDIO",
    "ENERCA":        "CASANARE",
    "CETSA":         "TULUA",
}


def _ctx_or_ranking_by_period(text: str) -> str:
    """Rank all OR operators by CU for a period range — answers 'qué OR tiene la tarifa más baja'."""
    sb = get_supabase()

    or_run = _resolve_run(text, "OR")
    if not or_run:
        return "No se encontró corrida oficial de OR."

    or_run_id = or_run["id"]

    # Resolve period range: explicit ("segundo semestre 2026") or next N
    period_range = _parse_period_range(text)
    if period_range:
        periods = period_range
        period_label = f"{periods[0]} a {periods[-1]}"
    else:
        periods = _next_n_periods(sb, or_run_id, n=3)
        period_label = f"próximos 3 periodos ({', '.join(periods)})"

    # Resolve scenario
    scen_match = re.search(r"\b(LOW|MEDIUM|HIGH|BAJO|MEDIO|ALTO)\b", text, re.IGNORECASE)
    scenario = (
        _SCENARIO_MAP.get(scen_match.group(1).upper(), scen_match.group(1).upper())
        if scen_match else "MEDIUM"
    )

    tension_level = _parse_tension_level(text)

    rows = (
        sb.table("simulation_results")
        .select("or_code,period,pb_scenario,cu")
        .eq("run_id", or_run_id)
        .eq("tension_level", tension_level)
        .eq("rate_type", "USER")
        .eq("pb_scenario", scenario)
        .in_("or_code", list(_OR_CODE_TO_BIA_MARKET.keys()))
        .in_("period", periods)
        .order("period").order("or_code")
        .limit(200)
        .execute()
        .data or []
    )

    if not rows:
        return f"No hay datos OR para {period_label}, escenario {scenario}, tension_level={tension_level}."

    # Group by or_code
    or_cus: dict = defaultdict(list)
    for r in rows:
        if r.get("cu") is not None:
            or_cus[r["or_code"]].append(float(r["cu"]))

    if not or_cus:
        return f"Sin datos CU de OR para {period_label}, escenario {scenario}."

    summary = []
    for or_code, cus in or_cus.items():
        market = _OR_CODE_TO_BIA_MARKET.get(or_code, "?")
        summary.append((or_code, market, round(sum(cus)/len(cus), 2), round(min(cus), 2), round(max(cus), 2)))

    summary.sort(key=lambda x: x[2])  # cheapest first

    lines = [
        f"=== Ranking OR por CU — {period_label} | Escenario: {scenario} ===",
        f"Corrida OR: #{or_run_id} | base {or_run.get('base_period')} | oficial: {'sí' if or_run.get('is_official') else 'no'}",
        f"(tension_level={tension_level}, rate_type=USER | periodos: {', '.join(periods)})",
        "",
        f"{'#':<3} {'OR':<16} {'Mercado':<16} {'CU_avg':>8} {'CU_min':>8} {'CU_max':>8}",
        "-" * 65,
    ]
    for rank, (or_code, market, avg_cu, min_cu, max_cu) in enumerate(summary, 1):
        lines.append(f"  {rank:<3} {or_code:<14} {market:<16} {avg_cu:>8} {min_cu:>8} {max_cu:>8}")

    log.info("OR RANKING | or_run=%s periods=%s scenario=%s ors=%d", or_run_id, periods, scenario, len(summary))
    return "\n".join(lines)


def _parse_tension_level(text: str) -> int:
    """Parse tension level from text. Defaults to 2 (T2/media tensión)."""
    m = re.search(
        r"\b(t[123]\b|tension\s*[123]|tensi[oó]n\s*[123]|nivel\s*[123]|"
        r"alta\s+tensi[oó]n|media\s+tensi[oó]n|baja\s+tensi[oó]n)\b",
        text, re.IGNORECASE,
    )
    if not m:
        return 2
    raw = m.group(0).lower()
    if any(x in raw for x in ["t1", "tension 1", "tensión 1", "nivel 1", "baja"]):
        return 1
    if any(x in raw for x in ["t3", "tension 3", "tensión 3", "nivel 3", "alta"]):
        return 3
    return 2


def _extract_or_codes(text: str) -> list[str]:
    """Return ALL OR codes explicitly mentioned in text, in order of appearance."""
    text_l = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    # Scan in alias order so longer aliases win (e.g. "celsia tolima" before "celsia")
    for code, aliases in _OR_COMP_ALIASES.items():
        for alias in aliases:
            if alias in text_l and code not in seen:
                found.append(code)
                seen.add(code)
                break
    # "enel" alone (not "enel x") → OR
    if "ENEL" not in seen and "enel" in text_l and "enel x" not in text_l:
        found.append("ENEL")
    return found


def _ctx_spread_or_one(
    sb, bia_run: dict, or_run: dict, or_code: str, tension_level: int
) -> str:
    """Build spread block for a single OR code. Shared by single and multi-OR paths."""
    bia_market = _OR_CODE_TO_BIA_MARKET.get(or_code)
    bia_run_id = bia_run["id"]
    or_run_id  = or_run["id"]

    bia_q = (
        sb.table("simulation_results")
        .select("market,period,pb_scenario,cu,g,t,d,c,p,r")
        .eq("run_id", bia_run_id)
        .eq("tension_level", tension_level)
        .eq("rate_type", "USER")
        .order("period").order("pb_scenario")
        .limit(200)
    )
    if bia_market:
        bia_q = bia_q.ilike("market", f"%{bia_market}%")
    bia_rows = bia_q.execute().data or []

    or_rows = (
        sb.table("simulation_results")
        .select("or_code,period,pb_scenario,cu")
        .eq("run_id", or_run_id)
        .eq("or_code", or_code)
        .eq("tension_level", tension_level)
        .eq("rate_type", "USER")
        .order("period").order("pb_scenario")
        .limit(200)
        .execute()
        .data or []
    )

    if not bia_rows:
        return f"  [Sin datos BIA para {or_code} ({bia_market}) | tension_level={tension_level}]"
    if not or_rows:
        return f"  [Sin datos OR para or_code={or_code} | tension_level={tension_level}]"

    or_index: dict = {}
    for r in or_rows:
        or_index[(r["period"], r["pb_scenario"])] = float(r["cu"] or 0)

    bia_groups: dict = defaultdict(list)
    for r in bia_rows:
        bia_groups[(r["market"], r["period"], r["pb_scenario"])].append(float(r["cu"] or 0))

    lines = [
        f"=== Spread BIA vs {or_code} (OR) | tension_level={tension_level} ===",
        f"  Mercado BIA: {bia_market or 'todos'}",
    ]
    for (market, period, scenario), bia_cus in sorted(bia_groups.items()):
        bia_cu = round(sum(bia_cus) / len(bia_cus), 2)
        or_cu  = or_index.get((period, scenario))
        if or_cu:
            spread = round(bia_cu - or_cu, 2)
            sign   = f"+{spread}" if spread > 0 else str(spread)
            lines.append(
                f"  {market} | {period} | {scenario}: BIA={bia_cu}  {or_code}={round(or_cu,2)}  spread={sign}"
            )
        else:
            lines.append(f"  {market} | {period} | {scenario}: BIA={bia_cu}  {or_code}=sin dato")

    log.info("SPREAD OR | or_code=%s tension=%d bia_rows=%d or_rows=%d",
             or_code, tension_level, len(bia_rows), len(or_rows))
    return "\n".join(lines)


def _ctx_spread_or(text: str, or_codes: list[str]) -> str:
    """Compute spread BIA vs one or more OR operators. Iterates all requested OR codes."""
    sb = get_supabase()

    bia_run = _resolve_run(text, "BIA")
    if not bia_run:
        return "No se encontró corrida oficial de BIA."
    or_run = _resolve_run(text, "OR")
    if not or_run:
        return "No se encontró corrida oficial de OR."

    tension_level = _parse_tension_level(text)

    header = [
        f"Corrida BIA: #{bia_run['id']} | base {bia_run.get('base_period')} | "
        f"{bia_run.get('triggered_by')} | oficial: {'sí' if bia_run.get('is_official') else 'no'}",
        f"Corrida OR:  #{or_run['id']} | base {or_run.get('base_period')} | "
        f"oficial: {'sí' if or_run.get('is_official') else 'no'}",
        f"Nota: OR como Operadores de Red (competidores de BIA en sus mercados)",
        f"(tension_level={tension_level}, rate_type=USER)",
        "",
    ]

    blocks = ["\n".join(header)]
    for code in or_codes:
        blocks.append(_ctx_spread_or_one(sb, bia_run, or_run, code, tension_level))

    return "\n\n".join(blocks)


def _ctx_spread_or_all(text: str) -> str:
    """Spread BIA vs ALL OR operators across all markets — summary view."""
    sb = get_supabase()

    bia_run = _resolve_run(text, "BIA")
    if not bia_run:
        return "No se encontró corrida oficial de BIA."
    or_run = _resolve_run(text, "OR")
    if not or_run:
        return "No se encontró corrida oficial de OR."

    bia_run_id = bia_run["id"]
    or_run_id  = or_run["id"]

    # Fetch BIA results — all markets, tension_level=2, USER, MEDIUM scenario only (keep context small)
    bia_rows = (
        sb.table("simulation_results")
        .select("market,period,pb_scenario,cu")
        .eq("run_id", bia_run_id)
        .eq("tension_level", 2)
        .eq("rate_type", "USER")
        .eq("pb_scenario", "MEDIUM")
        .order("period").order("market")
        .limit(200)
        .execute()
        .data or []
    )

    # Fetch OR results — all or_codes we track, same filters
    or_rows = (
        sb.table("simulation_results")
        .select("or_code,market,period,pb_scenario,cu")
        .eq("run_id", or_run_id)
        .in_("or_code", list(_OR_CODE_TO_BIA_MARKET.keys()))
        .eq("tension_level", 2)
        .eq("rate_type", "USER")
        .eq("pb_scenario", "MEDIUM")
        .order("period").order("or_code")
        .limit(200)
        .execute()
        .data or []
    )

    if not bia_rows:
        return f"No hay datos BIA en corrida #{bia_run_id}."
    if not or_rows:
        return f"No hay datos OR en corrida #{or_run_id}."

    # Index BIA by (market_upper, period)
    bia_index: dict = {}
    for r in bia_rows:
        key = (_normalize_market(r.get("market", "")), r["period"])
        bia_index[key] = float(r["cu"] or 0)

    # Index OR by (or_code, period)
    or_index: dict = defaultdict(dict)
    for r in or_rows:
        or_index[r["or_code"]][r["period"]] = float(r["cu"] or 0)

    lines = [
        f"=== Spread BIA vs OR — todos los mercados (MEDIUM, tension_level=2, rate_type=USER) ===",
        f"Corrida BIA: #{bia_run_id} | base {bia_run.get('base_period')} | {bia_run.get('triggered_by')} | oficial: {'sí' if bia_run.get('is_official') else 'no'}",
        f"Corrida OR:  #{or_run_id} | base {or_run.get('base_period')} | oficial: {'sí' if or_run.get('is_official') else 'no'}",
        "",
    ]

    # For each OR code, compute average spread across all periods where both have data
    summary_rows: list[tuple[str, str, float, float, float]] = []  # (or_code, market, avg_spread, min_spread, max_spread)
    for or_code, bia_market in _OR_CODE_TO_BIA_MARKET.items():
        or_periods = or_index.get(or_code, {})
        if not or_periods:
            continue
        spreads = []
        for period, or_cu in or_periods.items():
            bia_cu = bia_index.get((_normalize_market(bia_market), period))
            if bia_cu:
                spreads.append(round(bia_cu - or_cu, 2))
        if not spreads:
            continue
        avg_s = round(sum(spreads) / len(spreads), 2)
        summary_rows.append((or_code, bia_market, avg_s, min(spreads), max(spreads)))

    # Sort: most favorable for BIA first (most negative = BIA cheapest vs OR)
    summary_rows.sort(key=lambda x: x[2])

    lines.append(f"{'OR':<16} {'Mercado BIA':<16} {'Spread avg':>11} {'Min':>9} {'Max':>9} {'Posición'}")
    lines.append("-" * 75)
    for or_code, bia_market, avg_s, min_s, max_s in summary_rows:
        sign = f"+{avg_s}" if avg_s > 0 else str(avg_s)
        posicion = "🟢 BIA más barato" if avg_s < 0 else "🔴 BIA más caro"
        lines.append(f"  {or_code:<14} {bia_market:<16} {sign:>11} {min_s:>9} {max_s:>9}   {posicion}")

    log.info("SPREAD OR ALL | bia_run=%s or_run=%s markets=%d", bia_run_id, or_run_id, len(summary_rows))
    return "\n".join(lines)


def _ctx_spread(text: str) -> str:
    sb = get_supabase()
    text_l = text.lower()

    # Extract ALL OR codes mentioned — supports multi-OR questions
    asked_ors = _extract_or_codes(text_l)

    if asked_ors:
        return _ctx_spread_or(text, asked_ors)

    # Generic OR query ("frente a los OR", "vs OR", "competitivo frente a OR") → all markets
    asks_or_generic = bool(re.search(
        r"\b(frente a.*\bor\b|\bor\b.*mercado|vs.*\bor\b|\bor\b.*competitiv|competitiv.*\bor\b|"
        r"posici[oó]n.*\bor\b|\bor\b.*posici[oó]n|m[aá]s competitiv.*\bor\b)\b",
        text_l,
    ))
    if not asks_or_generic:
        # Simpler: "OR" appears and no specific OR alias was matched
        asks_or_generic = bool(re.search(r"\bor\b", text_l)) and "or_code" not in text_l
    if asks_or_generic:
        return _ctx_spread_or_all(text)

    # ── Comercializadores path (existing view) ────────────────────────────────
    # Detect market mention in the question and build filter candidates
    text_norm = _normalize_market(text)
    market_filter: list[str] | None = None
    for canonical, aliases in _MARKET_ALIASES.items():
        if any(_normalize_market(a) in text_norm for a in aliases) or canonical in text_norm:
            market_filter = aliases
            break
    if not market_filter:
        market_match = re.search(
            r"\b(bogot[aá]|medell[ií]n|cali|antioquia|barranquilla|cartagena|"
            r"tolima|boyac[aá]|caldas|risaralda|santander|huila|valle|cauca|"
            r"nari[nñ]o|caribe|cundinamarca|meta|casanare|qu[ií]nd[ií]o)\b",
            text, re.IGNORECASE,
        )
        if market_match:
            market_filter = [market_match.group(1)]

    try:
        q = sb.table("spread_vs_competitors").select("*")
        if market_filter:
            q = q.ilike("market", f"%{_normalize_market(market_filter[0])}%")
        rows = q.limit(60).execute().data or []

        if not rows and market_filter:
            rows = sb.table("spread_vs_competitors").select("*").limit(60).execute().data or []
    except Exception as e:
        return f"Vista spread_vs_competitors no disponible: {e}"

    if not rows:
        return "No hay datos en spread_vs_competitors."

    competitors_present = sorted({r.get("competitor") for r in rows if r.get("competitor")})
    _ALL_KNOWN_COMPS = ["GNCC", "EXEC", "ENBC", "NEUC", "DLRC", "ETTC", "QIEC", "RTQC", "SCEC"]
    _ALL_OR_COMPS = ["EPM", "ENEL", "EMCALI", "AFINIA", "AIRE", "CELSIA TOLIMA", "CELSIA VALLE",
                     "EBSA", "CHEC", "ESSA", "EEP"]
    all_expected = _ALL_KNOWN_COMPS + _ALL_OR_COMPS
    missing_comps = [c for c in all_expected if c not in competitors_present]

    lines = [
        "=== Spread vs competidores comercializadores (spread_vs_competitors) ===",
        f"Competidores CON datos en la vista: {', '.join(competitors_present) if competitors_present else 'ninguno'}",
        f"Competidores SIN datos en la vista: {', '.join(missing_comps) if missing_comps else 'ninguno'}",
        f"OR como competidores (consulta directa disponible): {', '.join(_ALL_OR_COMPS)}",
        "AVISO: responde solo con los competidores que tienen datos — menciona explicitamente cuales faltan.",
    ]

    _COMP_ALIASES = {
        "GNCC": ["vatia", "gncc"], "EXEC": ["enel x", "exec"], "ENBC": ["enerbit", "enbc"],
        "NEUC": ["neu", "neuc"], "DLRC": ["diceler", "dlrc"], "ETTC": ["enertotal", "ettc"],
        "QIEC": ["qi energy", "qiec"], "RTQC": ["ruitoque", "rtqc"], "SCEC": ["sol", "scec"],
    }
    asked_comp = next(
        (code for code, aliases in _COMP_ALIASES.items()
         if any(a in text_l for a in aliases)),
        None,
    )
    if asked_comp and asked_comp not in competitors_present:
        lines.append(
            f"AVISO: el competidor solicitado ({asked_comp}) no tiene datos en esta vista. "
            f"Competidores con datos: {', '.join(competitors_present)}"
        )

    for r in rows[:60]:
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    if len(rows) > 60:
        lines.append(f"  ... y {len(rows) - 60} filas mas.")
    return "\n".join(lines)


def _ctx_tandem(text: str) -> str:
    """
    Builds tándem context: qc of BIA vs the OR reference group.
    Both queries are scoped to a single run_id to avoid full-table scans on
    the 500k+ row simulation_results table.
    """
    sb = get_supabase()

    # 1. Resolve the most recent official BIA run
    bia_run = _resolve_run(text, "BIA")
    if not bia_run:
        return "=== Tándem ===\nNo se encontró corrida oficial de BIA para calcular tándem."

    bia_run_id = bia_run["id"]
    bia_base   = bia_run.get("base_period", "")
    log.info("TANDEM BIA run resuelto | run_id=%s base=%s oficial=%s", bia_run_id, bia_base, bia_run.get("is_official"))

    # 2. Resolve the most recent official OR run (agent_code='OR' is the batch OR run)
    or_run = _resolve_run(text, "OR")
    if not or_run:
        return "=== Tándem ===\nNo se encontró corrida oficial de OR para calcular tándem."

    or_run_id = or_run["id"]

    # 3. Fetch qc for BIA via SQL function (SELECT DISTINCT period, qc).
    # Falls back to paginated scan if the function doesn't exist yet.
    bia_rows = []
    try:
        rpc_rows = sb.rpc("get_bia_qc_by_period", {"p_run_id": bia_run_id}).execute().data or []
        if rpc_rows:
            bia_rows = rpc_rows
            log.info("TANDEM BIA rpc | run_id=%s periodos=%d", bia_run_id, len(bia_rows))
    except Exception as rpc_err:
        log.warning("TANDEM BIA rpc unavailable (%s) — falling back to paginated scan", rpc_err)

    if not bia_rows:
        # Fallback: paginate in blocks of 100 until we have all 12 distinct periods
        PAGE = 100
        offset = 0
        seen_periods: set[str] = set()
        while True:
            try:
                chunk = (
                    sb.table("simulation_results")
                    .select("period,qc")
                    .eq("run_id", bia_run_id)
                    .not_.is_("qc", "null")
                    .order("period")
                    .range(offset, offset + PAGE - 1)
                    .execute()
                    .data or []
                )
            except Exception as e:
                return f"=== Tándem ===\nError consultando qc de BIA: {e}"
            if not chunk:
                break
            bia_rows.extend(chunk)
            new_periods = {r["period"] for r in chunk if r.get("period")}
            seen_periods |= new_periods
            # Stop once qc stops changing (same period repeated) — means we've seen all
            if len(chunk) < PAGE:
                break
            offset += PAGE
        log.info("TANDEM BIA fallback scan | run_id=%s filas_raw=%d", bia_run_id, len(bia_rows))

    if not bia_rows:
        log.warning("TANDEM BIA sin datos | run_id=%s", bia_run_id)
        return "=== Tándem ===\nSin datos de qc para BIA en la corrida oficial."

    # Build BIA qc dict: {period: qc_pct} — qc is uniform across all variants,
    # take first occurrence per period (dedup). Convert ratio → percentage.
    bia_qc: dict[str, float] = {}
    for r in bia_rows:
        if r.get("qc") is not None and r["period"] not in bia_qc:
            bia_qc[r["period"]] = float(r["qc"]) * 100.0

    log.info("TANDEM BIA periodos deduplicados: %d | periodos=%s", len(bia_qc), sorted(bia_qc.keys()))

    # 4. Fetch qc for OR reference group — scoped to or_run_id + or_code filter
    try:
        or_rows = (
            sb.table("simulation_results")
            .select("or_code,period,qc")
            .eq("run_id", or_run_id)
            .in_("or_code", _TANDEM_OR_REFS)
            .eq("tension_level", 2)
            .eq("rate_type", "USER")
            .eq("pb_scenario", "MEDIUM")
            .order("period")
            .limit(200)
            .execute()
            .data or []
        )
    except Exception as e:
        return f"=== Tándem ===\nError consultando qc de OR de referencia: {e}"

    # 5. Calculate per-period average qc of the OR reference group (converted to %)
    or_by_period: dict[str, list[float]] = defaultdict(list)
    for r in or_rows:
        if r.get("qc") is not None and r.get("period") and r.get("or_code") in _TANDEM_OR_REFS:
            or_by_period[r["period"]].append(float(r["qc"]) * 100.0)

    # 6. Build output — one row per period
    lines = [
        f"=== Tándem — posición de cobertura BIA vs OR de referencia ===",
        f"Corrida BIA: #{bia_run_id} | base_period={bia_base} | "
        f"fecha={bia_run.get('created_at','')[:10]} | "
        f"por={bia_run.get('triggered_by','')} | "
        f"oficial={'sí' if bia_run.get('is_official') else 'no'}",
        f"Corrida OR:  #{or_run_id} | base_period={or_run.get('base_period','')} | "
        f"fecha={or_run.get('created_at','')[:10]} | "
        f"oficial={'sí' if or_run.get('is_official') else 'no'}",
        f"OR de referencia: {', '.join(_TANDEM_OR_REFS)}",
        f"(escenario: MEDIUM | tension_level=2 | rate_type=USER | qc expresado en %)",
        "",
        f"{'Periodo':<10} {'qc_BIA':>8} {'qc_OR_avg':>10} {'banda_low':>10} {'banda_high':>11} {'estado':>12}",
        "-" * 65,
    ]

    all_periods = sorted(set(bia_qc.keys()) | set(or_by_period.keys()))
    for period in all_periods:
        qc_bia  = bia_qc.get(period)
        or_vals = or_by_period.get(period, [])
        if not or_vals:
            qc_or_avg = band_low = band_high = None
            estado = "sin datos OR"
        else:
            qc_or_avg = sum(or_vals) / len(or_vals)
            band_low  = qc_or_avg - 5.0
            band_high = qc_or_avg + 5.0
            if qc_bia is None:
                estado = "sin datos BIA"
            elif qc_bia < band_low:
                estado = "⬇ BAJO banda"
            elif qc_bia > band_high:
                estado = "⬆ SOBRE banda"
            else:
                estado = "✅ en banda"

        bia_str  = f"{qc_bia:.1f}%"  if qc_bia    is not None else "  N/A"
        avg_str  = f"{qc_or_avg:.1f}%" if qc_or_avg is not None else "  N/A"
        low_str  = f"{band_low:.1f}%"  if band_low  is not None else "  N/A"
        high_str = f"{band_high:.1f}%" if band_high  is not None else "  N/A"
        lines.append(f"{period:<10} {bia_str:>8} {avg_str:>10} {low_str:>10} {high_str:>11} {estado:>12}")

    # Summary counts for the LLM
    n_below = sum(1 for p in all_periods
                  if bia_qc.get(p) is not None and or_by_period.get(p)
                  and bia_qc[p] < (sum(or_by_period[p]) / len(or_by_period[p])) - 5.0)
    n_above = sum(1 for p in all_periods
                  if bia_qc.get(p) is not None and or_by_period.get(p)
                  and bia_qc[p] > (sum(or_by_period[p]) / len(or_by_period[p])) + 5.0)
    n_in    = len(all_periods) - n_below - n_above
    lines.append("")
    lines.append(f"Resumen: {n_in} periodos en banda | {n_below} por debajo | {n_above} por encima")

    return "\n".join(lines)


def _ctx_cu_compare(run_a: dict, run_b: dict, all_periods: bool = False) -> str:
    """
    Compare two simulation runs side-by-side: CU and G delta per (market, period, scenario).
    Aggregation done in DB via RPC; falls back to direct query capped at 200 rows per run.
    """
    sb = get_supabase()

    def _fetch(run: dict) -> dict:
        """Fetch avg CU/G per (market, period, scenario) — max 200 rows."""
        rows = (
            sb.table("simulation_results")
            .select("market,period,pb_scenario,cu,g,c,t,d,p,r,aj")
            .eq("run_id", run["id"])
            .eq("tension_level", 2)
            .eq("rate_type", "USER")
            .order("period").order("market").order("pb_scenario")
            .limit(200)
            .execute()
            .data or []
        )
        # Average duplicates (same market/period/scenario) in Python
        groups: dict = defaultdict(list)
        for r in rows:
            groups[(r["market"], r["period"], r["pb_scenario"])].append(r)
        index: dict = {}
        for key, entries in groups.items():
            def _avg(col):
                vals = [float(e[col]) for e in entries if e.get(col) is not None]
                return round(sum(vals) / len(vals), 2) if vals else 0.0
            index[key] = {c: _avg(c) for c in ("cu", "g", "c", "t", "d", "p", "r")}
        return index

    idx_a = _fetch(run_a)
    idx_b = _fetch(run_b)

    # Fix 1: is_official comes from the simulation_runs row, not simulation_results
    def _run_label(run: dict) -> str:
        who     = (run.get("triggered_by") or "desconocido").split("@")[0]
        oficial = "sí" if run.get("is_official") else "no"
        return (
            f"#{run['id']} | {run.get('agent_code')} | base {run.get('base_period')} | "
            f"{run.get('created_at','')[:10]} | {who} | oficial: {oficial}"
        )

    if not idx_a:
        return f"=== Comparación de corridas ===\nSin datos para corrida #{run_a['id']}."
    if not idx_b:
        return f"=== Comparación de corridas ===\nSin datos para corrida #{run_b['id']}."

    common_keys = sorted(set(idx_a) & set(idx_b))
    if not common_keys:
        return (
            f"=== Comparación de corridas ===\n"
            f"Sin combinaciones en común entre #{run_a['id']} y #{run_b['id']}.\n"
            f"Corrida A: {len(idx_a)} filas | Corrida B: {len(idx_b)} filas."
        )

    # Fix 3: cap at 50 combos, prioritizing 2026 periods when all_periods requested
    MAX_COMBOS = 50
    if len(common_keys) > MAX_COMBOS:
        if all_periods:
            priority = [k for k in common_keys if "-2026" in k[1]]
            rest     = [k for k in common_keys if "-2026" not in k[1]]
            common_keys = (priority + rest)[:MAX_COMBOS]
        else:
            common_keys = common_keys[:MAX_COMBOS]

    lines = [
        "=== Comparación de corridas — CU y G delta ===",
        f"Corrida A: {_run_label(run_a)}",
        f"Corrida B: {_run_label(run_b)}",
        f"(tension_level=2, rate_type=USER | mostrando {len(common_keys)} de {len(set(idx_a)&set(idx_b))} combinaciones)",
        "",
    ]

    # Group by (market, period) — list all scenarios per group
    seen: set = set()
    for k in common_keys:
        market, period = k[0], k[1]
        if (market, period) in seen:
            continue
        seen.add((market, period))
        scenarios = sorted({kk[2] for kk in common_keys if kk[0] == market and kk[1] == period})
        parts = []
        for sc in scenarios:
            r_a = idx_a.get((market, period, sc), {})
            r_b = idx_b.get((market, period, sc), {})
            # Fix 2: compare G explicitly, not just CU
            cu_a = round(float(r_a.get("cu") or 0), 2)
            cu_b = round(float(r_b.get("cu") or 0), 2)
            g_a  = round(float(r_a.get("g")  or 0), 2)
            g_b  = round(float(r_b.get("g")  or 0), 2)
            dcu  = round(cu_b - cu_a, 2)
            dg   = round(g_b  - g_a,  2)
            s_cu = f"+{dcu}" if dcu > 0 else str(dcu)
            s_g  = f"+{dg}"  if dg  > 0 else str(dg)
            parts.append(f"{sc}: CU A={cu_a} B={cu_b} Δ={s_cu} | G A={g_a} B={g_b} Δ={s_g}")
        lines.append(f"  {market} | {period} | " + " | ".join(parts))

    log.info("CU COMPARE | run_a=%s run_b=%s shown=%d total=%d",
             run_a["id"], run_b["id"], len(common_keys), len(set(idx_a) & set(idx_b)))
    return "\n".join(lines)


def build_context(user_text: str) -> str:
    """Route to the right data sources based on the question's intent."""
    want_cu         = bool(_CU_KEYWORDS.search(user_text))
    want_spread     = bool(_SPREAD_KEYWORDS.search(user_text))
    want_runs       = bool(_RUNS_KEYWORDS.search(user_text))
    want_tandem     = bool(_TANDEM_KEYWORDS.search(user_text))
    want_or_ranking = bool(_OR_RANKING_KEYWORDS.search(user_text))

    if not any([want_cu, want_spread, want_runs, want_tandem]):
        want_runs = True

    # Detect comparison intent
    raw_names = re.findall(
        r"(?:corri[oó]|ejecut[oó]|corrida\s+de|la\s+de|de)\s+([a-záéíóúñ]{3,})",
        user_text.lower(),
    )
    compare_names = [n for n in dict.fromkeys(raw_names) if n not in _EXCLUDED_PERSON_WORDS]

    # "últimas dos corridas" / "corrida anterior" / "corrida previa" without person names
    wants_last_two = bool(re.search(
        r"\b([uú]ltimas?\s+(dos|2)\s+corridas?|corrida\s+(anterior|previa)|"
        r"[uú]ltimas?\s+dos\s+corridas?|anterior\s+corrida|comparar\s+corridas?)\b",
        user_text, re.IGNORECASE,
    )) and not compare_names

    # Fix 2: fetch all periods when question references 2026 or asks for full history
    wants_all_periods = bool(re.search(
        r"\b(2026|todos los periodos?|histor|completo|completa|todos los meses)\b",
        user_text, re.IGNORECASE,
    ))

    is_compare = len(compare_names) >= 2 or wants_last_two

    sections: list[str] = []
    if want_runs:
        sections.append(_ctx_simulation_runs(user_text))

    if is_compare:
        sb = get_supabase()
        runs_by_name: list[dict] = []

        if wants_last_two:
            # Auto-resolve: two most recent official BIA runs
            official_runs = (
                sb.table("simulation_runs")
                .select("id,agent_code,base_period,is_official,triggered_by,created_at,status")
                .eq("agent_code", "BIA")
                .eq("status", "COMPLETED")
                .eq("is_official", True)
                .order("created_at", desc=True)
                .limit(2)
                .execute()
                .data or []
            )
            runs_by_name = official_runs
            log.info("COMPARE last-two | runs=%s", [r['id'] for r in runs_by_name])
        else:
            for name in compare_names[:2]:
                r = (
                    sb.table("simulation_runs")
                    .select("id,agent_code,base_period,is_official,triggered_by,created_at,status")
                    .eq("agent_code", "BIA")
                    .eq("status", "COMPLETED")
                    .ilike("triggered_by", f"%{name}%")
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                    .data or []
                )
                if r:
                    runs_by_name.append(r[0])
            log.info("COMPARE | names=%s runs=%s", compare_names, [r['id'] for r in runs_by_name])

        if len(runs_by_name) == 2:
            sections.append(_ctx_cu_compare(runs_by_name[0], runs_by_name[1], all_periods=wants_all_periods))
        elif len(runs_by_name) == 1:
            who = "corrida anterior" if wants_last_two else compare_names[1]
            sections.append(
                f"Solo encontré corrida #{runs_by_name[0]['id']}. "
                f"No se encontró la {who} en simulation_runs."
            )
    elif want_cu:
        sections.append(_ctx_cu_components(user_text))

    if want_or_ranking:
        sections.append(_ctx_or_ranking_by_period(user_text))
    elif want_spread:
        sections.append(_ctx_spread(user_text))
    if want_tandem:
        sections.append(_ctx_tandem(user_text))

    return "\n\n".join(sections)


# ── Slack helpers ─────────────────────────────────────────────────────────────

async def send_slack_message(channel: str, text: str) -> None:
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": channel, "text": text},
        )


_MAX_CONTEXT_CHARS = 8000

async def handle_mention(event: dict) -> None:
    user_text = event.get("text", "")
    channel   = event.get("channel", SLACK_CHANNEL_ID)
    context   = build_context(user_text)

    # Hard cap: truncate at section boundaries to stay under memory/token limit
    if len(context) > _MAX_CONTEXT_CHARS:
        truncated = context[:_MAX_CONTEXT_CHARS]
        # Try to cut at a clean newline rather than mid-line
        last_nl = truncated.rfind("\n")
        if last_nl > _MAX_CONTEXT_CHARS * 0.8:
            truncated = truncated[:last_nl]
        context = truncated + f"\n[contexto truncado — {len(context)} chars totales, mostrando {len(truncated)}]"
        log.warning("CONTEXT TRUNCATED | original=%d chars", len(context))

    message = get_claude().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Datos de contexto:\n{context}\n\nPregunta: {user_text}",
            }
        ],
    )

    reply = message.content[0].text if message.content else "No pude generar una respuesta."
    await send_slack_message(channel, reply)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    t0 = datetime.now(timezone.utc)
    body = await request.json()

    # URL verification challenge
    if body.get("type") == "url_verification":
        log.info("SLACK challenge received — responding immediately")
        return JSONResponse({"challenge": body["challenge"]})

    event    = body.get("event", {})
    event_ts = event.get("event_ts") or event.get("ts", "")

    log.info("SLACK event received | type=%s event_ts=%s t=%s",
             event.get("type"), event_ts, t0.isoformat())

    if event.get("type") == "app_mention":
        if event_ts and _is_duplicate(event_ts):
            return Response(status_code=200)

        log.info("PROCESSING: %s", event_ts)
        background_tasks.add_task(handle_mention, event)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("SLACK 200 sent | event_ts=%s elapsed=%.3fs", event_ts, elapsed)
    return Response(status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/debug/dedup")
async def debug_dedup(event_ts: str = ""):
    sb = get_supabase()
    try:
        count_res = sb.table("processed_events").select("event_ts", count="exact").execute()
        total = count_res.count

        recent = (
            sb.table("processed_events")
            .select("event_ts,processed_at")
            .order("processed_at", desc=True)
            .limit(5)
            .execute()
            .data or []
        )

        test_result = None
        if event_ts:
            exists = (
                sb.table("processed_events")
                .select("event_ts,processed_at")
                .eq("event_ts", event_ts)
                .execute()
                .data or []
            )
            test_result = exists[0] if exists else "not found"

        return {
            "total_rows": total,
            "recent_5": recent,
            "query_event_ts": event_ts or None,
            "query_result": test_result,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/debug/tandem")
async def debug_tandem():
    """
    Diagnostic: runs SELECT DISTINCT period, qc FROM simulation_results
    WHERE run_id=21511 AND qc IS NOT NULL ORDER BY period
    and returns the raw rows as JSON.
    """
    sb = get_supabase()
    try:
        rows = (
            sb.table("simulation_results")
            .select("period,qc")
            .eq("run_id", 21511)
            .not_.is_("qc", "null")
            .order("period")
            .limit(500)
            .execute()
            .data or []
        )
        # Deduplicate by period (same as _ctx_tandem)
        seen: dict[str, float] = {}
        for r in rows:
            if r.get("period") and r.get("qc") is not None and r["period"] not in seen:
                seen[r["period"]] = float(r["qc"])

        return {
            "run_id": 21511,
            "raw_row_count": len(rows),
            "distinct_periods": len(seen),
            "periods": [
                {"period": p, "qc_ratio": v, "qc_pct": round(v * 100, 2)}
                for p, v in sorted(seen.items())
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
