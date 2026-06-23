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
    Returns True if this event_ts was already processed.
    Uses upsert with ON CONFLICT DO NOTHING (ignore_duplicates=True).
    PostgREST returns an empty data list when the row already existed,
    and a one-element list when it was freshly inserted — that's our mutex.
    Any infra error lets the event through so messages are never silently dropped.
    """
    sb = get_supabase()
    try:
        result = (
            sb.table("processed_events")
            .upsert({"event_ts": event_ts}, on_conflict="event_ts", ignore_duplicates=True)
            .execute()
        )
        inserted = bool(result.data)  # empty list = conflict (already existed)
        if inserted:
            log.info("DEDUP INSERT OK | event_ts=%s", event_ts)
            # Cleanup stale records — ignore errors
            try:
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
                sb.table("processed_events").delete().lt("processed_at", cutoff).execute()
            except Exception:
                pass
            return False  # new event — process it
        else:
            log.warning("DUPLICATE BLOCKED (upsert no-op): %s", event_ts)
            return True  # already existed — duplicate
    except Exception as e:
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
    r"\b(cu|costo unitario|tarifa|componente|g\b|c\b|t\b|d\b|p\b|r\b|g_base|desglose)\b",
    re.IGNORECASE,
)
_SPREAD_KEYWORDS = re.compile(
    r"\b(spread|competidor|competencia|competitiv|rival|vs\b|versus|comparar|diferencia)\b",
    re.IGNORECASE,
)
_RUNS_KEYWORDS = re.compile(
    r"\b(corrida|simulaci[oó]n|run|reciente|[uú]ltim[ao]|ejecut|qui[eé]n corri[oó]|hoy|ayer)\b",
    re.IGNORECASE,
)
_TANDEM_KEYWORDS = re.compile(
    r"\b(t[aá]ndem|tandem|cobertura|banda|posici[oó]n|qc)\b",
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

    # Person detection: exclude agent codes so "de BIA" doesn't match as a person
    person_match = re.search(
        r"(?:corri[oó]|ejecut[oó])\s+([a-záéíóúñ]+)",
        text_lower,
    )
    if person_match and person_match.group(1).lower() in _AGENT_CODES:
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

    # Strategy 1 — person name
    if person_match:
        name = person_match.group(1).strip()
        rows = q().ilike("triggered_by", f"{name}%").order("created_at", desc=True).limit(1).execute().data or []
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

    # Detect person filter
    person_match = re.search(r"(?:corri[oó]|ejecut[oó]|de)\s+([a-záéíóúñ]+)", text_lower)
    wants_today     = bool(re.search(r"\bhoy\b", text_lower))
    wants_yesterday = bool(re.search(r"\bayer\b", text_lower))

    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    q = (
        sb.table("simulation_runs")
        .select("id,agent_code,base_period,is_official,triggered_by,created_at,status")
        .eq("status", "COMPLETED")
    )

    if person_match:
        q = q.ilike("triggered_by", f"{person_match.group(1).strip()}%")
    elif wants_today:
        q = q.gte("created_at", today_start.isoformat())
    elif wants_yesterday:
        q = q.gte("created_at", yesterday_start.isoformat()).lt("created_at", today_start.isoformat())

    rows = q.order("created_at", desc=True).limit(20).execute().data or []

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


def _next_n_periods(sb, base_period: str, agent_code: str, n: int = 3) -> list[str]:
    rows = (
        sb.table("simulation_results")
        .select("period")
        .eq("base_period", base_period)
        .eq("agent_code", agent_code)
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

    # ── Discover available pb_scenario ───────────────────────────────────────
    avail_rows = (
        sb.table("simulation_results")
        .select("pb_scenario")
        .eq("base_period", base_period)
        .eq("agent_code", agent_code)
        .limit(2000)
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
    if period_asked:
        periods_to_fetch = [period_asked]
        period_label = f"period={period_asked}"
    else:
        periods_to_fetch = _next_n_periods(sb, base_period, agent_code, n=3)
        period_label = f"proximos 3 periodos ({', '.join(periods_to_fetch)})"

    # ── Main data query (tension_level=2, rate_type=USER as canonical slice) ──
    query = (
        sb.table("simulation_results")
        .select(
            "agent_code,base_period,market,period,pb_scenario,"
            "cu,g,c,t,d,p,r,g_base,g_transitorio,aj,alpha"
        )
        .eq("base_period", base_period)
        .eq("agent_code", agent_code)
        .eq("tension_level", 2)
        .eq("rate_type", "USER")
        .in_("pb_scenario", scenarios_to_fetch)
        .in_("period", periods_to_fetch)
    )
    if market_filter:
        query = query.ilike("market", f"%{market_filter}%")

    rows = query.order("period").order("market").order("pb_scenario").limit(1500).execute().data or []

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


def _ctx_spread(text: str) -> str:
    sb = get_supabase()
    try:
        rows = (
            sb.table("spread_vs_competitors")
            .select("*")
            .limit(60)
            .execute()
            .data or []
        )
    except Exception as e:
        return f"Vista spread_vs_competitors no disponible: {e}"

    if not rows:
        return "No hay datos en spread_vs_competitors."

    lines = ["=== Spread vs competidores (spread_vs_competitors) ==="]
    for r in rows[:30]:
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    if len(rows) > 30:
        lines.append(f"  ... y {len(rows) - 30} filas mas.")
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
        # Fallback: paginate in blocks of 500 until we have all 12 distinct periods
        PAGE = 500
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


def build_context(user_text: str) -> str:
    """Route to the right data sources based on the question's intent."""
    want_cu     = bool(_CU_KEYWORDS.search(user_text))
    want_spread = bool(_SPREAD_KEYWORDS.search(user_text))
    want_runs   = bool(_RUNS_KEYWORDS.search(user_text))
    want_tandem = bool(_TANDEM_KEYWORDS.search(user_text))

    if not any([want_cu, want_spread, want_runs, want_tandem]):
        want_runs = True

    sections: list[str] = []
    if want_runs:
        sections.append(_ctx_simulation_runs(user_text))
    if want_cu:
        sections.append(_ctx_cu_components(user_text))
    if want_spread:
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


async def handle_mention(event: dict) -> None:
    user_text = event.get("text", "")
    channel   = event.get("channel", SLACK_CHANNEL_ID)
    context   = build_context(user_text)

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
        # Claim atomically via Supabase PK — distributed mutex across all instances
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
