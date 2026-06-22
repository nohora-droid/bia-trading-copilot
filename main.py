import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx
from supabase import create_client, Client
import anthropic
from dotenv import load_dotenv

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


# In-memory dedup set — cleared on restart, sufficient for most duplicate scenarios
processed_events: set[str] = set()

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


def build_context(user_text: str) -> str:
    """Route to the right data sources based on the question's intent."""
    want_cu     = bool(_CU_KEYWORDS.search(user_text))
    want_spread = bool(_SPREAD_KEYWORDS.search(user_text))
    want_runs   = bool(_RUNS_KEYWORDS.search(user_text))

    if not any([want_cu, want_spread, want_runs]):
        want_runs = True

    sections: list[str] = []
    if want_runs:
        sections.append(_ctx_simulation_runs(user_text))
    if want_cu:
        sections.append(_ctx_cu_components(user_text))
    if want_spread:
        sections.append(_ctx_spread(user_text))

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
    event_ts = event.get("event_ts") or event.get("ts", "")
    if event_ts in processed_events:
        return
    processed_events.add(event_ts)

    user_text = event.get("text", "")
    channel   = event.get("channel", SLACK_CHANNEL_ID)
    context   = build_context(user_text)

    message = get_claude().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "Eres el BIA AI Trading Copilot, asistente especializado en trading de energia electrica en Colombia. "
            "Tienes acceso a datos reales de simulaciones (CU y componentes G, C, T, D, P, R), "
            "corridas de simulacion y spread competitivo. "
            "IMPORTANTE: En TODA respuesta sobre tarifas o resultados, incluye siempre la linea de trazabilidad "
            "de la corrida usada (Corrida #ID | agente | base | fecha | por quien | oficial: si/no). "
            "Cuando respondas sobre CU, menciona mercado, periodo proyectado y escenario de precio de bolsa. "
            "Se conciso y preciso. Usa formato Slack (*negrita*, listas con -). "
            "Responde en el mismo idioma que el usuario."
        ),
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
    body = await request.json()

    if body.get("type") == "url_verification":
        return JSONResponse({"challenge": body["challenge"]})

    event = body.get("event", {})
    if event.get("type") == "app_mention":
        background_tasks.add_task(handle_mention, event)

    return JSONResponse({"ok": True})


@app.get("/health")
async def health():
    return {"status": "ok"}
