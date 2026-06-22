import os
import re
import unicodedata
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

# ── Intent detection keywords ─────────────────────────────────────────────────

_CU_KEYWORDS = re.compile(
    r"\b(cu|costo unitario|tarifa|componente|g\b|c\b|t\b|d\b|p\b|r\b|g_base|desglose)\b",
    re.IGNORECASE,
)
_SPREAD_KEYWORDS = re.compile(
    r"\b(spread|competidor|competencia|competitiv|rival|vs\b|versus|comparar|diferencia)\b",
    re.IGNORECASE,
)
_RUNS_KEYWORDS = re.compile(
    r"\b(corrida|simulaci[oó]n|run|reciente|[uú]ltim[ao]|ejecut)\b",
    re.IGNORECASE,
)

# ── Context builders ──────────────────────────────────────────────────────────

def _ctx_simulation_runs() -> str:
    rows = (
        get_supabase()
        .table("simulation_runs")
        .select("id,status,created_at,params")
        .order("created_at", desc=True)
        .limit(20)
        .execute()
        .data or []
    )
    if not rows:
        return "No hay corridas recientes en simulation_runs."
    lines = ["=== Corridas recientes (simulation_runs) ==="]
    for r in rows:
        lines.append(
            f"run_id={r.get('id')} status={r.get('status')} "
            f"created_at={r.get('created_at')} params={r.get('params')}"
        )
    return "\n".join(lines)


_SCENARIO_MAP = {"BAJO": "LOW", "MEDIO": "MEDIUM", "ALTO": "HIGH"}

# Known market names that may appear in questions
_MARKETS = (
    "ANTIOQUIA|BOGOTA|BOYACA|CALDAS|CALI|CARIBE MAR|CARIBE SOL|CARTAGO|CASANARE|"
    "CUNDINAMARCA|MEDELLIN|NARINO|NARIÑO|SANTANDER|TOLIMA|VALLE|COSTA|LLANOS|SUROCCIDENTE"
)

# Spanish month names → MM number
_MONTH_NAMES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}


def _parse_period(text: str) -> str | None:
    """Extract a projected period (MM-YYYY) from natural language or numeric form."""
    # Numeric: 05-2026 or 05/2026
    m = re.search(r"\b(\d{2})[-/](\d{4})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Spanish month name + year: "mayo 2026", "mayo de 2026"
    m = re.search(
        r"\b(" + "|".join(_MONTH_NAMES) + r")\b\s+(?:de\s+)?(\d{4})\b",
        text, re.IGNORECASE,
    )
    if m:
        return f"{_MONTH_NAMES[m.group(1).lower()]}-{m.group(2)}"
    return None


def _latest_base_period(sb, agent_code: str) -> str:
    """Return the most recent base_period available for this agent."""
    rows = (
        sb.table("simulation_results")
        .select("base_period")
        .eq("agent_code", agent_code)
        .limit(2000)
        .execute()
        .data or []
    )
    periods = sorted({r["base_period"] for r in rows}, reverse=True)
    return periods[0] if periods else "05-2026"


def _next_n_periods(sb, base_period: str, agent_code: str, n: int = 3) -> list[str]:
    """Return the first n projected periods available for this base_period."""
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

    # ── Extract filters from the question ────────────────────────────────────
    agent_match = re.search(r"\b(NEUC|BIA|EXEC|GNCC|OR)\b", text, re.IGNORECASE)
    scen_match  = re.search(r"\b(LOW|MEDIUM|HIGH|BAJO|MEDIO|ALTO)\b", text, re.IGNORECASE)

    # Normalize accents before matching markets (e.g. "Bogotá" → "Bogota")
    text_norm = unicodedata.normalize("NFD", text)
    text_norm = "".join(c for c in text_norm if unicodedata.category(c) != "Mn")
    mkt_match = re.search(_MARKETS, text_norm, re.IGNORECASE)

    # period = projected month the user is asking about; default → next 3 months
    period_asked  = _parse_period(text_norm)
    agent_code    = agent_match.group(1).upper() if agent_match else "BIA"  # default BIA
    scenario_asked = (
        _SCENARIO_MAP.get(scen_match.group(1).upper(), scen_match.group(1).upper())
        if scen_match else None
    )
    market_filter = mkt_match.group(0).upper() if mkt_match else None

    # ── Step 1: use the most recent base_period as data source ───────────────
    base_period = _latest_base_period(sb, agent_code)

    # ── Step 2: discover available pb_scenario for this run ──────────────────
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
        return f"No hay datos en simulation_results para agent_code={agent_code}, base_period={base_period}."

    # ── Step 3: resolve scenario ─────────────────────────────────────────────
    scenario_warning: str | None = None
    if scenario_asked:
        if scenario_asked in available:
            scenarios_to_fetch = [scenario_asked]
        else:
            scenarios_to_fetch = available
            scenario_warning = (
                f"No encontre el escenario *{scenario_asked}* para {agent_code}. "
                f"Te muestro los disponibles: *{', '.join(available)}*."
            )
    else:
        scenarios_to_fetch = available

    # ── Step 4: resolve periods to show ──────────────────────────────────────
    if period_asked:
        periods_to_fetch = [period_asked]
        period_label = f"period={period_asked}"
    else:
        periods_to_fetch = _next_n_periods(sb, base_period, agent_code, n=3)
        period_label = f"proximos 3 periodos ({', '.join(periods_to_fetch)})"

    # ── Step 5: main data query ───────────────────────────────────────────────
    # tension_level=2 + rate_type=USER as canonical representative slice.
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

    # ── Step 6: group and format ──────────────────────────────────────────────
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["market"], r["period"], r["pb_scenario"])].append(r)

    lines: list[str] = []
    if scenario_warning:
        lines.append(scenario_warning)

    lines += [
        f"=== CU y componentes — {agent_code} | corrida base_period={base_period} | {period_label}"
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
        lines.append(f"  ... y {len(rows) - 30} filas más.")
    return "\n".join(lines)


def build_context(user_text: str) -> str:
    """Route to the right data sources based on the question's intent."""
    sections: list[str] = []

    want_cu = bool(_CU_KEYWORDS.search(user_text))
    want_spread = bool(_SPREAD_KEYWORDS.search(user_text))
    want_runs = bool(_RUNS_KEYWORDS.search(user_text))

    # Default: show recent runs if no specific intent detected
    if not any([want_cu, want_spread, want_runs]):
        want_runs = True

    if want_runs:
        sections.append(_ctx_simulation_runs())
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
    channel = event.get("channel", SLACK_CHANNEL_ID)

    context = build_context(user_text)

    message = get_claude().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "Eres el BIA AI Trading Copilot, un asistente especializado en trading de energía eléctrica en Colombia. "
            "Tienes acceso a datos reales de simulaciones de tarifas (CU y sus componentes G, C, T, D, P, R), "
            "corridas de simulación recientes y spread competitivo vs otros agentes del mercado. "
            "Cuando respondas sobre CU o componentes, menciona el mercado, período y escenario de precio de bolsa. "
            "Sé conciso y preciso. Usa formato Slack (*negrita*, _cursiva_, listas con -). "
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
