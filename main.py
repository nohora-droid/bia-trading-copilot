import os
import re
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


def _ctx_cu_components(text: str) -> str:
    sb = get_supabase()

    # Extract agent_code (e.g. NEUC, BIA, EXEC, GNCC, OR) and base_period (MM-YYYY)
    agent_match = re.search(r"\b(NEUC|BIA|EXEC|GNCC|OR)\b", text, re.IGNORECASE)
    period_match = re.search(r"\b(\d{2}-\d{4})\b", text)

    query = sb.table("simulation_results").select(
        "agent_code,base_period,market,period,pb_scenario,tension_level,rate_type,"
        "cu,g,c,t,d,p,r,g_base,g_transitorio,aj,alpha"
    )

    if agent_match:
        query = query.eq("agent_code", agent_match.group(1).upper())
    if period_match:
        query = query.eq("base_period", period_match.group(1))

    rows = query.order("period").limit(100).execute().data or []

    if not rows:
        filters = []
        if agent_match:
            filters.append(f"agent_code={agent_match.group(1).upper()}")
        if period_match:
            filters.append(f"base_period={period_match.group(1)}")
        hint = f" (filtros: {', '.join(filters)})" if filters else ""
        return f"No se encontraron resultados en simulation_results{hint}."

    # Summarize: group by market+period, show avg CU and components
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in rows:
        key = (r["market"], r["period"], r["pb_scenario"])
        groups[key].append(r)

    lines = [
        f"=== CU y componentes — {rows[0]['agent_code']} base {rows[0]['base_period']} ===",
        f"({len(rows)} registros, {len(groups)} combinaciones mercado/período/escenario)",
    ]
    for (market, period, scenario), entries in list(groups.items())[:30]:
        avg = lambda col: round(sum(e[col] or 0 for e in entries) / len(entries), 2)
        lines.append(
            f"  {market} | {period} | {scenario}: "
            f"CU={avg('cu')} G={avg('g')} C={avg('c')} "
            f"T={avg('t')} D={avg('d')} P={avg('p')} R={avg('r')}"
        )
    if len(groups) > 30:
        lines.append(f"  ... y {len(groups) - 30} combinaciones más.")
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
