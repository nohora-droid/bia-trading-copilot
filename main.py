import os
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


async def get_simulation_context() -> str:
    result = get_supabase().table("simulation_runs").select("*").order("created_at", desc=True).limit(20).execute()
    runs = result.data or []
    if not runs:
        return "No hay simulaciones recientes disponibles."
    lines = ["Simulaciones recientes (simulation_runs):"]
    for r in runs:
        lines.append(
            f"- run_id={r.get('id')} status={r.get('status')} "
            f"created_at={r.get('created_at')} params={r.get('params')}"
        )
    return "\n".join(lines)


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

    context = await get_simulation_context()

    message = get_claude().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "Eres el BIA AI Trading Copilot, un asistente especializado en trading de energía. "
            "Respondes preguntas sobre simulaciones, posiciones de compra/venta y métricas energéticas. "
            "Sé conciso y preciso. Responde en el mismo idioma que el usuario."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Contexto de simulaciones:\n{context}\n\nPregunta del usuario: {user_text}",
            }
        ],
    )

    reply = message.content[0].text if message.content else "No pude generar una respuesta."
    await send_slack_message(channel, reply)


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
