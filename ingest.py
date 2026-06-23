"""
Ingesta de resultados de simulación desde la API de Olibia Pricing hacia Supabase.
Uso: python ingest.py
"""

import os
import asyncio
from datetime import datetime, timezone
import httpx
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OLIBIA_API_KEY = "bia_06257853e300d9ad3e81e1f8124e4041d9712bcbb6a00d04679ac54db56e7a8c746061a817db8ecbde622f5a6cc071c58f714d37741e06e7c83406b095350f00"
OLIBIA_BASE_URL = "https://integrations.bia.app/ms-olibia-pricing/v1/simulations"
BATCH_SIZE = 5

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_run_ids() -> list[int]:
    result = supabase.table("simulation_runs").select("id").execute()
    return [row["id"] for row in (result.data or [])]


async def fetch_batch(client: httpx.AsyncClient, run_ids: list[int]) -> list[dict]:
    ids_param = ",".join(str(i) for i in run_ids)
    response = await client.get(
        f"{OLIBIA_BASE_URL}/{ids_param}",
        headers={"api-key": OLIBIA_API_KEY},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("runs", []) if isinstance(data, dict) else data


# OR runs return market names instead of or_code — map them here
_MARKET_TO_OR_CODE = {
    "ANTIOQUIA": "EPM",
    "BOGOTA": "ENEL",
    "CUNDINAMARCA": "ENEL",
    "BOYACA": "EBSA",
    "CALDAS": "CHEC",
    "CALI": "EMCALI",
    "YUMBO": "EMCALI",
    "CARIBE MAR": "AFINIA",
    "CARIBE SOL": "AIRE",
    "CARTAGO": "EEP",
    "PEREIRA": "EEP",
    "RISARALDA": "EEP",
    "CASANARE": "ENERCA",
    "CAUCA": "CEO",
    "HUILA": "ELECTROHUILA",
    "META": "EMSA",
    "NARIÑO": "CEDENAR",
    "NORTE SANTANDER": "CENS",
    "QUINDIO": "EDEQ",
    "SANTANDER": "ESSA",
    "TOLIMA": "CELSIA TOLIMA",
    "TULUA": "CETSA",
    "VALLE": "CELSIA VALLE",
}


def flatten_run(run: dict, fetched_at: str) -> list[dict]:
    """Explode a run into one row per market+result combination."""
    rows = []
    run_id = run["run_id"]
    agent_code = run.get("agent_code", "")
    base_period = run.get("base_period", "")

    for market_entry in run.get("results_by_market", []):
        market = market_entry.get("market")
        # API returns or_code=None for OR runs — resolve from market name
        or_code = market_entry.get("or_code") or _MARKET_TO_OR_CODE.get(
            (market or "").upper().strip(), None
        )

        for res in market_entry.get("results", []):
            rows.append({
                "run_id": run_id,
                "agent_code": agent_code,
                "base_period": base_period,
                "market": market,
                "or_code": or_code,
                "period": res.get("period"),
                "pb_scenario": res.get("pb_scenario"),
                "tension_level": res.get("tension_level"),
                "rate_type": res.get("rate_type"),
                "cu": res.get("cu"),
                "g": res.get("g"),
                "c": res.get("c"),
                "t": res.get("t"),
                "d": res.get("d"),
                "p": res.get("p"),
                "r": res.get("r"),
                "g_base": res.get("g_base"),
                "g_transitorio": res.get("g_transitorio"),
                "aj": res.get("aj"),
                "alpha": res.get("alpha"),
                "qc": res.get("qc"),
                "qagd": res.get("qagd"),
                "dcr_kwh": res.get("dcr_kwh"),
                "mc_used": res.get("mc_used"),
                "pb_used": res.get("pb_used"),
                "fetched_at": fetched_at,
                "data": res,
            })
    return rows


def save_rows(rows: list[dict]) -> None:
    if not rows:
        return
    # Delete and insert one run_id at a time to stay within Supabase statement timeout
    by_run: dict[int, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run_id"], []).append(row)
    for run_id, run_rows in by_run.items():
        supabase.table("simulation_results").delete().eq("run_id", run_id).execute()
        supabase.table("simulation_results").insert(run_rows).execute()


async def run_ingestion() -> None:
    print("Obteniendo run_ids desde Supabase...")
    run_ids = get_run_ids()
    print(f"Total run_ids: {len(run_ids)}")

    if not run_ids:
        print("No hay run_ids para procesar.")
        return

    batches = [run_ids[i : i + BATCH_SIZE] for i in range(0, len(run_ids), BATCH_SIZE)]
    total_rows = 0

    async with httpx.AsyncClient() as client:
        for idx, batch in enumerate(batches, 1):
            print(f"Procesando batch {idx}/{len(batches)} ({len(batch)} runs)...")
            fetched_at = datetime.now(timezone.utc).isoformat()
            try:
                runs = await fetch_batch(client, batch)
                rows = []
                for run in runs:
                    rows.extend(flatten_run(run, fetched_at))
                save_rows(rows)
                total_rows += len(rows)
                print(f"  Guardadas {len(rows)} filas de {len(runs)} runs.")
            except httpx.HTTPStatusError as e:
                print(f"  Error HTTP: {e.response.status_code} — {e.response.text[:200]}")
            except Exception as e:
                print(f"  Error: {e}")

    print(f"\nIngesta completa. Total filas guardadas: {total_rows}")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
