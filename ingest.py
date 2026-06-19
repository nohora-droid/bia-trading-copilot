"""
Ingesta de resultados de simulación desde la API de Olibia Pricing hacia Supabase.
Uso: python ingest.py
"""

import os
import asyncio
import httpx
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OLIBIA_API_KEY = "bia_06257853e300d9ad3e81e1f8124e4041d9712bcbb6a00d04679ac54db56e7a8c746061a817db8ecbde622f5a6cc071c58f714d37741e06e7c83406b095350f00"
OLIBIA_BASE_URL = "https://integrations.bia.app/ms-olibia-pricing/v1/simulations"
BATCH_SIZE = 50

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_run_ids() -> list[str]:
    result = supabase.table("simulation_runs").select("id").execute()
    return [row["id"] for row in (result.data or [])]


async def fetch_batch(client: httpx.AsyncClient, run_ids: list[str]) -> list[dict]:
    ids_param = ",".join(str(i) for i in run_ids)
    response = await client.get(
        f"{OLIBIA_BASE_URL}/{ids_param}",
        headers={"api-key": OLIBIA_API_KEY},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data
    return data.get("results", data.get("data", []))


def upsert_results(results: list[dict]) -> None:
    if not results:
        return
    supabase.table("simulation_results").upsert(results, on_conflict="id").execute()


async def run_ingestion() -> None:
    print("Obteniendo run_ids desde Supabase...")
    run_ids = get_run_ids()
    print(f"Total run_ids: {len(run_ids)}")

    if not run_ids:
        print("No hay run_ids para procesar.")
        return

    batches = [run_ids[i : i + BATCH_SIZE] for i in range(0, len(run_ids), BATCH_SIZE)]
    total_saved = 0

    async with httpx.AsyncClient() as client:
        for idx, batch in enumerate(batches, 1):
            print(f"Procesando batch {idx}/{len(batches)} ({len(batch)} ids)...")
            try:
                results = await fetch_batch(client, batch)
                upsert_results(results)
                total_saved += len(results)
                print(f"  Guardados {len(results)} resultados.")
            except httpx.HTTPStatusError as e:
                print(f"  Error HTTP en batch {idx}: {e.response.status_code} — {e.response.text[:200]}")
            except Exception as e:
                print(f"  Error en batch {idx}: {e}")

    print(f"Ingesta completa. Total guardados: {total_saved}")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
