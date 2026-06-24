"""
Test script — runs questions directly through build_context() + Claude API
without going through Slack.
Usage: python test_copilot.py
Output: printed to console and saved to test_results.txt
"""
import os, sys, time, io
from dotenv import load_dotenv

# Force UTF-8 output on Windows so emojis don't crash print()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

# Add project dir to path so we can import main
sys.path.insert(0, os.path.dirname(__file__))
import main as copilot

QUESTIONS = [
    "¿qué meses debería priorizar para contratar energía y por qué?",
    "¿cómo está el spread de BIA vs VATIA en Bogotá para los próximos meses?",
    "¿qué cambió en el CU de BIA entre la corrida de Allison y la que corrió Juliana?",
    "¿cómo está BIA frente a la regla de tándem?",
    "¿cuál es el CU de BIA en Bogotá para agosto 2026 escenario MEDIUM?",
    "¿cuál es el principal riesgo para los próximos 6 meses?",
    "¿conviene contratar ahora o esperar?",
    "¿qué competidor tiene la tarifa más baja para el próximo trimestre?",
    # P9–P18
    "¿cómo está el spread de BIA vs ENEL en Bogotá?",
    "¿cómo está el spread de BIA vs EPM en Antioquia?",
    "¿cómo está el spread de BIA vs EMCALI en Cali?",
    "¿cómo está el spread de BIA vs Air-e en Caribe Sol?",
    "¿cómo está el spread de BIA vs Afinia en Caribe Mar?",
    "¿en qué meses se prevén las mayores variaciones de tarifa para BIA?",
    "¿qué OR tiene la tarifa más baja para el segundo semestre 2026?",
    "¿en qué mercados BIA es más competitivo frente a los OR?",
    "¿cuál es la tendencia del componente G para los próximos meses?",
    "¿qué tan sensible es el CU de BIA a un incremento del 20% en el precio de bolsa?",
]

def run_question(q: str) -> tuple[str, str, float]:
    t0 = time.time()
    context = copilot.build_context(q)
    claude = copilot.get_claude()
    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=copilot._SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Datos de contexto:\n{context}\n\nPregunta: {q}"}],
    )
    answer = msg.content[0].text if msg.content else "(sin respuesta)"
    elapsed = time.time() - t0
    return context, answer, elapsed


def main():
    lines = []
    sep = "=" * 80

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{sep}")
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        print(sep)

        try:
            context, answer, elapsed = run_question(q)
            ctx_preview = context[:400].replace("\n", " ") + ("…" if len(context) > 400 else "")
            print(f"[contexto ({len(context)} chars)]: {ctx_preview}")
            print(f"\n{answer}")
            print(f"\n⏱  {elapsed:.1f}s")

            block = (
                f"{sep}\n"
                f"PREGUNTA {i}: {q}\n"
                f"{sep}\n"
                f"CONTEXTO ({len(context)} chars):\n{context[:800]}{'…' if len(context)>800 else ''}\n\n"
                f"RESPUESTA:\n{answer}\n\n"
                f"Tiempo: {elapsed:.1f}s\n"
            )
        except Exception as e:
            msg = f"ERROR: {e}"
            print(msg)
            block = f"{sep}\nPREGUNTA {i}: {q}\n{sep}\n{msg}\n"

        lines.append(block)

    output_path = os.path.join(os.path.dirname(__file__), "test_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
