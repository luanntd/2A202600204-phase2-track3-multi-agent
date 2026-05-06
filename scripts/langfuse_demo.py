"""Generate Langfuse trace traffic for demo/screenshot purposes."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Validate Langfuse keys present
for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
    if not os.getenv(key):
        print(f"ERROR: {key} not set in .env")
        sys.exit(1)

from langfuse.openai import openai  # noqa: E402  — must import after env is loaded

QUERIES = [
    ("What is LangGraph and how does it differ from LangChain?", "langfuse-demo"),
    ("Explain multi-agent AI systems in 3 sentences.", "langfuse-demo"),
    ("What are the main benefits of LLM observability tools?", "langfuse-demo"),
]

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def run_traces() -> None:
    print(f"Sending {len(QUERIES)} traces to Langfuse ({os.getenv('LANGFUSE_BASE_URL')})...\n")

    for i, (prompt, session) in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {prompt[:60]}...")
        response = openai.chat.completions.create(
            name=f"demo-query-{i}",
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Be concise."},
                {"role": "user", "content": prompt},
            ],
            metadata={"session_id": session, "query_index": i},
        )
        answer = response.choices[0].message.content or ""
        print(f"    -> {answer[:80]}...\n")

    # Ensure all events are flushed before exit
    from langfuse import get_client
    get_client().flush()
    print("Done. Open cloud.langfuse.com > Traces to see the traffic.")


if __name__ == "__main__":
    run_traces()
