import argparse
import json
from pathlib import Path

from .config import Budget
from .gate import StandInGate
from .harness import Harness
from .llm import OpenAICompatLLM, StubLLM


def main() -> None:
    p = argparse.ArgumentParser(prog="exception_agent", description="Exception-agent harness prototype")
    p.add_argument("--queue", type=Path, required=True, help="directory of building JSON files")
    p.add_argument("--out", type=Path, required=True, help="run directory for meshes and logs")
    p.add_argument("--llm", choices=["stub", "openai"], default="stub")
    p.add_argument("--stub-script", type=Path, help="JSON: {building_id: [actions]}")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="local")
    p.add_argument("--api-key", default="none")
    p.add_argument("--json-mode", choices=["schema", "object", "none"], default="schema")
    p.add_argument("--no-vision", action="store_true", help="do not send images to the LLM")
    p.add_argument("--extra-body", type=json.loads, default={},
                   help="JSON merged into each request, e.g. a server switch for thinking mode")
    p.add_argument("--steps", type=int, default=Budget.max_steps)
    p.add_argument("--minutes", type=float, default=Budget.max_minutes)
    a = p.parse_args()

    if a.llm == "stub":
        llm = StubLLM(json.loads(a.stub_script.read_text()) if a.stub_script else {})
    else:
        llm = OpenAICompatLLM(a.base_url, a.model, a.api_key, a.json_mode,
                              vision=not a.no_vision, extra_body=a.extra_body)
    harness = Harness(llm, StandInGate(), a.out, Budget(a.steps, a.minutes))
    print(json.dumps(harness.run_queue(a.queue), indent=2))


if __name__ == "__main__":
    main()
