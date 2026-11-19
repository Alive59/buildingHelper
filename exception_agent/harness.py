"""The loop. Harness code drives it; the LLM contributes one action per step."""
import json
import time
from collections import Counter
from pathlib import Path

from .actions import ACTION_SCHEMA, RejectedOutput, apply_fix, parse_action
from .builder import BuildError, build
from .config import Budget, Limits
from .gate import Gate, failure_reasons
from .llm import LLMClient, LLMError
from .mesh import Mesh
from .prompt import system_prompt, user_prompt

REQUIRED_KEYS = ("id", "footprint", "ground_z")


class Harness:
    def __init__(self, llm: LLMClient, gate: Gate, out_dir: Path,
                 budget: Budget = Budget(), limits: Limits = Limits()):
        self.llm, self.gate, self.out_dir = llm, gate, Path(out_dir)
        self.budget, self.limits = budget, limits
        self.system = system_prompt(gate.thresholds)

    # --- queue ---------------------------------------------------------------

    def run_queue(self, queue_dir: Path) -> dict:
        outcomes = []
        for path in sorted(Path(queue_dir).glob("*.json")):
            try:
                building = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                building = {"id": path.stem, "_load_error": str(e)}
            outcomes.append(self.run_building(building))
        summary = summarize(outcomes)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "results.jsonl").write_text(
            "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in outcomes))
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # --- one building --------------------------------------------------------

    def run_building(self, building: dict) -> dict:
        bid = str(building.get("id", "unknown"))
        bdir = self.out_dir / bid
        bdir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        prior, failure_types = self._prior_attempts(building)

        def finish(outcome: str, steps: int, **extra) -> dict:
            result = {"building_id": bid, "outcome": outcome, "provenance": "agent",
                      "steps": steps, "seconds": round(time.monotonic() - start, 2),
                      "prior_failure_types": failure_types, **extra}
            if outcome == "accepted":
                # Hard case for BldgXL retraining: kept separate, usable only after review.
                result["retrain_candidate"] = {"reviewed": False}
            if outcome == "human_review":
                result["human_agrees"] = None  # filled in by the reviewer (give-up accuracy)
            (bdir / "outcome.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
            return result

        problem = self._input_problem(building)
        if problem:
            return finish("unrecoverable", 0, summary=problem)

        digest: list[str] = []
        last_spec = None
        deadline = start + self.budget.max_minutes * 60
        step = 0
        stop = "step budget exhausted"
        while step < self.budget.max_steps:
            if time.monotonic() >= deadline:
                stop = "time budget exhausted"
                break
            step += 1
            t0 = time.monotonic()
            log = {"step": step, "action": None, "error": None, "mesh_path": None, "gate": None}
            messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user_prompt(
                    building, prior, self._history(digest), self.budget.max_steps - step + 1,
                    [p["name"] for p in last_spec["parts"]] if last_spec else None)},
            ]
            log["prompt"] = messages
            try:
                raw = self.llm.complete(messages, ACTION_SCHEMA, bid, deadline - time.monotonic())
            except LLMError as e:
                log["error"] = f"LLM endpoint failed: {e}"
                self._log(bdir, log, t0)
                return finish("unrecoverable", step, summary=log["error"])
            log["raw_output"] = raw

            label = "invalid output"
            try:
                action = parse_action(raw)
                log["action"] = action
                label = action["action"]
                if label == "give_up":
                    self._log(bdir, log, t0)
                    return finish("human_review", step, stopped_by="give_up",
                                  diagnosis=action["diagnosis"])
                spec = (action["geometry"] if label == "rebuild_full"
                        else apply_fix(last_spec, action))
                mesh = build(spec, building["footprint"], building["ground_z"], self.limits)
            except (RejectedOutput, BuildError) as e:
                log["error"] = str(e)
                digest.append(f"- step {step}: {label} -> REJECTED: {e}")
                self._log(bdir, log, t0)
                continue

            last_spec = spec
            mesh_path = bdir / f"step_{step:02d}.obj"
            mesh.write_obj(mesh_path)
            report = self.gate.evaluate(mesh, building)
            log["mesh_path"], log["gate"] = str(mesh_path), report.to_dict()
            self._log(bdir, log, t0)
            if report.passed:
                return finish("accepted", step, mesh_path=str(mesh_path))
            what = label + (f"({action['target']})" if label == "fix_part" else "")
            digest.append(f"- step {step}: {what} [{_brief(spec)}] -> FAIL: "
                          + "; ".join(failure_reasons(report)))

        summary = (f"{stop} after {step} steps. Initial failures: "
                   f"{', '.join(failure_types) or 'not recorded'}. Agent steps:\n"
                   + ("\n".join(digest) or "- none"))
        return finish("human_review", step, stopped_by="budget", summary=summary)

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _input_problem(building: dict) -> str | None:
        if "_load_error" in building:
            return f"queue entry unreadable: {building['_load_error']}"
        missing = [k for k in REQUIRED_KEYS if k not in building]
        if missing:
            return f"queue entry missing {missing}"
        if building.get("status") == "passed":
            return "building already passed the gate; the agent must not touch it"
        if len(building["footprint"]) < 3:
            return "source footprint has fewer than 3 vertices"
        return None

    def _prior_attempts(self, building: dict) -> tuple[list[str], list[str]]:
        """LLM-readable lines for the pipeline's earlier attempts, and their failed checks."""
        lines, types = [], Counter()
        for att in building.get("attempts") or []:
            reasons = att.get("reasons")
            if reasons is None and att.get("mesh_path") and Path(att["mesh_path"]).exists():
                report = self.gate.evaluate(Mesh.read_obj(att["mesh_path"]), building)
                reasons = failure_reasons(report)
                types.update(c.name for c in report.failed)
            types.update(att.get("failed_checks") or [])
            lines.append(f"- {att.get('route', '?')}: " + ("; ".join(reasons) if reasons
                         else att.get("note") or "failed, no gate report"))
        return lines, sorted(types)

    def _history(self, digest: list[str]) -> str:
        """Running digest, newest kept; oldest dropped to stay within the context budget."""
        kept, used = [], 0
        for line in reversed(digest):
            if kept and used + len(line) > self.limits.history_chars:
                break
            kept.append(line[: self.limits.history_chars])
            used += len(line)
        dropped = len(digest) - len(kept)
        head = [f"- ({dropped} earlier steps omitted)"] if dropped else []
        return "\n".join(head + kept[::-1])

    @staticmethod
    def _log(bdir: Path, log: dict, t0: float) -> None:
        log["seconds"] = round(time.monotonic() - t0, 3)
        with open(bdir / "steps.jsonl", "a") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")


def _brief(spec: dict) -> str:
    return ", ".join(f"{p['name']}: {p['roof']['type']} eave {p['eave_height']:g} "
                     f"rise {p['roof']['rise']:g}" for p in spec["parts"])


def summarize(outcomes: list[dict]) -> dict:
    n = len(outcomes)
    accepted = [o for o in outcomes if o["outcome"] == "accepted"]
    by_type: dict[str, dict] = {}
    for o in outcomes:
        for t in o["prior_failure_types"] or ["unknown"]:
            row = by_type.setdefault(t, {"buildings": 0, "accepted": 0})
            row["buildings"] += 1
            row["accepted"] += o["outcome"] == "accepted"
    return {
        "buildings": n,
        "outcomes": dict(Counter(o["outcome"] for o in outcomes)),
        "rescue_rate": round(len(accepted) / n, 3) if n else None,
        "mean_steps": round(sum(o["steps"] for o in outcomes) / n, 2) if n else None,
        "mean_seconds": round(sum(o["seconds"] for o in outcomes) / n, 2) if n else None,
        "rescue_by_prior_failure_type": by_type,
    }
