"""The LLM's action: schema, validation and the permission check.

Permissions are structural. The schema admits three actions and no extra keys,
so there is no field through which the LLM could name another building, a
threshold or a file. Anything that fails here is rejected and counts as a step.
"""
import json

import jsonschema

from .builder import GEOMETRY_SCHEMA

ACTIONS = ("rebuild_full", "fix_part", "give_up")

ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "reason", "target", "geometry", "diagnosis"],
    "properties": {
        "action": {"enum": list(ACTIONS)},
        "reason": {"type": "string", "description": "why this action, citing the gate report"},
        "target": {"type": ["string", "null"], "description": "part name, for fix_part"},
        "geometry": {"anyOf": [GEOMETRY_SCHEMA, {"type": "null"}]},
        "diagnosis": {"type": ["string", "null"], "description": "required for give_up"},
    },
}

_validator = jsonschema.Draft202012Validator(ACTION_SCHEMA)


class RejectedOutput(ValueError):
    """Raised with a message that is safe and useful to show the LLM."""


def parse_action(raw: str) -> dict:
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RejectedOutput(f"output is not valid JSON: {e.msg} at char {e.pos}") from None
    errors = sorted(_validator.iter_errors(action), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(map(str, e.path)) or "(root)"
        raise RejectedOutput(f"schema violation at {where}: {e.message[:200]}")

    kind = action["action"]
    if kind == "give_up":
        if not (action["diagnosis"] or "").strip():
            raise RejectedOutput("give_up requires a non-empty diagnosis")
    else:
        if action["geometry"] is None:
            raise RejectedOutput(f"{kind} requires geometry")
        if kind == "fix_part":
            if not action["target"]:
                raise RejectedOutput("fix_part requires target (the part name)")
            if len(action["geometry"]["parts"]) != 1:
                raise RejectedOutput("fix_part geometry must contain exactly one part: "
                                     "the replacement for the target")
    return action


def apply_fix(last_spec: dict | None, action: dict) -> dict:
    """fix_part: replace the target part in the last agent spec, keep the rest."""
    if last_spec is None:
        raise RejectedOutput("fix_part needs an earlier agent build to edit; use rebuild_full first")
    target = action["target"]
    names = [p["name"] for p in last_spec["parts"]]
    if target not in names:
        raise RejectedOutput(f"fix_part target '{target}' not found; parts are {names}")
    replacement = dict(action["geometry"]["parts"][0], name=target)
    return {"parts": [replacement if p["name"] == target else p for p in last_spec["parts"]]}
