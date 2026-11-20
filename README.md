# buildingHelper — exception-agent harness (prototype)

Harness for buildings that failed every tier of the bulk pipeline
(BldgXL → rule-based LoD2 → LoD1). The harness code runs the loop; a local LLM
chooses one action per step. Covers milestones 1–2 of the v0.1 spec, plus the
client for milestone 3.

## Run

```bash
python -m exception_agent --queue examples/queue --stub-script examples/stub_script.json --out runs/demo
```

```bash
python -m exception_agent --queue examples/queue --out runs/llm --llm openai --base-url http://localhost:8000/v1 --model <model>
```

```bash
python -m pytest -q
```

Requires Python 3.11+, `jsonschema` and `numpy`. Use `--json-mode object|none` if the
server does not support JSON-schema output; the harness validates either way.

## Real cases

```bash
python -m exception_agent.import_cases <cases_dir> data/queue12
```

Converts gen-model case folders (`meta.json`, `footprint.geojson`, `crop_model.png`,
`ndsm_pred.npy`, `ship_gen.obj`, `gt_lod2.obj`) into queue entries: footprint in a
local metric frame, the roof crop and predicted height map as images, the height
map as text hints, and the failed generation as the prior attempt, which the
harness re-gates. GT feeds only the placeholder scorer. `data/` is git-ignored.
Needs `numpy`.

## Layout

| file | role |
|---|---|
| `harness.py` | loop, budget, outcomes, per-step logging, run summary |
| `actions.py` | action schema, validation, permission check, `fix_part` |
| `builder.py` | parametric spec → closed mesh (flat / shed / gable / hip) |
| `gate.py` | gate interface (topology + BRICE), stand-in gate, failure-reason formatter |
| `diagnostics.py` | placeholder scorer; optional DSM and footprint hints (not gate conditions) |
| `prompt.py`, `guidance.md` | per-step prompt; edit `guidance.md` for the guidance |
| `llm.py` | stub LLM and OpenAI-compatible client (text + images) |
| `render.py` | top-down render of each build, sent back to the LLM |
| `import_cases.py` | case folders → queue entries |
| `config.py` | thresholds, budget, limits (frozen) |

## Input and output

Queue entry (`examples/queue/*.json`): `id`, `footprint` ([[x, y], …] in metres),
`ground_z`, `height_m` and `height_source` (prediction or LoD1),
`refs.roof_image` (path, relative to the queue directory), optional `dsm_samples`
([[x, y, z], …], only where a 3D reference exists),
`attempts` (per route: `reasons`, `failed_checks`, or a `mesh_path` to re-gate).

Run directory: `<id>/steps.jsonl` (prompt, raw output, parsed action, mesh path,
gate report, hints, seconds), `<id>/step_NN.obj` and `.png`, `<id>/outcome.json`, `results.jsonl`,
`summary.json` (rescue rate, steps, rescue by prior failure type).

Outcomes: `accepted`; `human_review` (`stopped_by` = `give_up` or `budget`);
`unrecoverable` (bad queue entry, building already passed, or LLM endpoint down).

## Placeholders to replace

- **Gate:** topology (watertight, manifold, 256 faces) plus one similarity score.
  The topology checks are real. The score is **not BRICE**: it compares the mesh
  with a hidden `_stub_reference` height field in the demo entries. Pass the real scorer as `StandInGate(similarity=...)`, or wrap the whole
  pipeline gate in a class with `evaluate(mesh, building) -> GateReport`.
- **Threshold:** `brice_min` in `config.py` is a guess.
- **Guidance:** `guidance.md` has TBD sections for the guideline criteria.
- **Build format:** option (a), parametric parts. Parts are separate closed
  shells with no boolean union; gable and hip need a 4-vertex footprint.
- **Budget:** N = 5 steps, T = 10 minutes.
