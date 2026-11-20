import json
from pathlib import Path

import pytest

from exception_agent.actions import RejectedOutput, parse_action
from exception_agent.builder import BuildError, build
from exception_agent.config import Budget
from exception_agent.diagnostics import hints
from exception_agent.gate import Check, GateReport, StandInGate, failure_reasons
from exception_agent.harness import Harness
from exception_agent.llm import StubLLM
from exception_agent.mesh import Mesh

ROOT = Path(__file__).resolve().parent.parent
RECT = [[0, 0], [12, 0], [12, 8], [0, 8]]
L = [[0, 0], [10, 0], [10, 6], [4, 6], [4, 12], [0, 12]]


def part(kind, footprint=None, rise=2.0, eave=5.0):
    return {"name": "main", "footprint": footprint, "eave_height": eave,
            "roof": {"type": kind, "rise": 0.0 if kind == "flat" else rise,
                     "ridge_axis": None, "upslope_azimuth_deg": 30.0 if kind == "shed" else None}}


def action(**kw):
    return json.dumps({"action": "rebuild_full", "reason": "r", "target": None,
                       "geometry": {"parts": [part("flat")]}, "diagnosis": None, **kw})


@pytest.mark.parametrize("kind,fp", [("flat", RECT), ("shed", RECT), ("gable", RECT), ("hip", RECT),
                                     ("flat", L), ("shed", L), ("flat", RECT[::-1])])
def test_builder_makes_closed_manifold_solids(kind, fp):
    mesh = build({"parts": [part(kind)]}, fp, 0.0)
    checks = {c.name: c for c in StandInGate().evaluate(mesh, {"footprint": fp, "ground_z": 0.0}).checks}
    assert checks["watertight"].passed and checks["manifold"].passed
    assert not hints(mesh, {"footprint": fp, "ground_z": 0.0})


def test_gable_ridge_follows_axis():
    spec = {"parts": [part("gable")]}
    assert build(spec, RECT, 0.0).top_z_at(6, 4) == pytest.approx(7.0)  # ridge along x (long side)
    spec["parts"][0]["roof"]["ridge_axis"] = "short"
    m = build(spec, RECT, 0.0)
    assert m.top_z_at(6, 1) == pytest.approx(7.0) and m.top_z_at(1, 4) < 6


def test_builder_rejects_gable_on_non_quad():
    with pytest.raises(BuildError, match="4-vertex"):
        build({"parts": [part("gable")]}, L, 0.0)


def test_gate_reports_open_and_oversized_meshes():
    mesh = build({"parts": [part("flat")]}, RECT, 0.0)
    mesh.faces.pop()
    report = StandInGate().evaluate(mesh, {"footprint": RECT, "ground_z": 0.0})
    assert not report.passed
    assert failure_reasons(report)[0].startswith("not watertight: 3 open boundary edges")


def test_formatter_wording():
    report = GateReport(False, [
        Check("faces", False, 312, 256),
        Check("brice", False, 0.41, 0.5), Check("brice", False, None, 0.5),
        Check("manifold", False, 1, 0, {"non_manifold_edges": 1, "faces": [12, 13, 14]}),
        Check("new_check", False, 0.4, 0.6)])
    assert failure_reasons(report) == [
        "312 faces > 256",
        "BRICE 0.41 < 0.50 (similarity too low)", "BRICE could not be evaluated",
        "not manifold: 1 non-manifold edges (e.g. faces 12, 13, 14)",
        "new_check failed: 0.4 vs threshold 0.6"]


@pytest.mark.parametrize("raw,msg", [
    ("not json", "not valid JSON"),
    (action(action="edit_thresholds"), "schema violation at action"),
    (action(max_faces=9999), "schema violation"),
    (action(geometry=None), "requires geometry"),
    (action(action="give_up", geometry=None), "diagnosis"),
    (action(action="fix_part"), "requires target"),
])
def test_bad_outputs_are_rejected(raw, msg):
    with pytest.raises(RejectedOutput, match=msg):
        parse_action(raw)


def test_demo_queue_end_to_end(tmp_path):
    script = json.loads((ROOT / "examples/stub_script.json").read_text())
    summary = Harness(StubLLM(script), StandInGate(), tmp_path).run_queue(ROOT / "examples/queue")
    out = {p.parent.name: json.loads(p.read_text()) for p in tmp_path.glob("*/outcome.json")}
    assert (out["B001"]["outcome"], out["B001"]["steps"]) == ("accepted", 2)
    assert (out["B002"]["outcome"], out["B002"]["stopped_by"], out["B002"]["steps"]) == ("human_review", "budget", 5)
    assert (out["B003"]["outcome"], out["B003"]["stopped_by"]) == ("human_review", "give_up")
    assert summary["rescue_rate"] == pytest.approx(1 / 3, abs=1e-3)
    steps = [json.loads(line) for line in (tmp_path / "B002/steps.jsonl").read_text().splitlines()]
    assert len(steps) == 5 and steps[0]["error"] and all("prompt" in s and "raw_output" in s for s in steps)
    assert Mesh.read_obj(out["B001"]["mesh_path"]).faces


def test_fix_part_without_build_and_passed_building(tmp_path):
    fix = json.loads(action(action="fix_part", target="main"))
    h = Harness(StubLLM({"X": [fix]}), StandInGate(), tmp_path, Budget(max_steps=1))
    b = {"id": "X", "footprint": RECT, "ground_z": 0.0}
    assert "use rebuild_full first" in h.run_building(b)["summary"]
    assert h.run_building({**b, "status": "passed"})["outcome"] == "unrecoverable"


def test_history_digest_is_capped(tmp_path):
    h = Harness(StubLLM({}), StandInGate(), tmp_path)
    text = h._history([f"- step {i}: " + "x" * 400 for i in range(1, 9)])
    assert len(text) < 1700 and "earlier steps omitted" in text and "step 8" in text


def test_gate_is_topology_plus_similarity_and_hints_give_direction():
    truth = build({"parts": [part("flat", eave=9.0)]}, RECT, 0.0)
    b = {"id": "X", "footprint": RECT, "ground_z": 0.0,
         "dsm_samples": [[x + 0.5, y + 0.5, 9.0] for x in range(12) for y in range(8)]}
    low = build({"parts": [part("flat", eave=5.0)]}, RECT, 0.0)
    report = StandInGate().evaluate(low, b)
    assert [c.name for c in report.checks] == ["faces", "watertight", "manifold", "brice"]
    assert failure_reasons(report) == ["BRICE 0.20 < 0.50 (similarity too low)"]
    assert hints(low, b)[0].startswith("roof is 4.0 m lower than DSM on average")
    assert StandInGate().evaluate(truth, b).passed
    assert not StandInGate().evaluate(truth, {**b, "dsm_samples": []}).passed  # no score, no accept
    assert StandInGate(similarity=lambda m, bld: 0.9).evaluate(low, b).passed  # pluggable scorer
