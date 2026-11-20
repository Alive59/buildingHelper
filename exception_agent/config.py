"""Fixed settings. Frozen so nothing downstream of the LLM can change them."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GateThresholds:
    max_faces: int = 256
    brice_min: float = 0.5  # placeholder: replace with the pipeline's BRICE threshold


@dataclass(frozen=True)
class Budget:
    max_steps: int = 5
    max_minutes: float = 10.0


@dataclass(frozen=True)
class Limits:
    history_chars: int = 1500  # cap on the history digest given to the LLM
    max_parts: int = 8
    max_height_m: float = 500.0
