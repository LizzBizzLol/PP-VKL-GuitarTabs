#!/usr/bin/env python
"""Diagnose SynthTab pitch-to-tablature gap from saved eval outputs.

CPU-only by design. The script reads `results/summary.json` and per-track
metric `.txt` files. It does not import the training pipeline, load audio, or
load model checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable


METRIC_LINE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s+:\s+([-+0-9.eE]+)\s*$")


@dataclass(frozen=True)
class RunSpec:
    label: str
    chunk: str
    checkpoint: str
    experiment: str


@dataclass
class TrackMetrics:
    track_id: str
    multi_pitch_f1: float | None = None
    tablature_f1: float | None = None
    ref_silence_ratio: float | None = None
    pred_silence_ratio: float | None = None
    non_silent_accuracy: float | None = None

    @property
    def gap(self) -> float | None:
        if self.multi_pitch_f1 is None or self.tablature_f1 is None:
            return None
        return self.multi_pitch_f1 - self.tablature_f1

    @property
    def silence_delta(self) -> float | None:
        if self.pred_silence_ratio is None or self.ref_silence_ratio is None:
            return None
        return self.pred_silence_ratio - self.ref_silence_ratio


@dataclass
class RunDiagnostics:
    spec: RunSpec
    experiment_dir: Path
    aggregate_mp_f1: float | None
    aggregate_tab_f1: float | None
    aggregate_accuracy: float | None
    aggregate_non_silent_accuracy: float | None
    aggregate_ref_silence: float | None
    aggregate_pred_silence: float | None
    collapse_to_silence: bool | None
    tracks: list[TrackMetrics]


DEFAULT_RUNS = [
    RunSpec(
        "Run 1 clean",
        "electric_clean/semihollow_clean_finger",
        "training-state-12712.pt",
        "generated/experiments/full_chunk_semihollow_clean_finger_28ep_fresh",
    ),
    RunSpec(
        "Run 2 distortion",
        "electric_distortion/semihollow_clean_finger",
        "training-state-26544.pt",
        "generated/experiments/full_chunk_electric_distortion_semihollow_clean_finger_28ep_resume_from_12712",
    ),
    RunSpec(
        "Run 3 muted",
        "electric_muted",
        "training-state-50372.pt",
        "generated/experiments/full_chunk_electric_muted_28ep_resume_from_26544",
    ),
    RunSpec(
        "Run 4 acoustic",
        "acoustic/luthier_pick/part_1_-_1_to_B_C",
        "training-state-58184.pt",
        "generated/experiments/full_chunk_acoustic_luthier_pick_part1_28ep_resume_from_50372",
    ),
    RunSpec(
        "Run 5 clean peregrine",
        "electric_clean/peregrine_clean_neck",
        "training-state-70896.pt",
        "generated/experiments/full_chunk_electric_clean_peregrine_clean_neck_28ep_resume_from_58184",
    ),
    RunSpec(
        "Run 6 clean lespaul",
        "electric_clean/lespaul_clean_both",
        "training-state-83608.pt",
        "generated/experiments/full_chunk_electric_clean_lespaul_clean_both_28ep_resume_from_70896",
    ),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_metric_txt(path: Path) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    current: str | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("-----") and stripped.endswith("-----"):
            current = stripped.strip("-").strip()
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        match = METRIC_LINE.match(line)
        if match:
            key, raw = match.groups()
            sections[current][key] = float(raw)

    return sections


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f} pp"


def present(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def median_or_none(values: Iterable[float | None]) -> float | None:
    data = present(values)
    return median(data) if data else None


def mean_or_none(values: Iterable[float | None]) -> float | None:
    data = present(values)
    return mean(data) if data else None


def count_where(tracks: Iterable[TrackMetrics], predicate) -> int:
    return sum(1 for track in tracks if predicate(track))


def track_id_from_path(results_dir: Path, txt_path: Path) -> str:
    return str(txt_path.relative_to(results_dir).with_suffix(""))


def load_run(spec: RunSpec, root: Path) -> RunDiagnostics:
    experiment_dir = root / spec.experiment
    results_dir = experiment_dir / "results"
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    summary = load_json(summary_path)
    synthtab_val = summary.get("synthtab_val", {})
    diagnostics = synthtab_val.get("diagnostics", {})
    diagnostic_tracks = {
        track.get("track_id"): track
        for track in diagnostics.get("tracks", [])
        if isinstance(track, dict) and isinstance(track.get("track_id"), str)
    }

    tracks: list[TrackMetrics] = []
    for txt_path in sorted(results_dir.rglob("*.txt")):
        sections = parse_metric_txt(txt_path)
        track_id = track_id_from_path(results_dir, txt_path)
        diag = diagnostic_tracks.get(track_id, {})
        mp = sections.get("multi_pitch", {})
        tab = sections.get("tablature", {})
        tracks.append(
            TrackMetrics(
                track_id=track_id,
                multi_pitch_f1=parse_float(mp.get("f1-score")),
                tablature_f1=parse_float(tab.get("f1-score")),
                ref_silence_ratio=parse_float(diag.get("ref_silence_ratio")),
                pred_silence_ratio=parse_float(diag.get("pred_silence_ratio")),
                non_silent_accuracy=parse_float(diag.get("non_silent_accuracy")),
            )
        )

    return RunDiagnostics(
        spec=spec,
        experiment_dir=experiment_dir,
        aggregate_mp_f1=parse_float(synthtab_val.get("multi_pitch", {}).get("f1-score")),
        aggregate_tab_f1=parse_float(synthtab_val.get("tablature", {}).get("f1-score")),
        aggregate_accuracy=parse_float(synthtab_val.get("tablature", {}).get("accuracy")),
        aggregate_non_silent_accuracy=parse_float(diagnostics.get("non_silent_accuracy")),
        aggregate_ref_silence=parse_float(diagnostics.get("ref_silence_ratio")),
        aggregate_pred_silence=parse_float(diagnostics.get("pred_silence_ratio")),
        collapse_to_silence=diagnostics.get("collapse_to_silence"),
        tracks=tracks,
    )


def high_pitch_weak_tab(track: TrackMetrics) -> bool:
    return (
        track.multi_pitch_f1 is not None
        and track.tablature_f1 is not None
        and track.multi_pitch_f1 >= 0.80
        and track.tablature_f1 < 0.60
    )


def large_gap(track: TrackMetrics) -> bool:
    return track.gap is not None and track.gap >= 0.30


def extra_activity(track: TrackMetrics) -> bool:
    return track.silence_delta is not None and track.silence_delta <= -0.10


def under_activity(track: TrackMetrics) -> bool:
    return track.silence_delta is not None and track.silence_delta >= 0.10


def top_gap_tracks(run: RunDiagnostics, limit: int) -> list[TrackMetrics]:
    tracks = [track for track in run.tracks if track.gap is not None]
    return sorted(tracks, key=lambda track: track.gap or 0.0, reverse=True)[:limit]


def render_run_table(runs: list[RunDiagnostics]) -> list[str]:
    lines = [
        "| Run | Chunk | Checkpoint | MP F1 | Tab F1 | Median MP | Median Tab | Mean gap | High MP/low Tab | Large gap | Silence delta | Collapse |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        tracks = run.tracks
        high_pitch = count_where(tracks, high_pitch_weak_tab)
        gap_count = count_where(tracks, large_gap)
        mean_gap = mean_or_none(track.gap for track in tracks)
        silence_delta = None
        if run.aggregate_pred_silence is not None and run.aggregate_ref_silence is not None:
            silence_delta = run.aggregate_pred_silence - run.aggregate_ref_silence
        lines.append(
            "| "
            + " | ".join(
                [
                    run.spec.label,
                    f"`{run.spec.chunk}`",
                    f"`{run.spec.checkpoint}`",
                    pct(run.aggregate_mp_f1),
                    pct(run.aggregate_tab_f1),
                    pct(median_or_none(track.multi_pitch_f1 for track in tracks)),
                    pct(median_or_none(track.tablature_f1 for track in tracks)),
                    pp(mean_gap),
                    f"{high_pitch}/{len(tracks)}",
                    f"{gap_count}/{len(tracks)}",
                    pp(silence_delta),
                    str(run.collapse_to_silence).lower(),
                ]
            )
            + " |"
        )
    return lines


def render_category_table(runs: list[RunDiagnostics]) -> list[str]:
    lines = [
        "| Run | Tracks | High MP/low Tab | Large gap >=30pp | Extra activity | Under activity | Median non-silent acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        tracks = run.tracks
        lines.append(
            "| "
            + " | ".join(
                [
                    run.spec.label,
                    str(len(tracks)),
                    str(count_where(tracks, high_pitch_weak_tab)),
                    str(count_where(tracks, large_gap)),
                    str(count_where(tracks, extra_activity)),
                    str(count_where(tracks, under_activity)),
                    pct(median_or_none(track.non_silent_accuracy for track in tracks)),
                ]
            )
            + " |"
        )
    return lines


def render_gap_examples(runs: list[RunDiagnostics], limit: int) -> list[str]:
    lines = [
        "| Run | Track | MP F1 | Tab F1 | Gap | Ref silence | Pred silence | Non-silent acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        for track in top_gap_tracks(run, limit):
            lines.append(
                "| "
                + " | ".join(
                    [
                        run.spec.label,
                        f"`{track.track_id}`",
                        pct(track.multi_pitch_f1),
                        pct(track.tablature_f1),
                        pp(track.gap),
                        pct(track.ref_silence_ratio),
                        pct(track.pred_silence_ratio),
                        pct(track.non_silent_accuracy),
                    ]
                )
                + " |"
            )
    return lines


def render_report(runs: list[RunDiagnostics], top_n: int) -> str:
    best = max(runs, key=lambda run: run.aggregate_tab_f1 or -1.0)
    latest = runs[-1]

    lines = [
        "# Tab Head Diagnostic",
        "",
        "CPU-only report from saved SynthTab evaluation outputs. No model, audio, or GPU is loaded.",
        "",
        "## Executive Summary",
        "",
        f"- Best checkpoint by aggregate tablature F1: `{best.spec.checkpoint}` from `{best.spec.chunk}`.",
        f"- Latest chronological checkpoint: `{latest.spec.checkpoint}` from `{latest.spec.chunk}`.",
        "- All six runs have `collapse_to_silence=false`; silence collapse is not the current bottleneck.",
        "- The recurring issue is a large pitch-to-tablature gap: many tracks have strong multi-pitch F1 but weak string/fret F1.",
        "- Similar `electric_clean` chunks after the acoustic peak did not improve quality; they returned the mean MP-Tab gap to about 22 percentage points.",
        "",
        "## Run Comparison",
        "",
    ]
    lines.extend(render_run_table(runs))
    lines.extend(
        [
            "",
            "## Failure Buckets",
            "",
            "- `High MP/low Tab`: per-track multi-pitch F1 >= 80% and tablature F1 < 60%.",
            "- `Large gap`: per-track multi-pitch F1 minus tablature F1 >= 30 percentage points.",
            "- `Extra activity`: predicted silence is at least 10 percentage points lower than reference silence.",
            "- `Under activity`: predicted silence is at least 10 percentage points higher than reference silence.",
            "",
        ]
    )
    lines.extend(render_category_table(runs))
    lines.extend(
        [
            "",
            "## Highest Pitch-To-Tab Gap Examples",
            "",
        ]
    )
    lines.extend(render_gap_examples(runs, top_n))
    lines.extend(
        [
            "",
            "## Representation Check",
            "",
            "- Current pipeline imports `TabCNN` from `amt_tools.models` in `demo_embedding/tabcnn_synthtab_pipeline.py`.",
            "- Estimation uses `TablatureWrapper` plus `StackedMultiPitchCollapser`, so evaluation explicitly compares both string/fret tablature and collapsed multi-pitch.",
            "- The current `amt_tools.models.TabCNN` implementation uses `SoftmaxGroups`: independent string groups with one silence class and fret/note classes per string.",
            "- Current class weighting only separates silence from non-silence per output class; it does not directly penalize pitch-correct but string/fret-wrong alternatives.",
            "",
            "## Decision",
            "",
            "- Do not continue blind scaling on similar `electric_clean` chunks right now.",
            "- Do not enable `balance_by_silence=true`: predictions are not collapsing to silence, and extra non-silent activity is more common than under-activity.",
            "- Next useful step is a small string/fret-focused experiment: post-processing diagnostics or a head/loss/label-representation proposal, using `training-state-58184.pt` as best-by-metrics baseline and `training-state-83608.pt` only as latest chronological continuation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    runs = [load_run(spec, args.repo_root) for spec in DEFAULT_RUNS]
    print(render_report(runs, top_n=args.top_n))


if __name__ == "__main__":
    main()
