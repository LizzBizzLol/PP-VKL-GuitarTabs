#!/usr/bin/env python
"""Frame-level string/fret diagnostics for SynthTab checkpoints.

This is an evaluation-only tool. It loads a trained checkpoint and a small
validation subset, runs inference, and measures how often the model gets the
pitch right but places it on the wrong string/fret.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import amt_tools.tools as tools
from amt_tools.inference import run_offline


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from tabcnn_synthtab_pipeline import (  # noqa: E402
    build_estimators,
    build_feature_extractor,
    build_model,
    build_profile,
    create_synthtab_dataset,
    load_config,
    resolve_device,
    safe_torch_load,
)


METRIC_LINE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s+:\s+([-+0-9.eE]+)\s*$")


@dataclass(frozen=True)
class TrackCandidate:
    track_id: str
    bucket: str
    multi_pitch_f1: float | None
    tablature_f1: float | None
    gap: float | None
    ref_silence_ratio: float | None = None


@dataclass
class Counts:
    ref_notes: int = 0
    pred_notes: int = 0
    exact_tp: int = 0
    pitch_tp: int = 0
    canonical_tp: int = 0
    active_frames_ref: int = 0
    active_frames_pred: int = 0
    frames: int = 0

    def add(self, other: "Counts") -> None:
        self.ref_notes += other.ref_notes
        self.pred_notes += other.pred_notes
        self.exact_tp += other.exact_tp
        self.pitch_tp += other.pitch_tp
        self.canonical_tp += other.canonical_tp
        self.active_frames_ref += other.active_frames_ref
        self.active_frames_pred += other.active_frames_pred
        self.frames += other.frames

    @property
    def pitch_correct_wrong_position(self) -> int:
        return max(0, self.pitch_tp - self.exact_tp)

    @property
    def missed_notes(self) -> int:
        return max(0, self.ref_notes - self.pitch_tp)

    @property
    def extra_notes(self) -> int:
        return max(0, self.pred_notes - self.pitch_tp)


@dataclass
class TrackReport:
    track_id: str
    bucket: str
    source_multi_pitch_f1: float | None
    source_tablature_f1: float | None
    source_gap: float | None
    source_ref_silence_ratio: float | None
    counts: Counts
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def metric_value(section: dict[str, float], key: str) -> float | None:
    value = section.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1(tp: int, pred: int, ref: int) -> float:
    precision = rate(tp, pred)
    recall = rate(tp, ref)
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def summarize_counts(counts: Counts) -> dict[str, float | int]:
    return {
        "frames": counts.frames,
        "ref_notes": counts.ref_notes,
        "pred_notes": counts.pred_notes,
        "exact_tp": counts.exact_tp,
        "pitch_tp": counts.pitch_tp,
        "pitch_correct_wrong_position": counts.pitch_correct_wrong_position,
        "missed_notes": counts.missed_notes,
        "extra_notes": counts.extra_notes,
        "exact_precision": rate(counts.exact_tp, counts.pred_notes),
        "exact_recall": rate(counts.exact_tp, counts.ref_notes),
        "exact_f1": f1(counts.exact_tp, counts.pred_notes, counts.ref_notes),
        "pitch_precision": rate(counts.pitch_tp, counts.pred_notes),
        "pitch_recall": rate(counts.pitch_tp, counts.ref_notes),
        "pitch_f1": f1(counts.pitch_tp, counts.pred_notes, counts.ref_notes),
        "oracle_tab_f1": f1(counts.pitch_tp, counts.pred_notes, counts.ref_notes),
        "canonical_f1": f1(counts.canonical_tp, counts.pred_notes, counts.ref_notes),
        "wrong_position_share_of_ref": rate(counts.pitch_correct_wrong_position, counts.ref_notes),
        "missed_share_of_ref": rate(counts.missed_notes, counts.ref_notes),
        "extra_share_of_pred": rate(counts.extra_notes, counts.pred_notes),
        "ref_active_frame_ratio": rate(counts.active_frames_ref, counts.frames),
        "pred_active_frame_ratio": rate(counts.active_frames_pred, counts.frames),
    }


def experiment_results_dir(experiment_dir: Path) -> Path:
    results_dir = experiment_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")
    return results_dir


def load_track_candidates(experiment_dir: Path, max_tracks: int, max_ref_silence: float) -> list[TrackCandidate]:
    results_dir = experiment_results_dir(experiment_dir)
    summary = load_json(results_dir / "summary.json")
    diagnostics = summary.get("synthtab_val", {}).get("diagnostics", {})
    diagnostic_tracks = {
        track.get("track_id"): track
        for track in diagnostics.get("tracks", [])
        if isinstance(track, dict) and isinstance(track.get("track_id"), str)
    }

    candidates: list[TrackCandidate] = []
    for txt_path in sorted(results_dir.rglob("*.txt")):
        sections = parse_metric_txt(txt_path)
        mp = metric_value(sections.get("multi_pitch", {}), "f1-score")
        tab = metric_value(sections.get("tablature", {}), "f1-score")
        gap = mp - tab if mp is not None and tab is not None else None
        track_id = str(txt_path.relative_to(results_dir).with_suffix(""))
        ref_silence = metric_value(diagnostic_tracks.get(track_id, {}), "ref_silence_ratio")
        candidates.append(TrackCandidate(track_id, "", mp, tab, gap, ref_silence))

    ranked = [
        candidate
        for candidate in candidates
        if candidate.gap is not None
        and candidate.tablature_f1 is not None
        and (candidate.ref_silence_ratio is None or candidate.ref_silence_ratio <= max_ref_silence)
    ]
    if not ranked:
        raise ValueError(f"No per-track metrics were found under {results_dir}")

    worst = sorted(ranked, key=lambda candidate: candidate.gap or 0.0, reverse=True)[:10]
    best = sorted(ranked, key=lambda candidate: candidate.tablature_f1 or 0.0, reverse=True)[:5]
    median_tab = float(np.median([candidate.tablature_f1 for candidate in ranked if candidate.tablature_f1 is not None]))
    typical = sorted(ranked, key=lambda candidate: abs((candidate.tablature_f1 or 0.0) - median_tab))[:5]

    selected: list[TrackCandidate] = []
    seen: set[str] = set()
    for bucket, group in [("worst_gap", worst), ("typical_tab", typical), ("best_tab", best)]:
        for candidate in group:
            if candidate.track_id in seen:
                continue
            selected.append(
                TrackCandidate(
                    candidate.track_id,
                    bucket,
                    candidate.multi_pitch_f1,
                    candidate.tablature_f1,
                    candidate.gap,
                    candidate.ref_silence_ratio,
                )
            )
            seen.add(candidate.track_id)
            if len(selected) >= max_tracks:
                return selected
    return selected


def canonical_positions_for_pitches(profile: tools.GuitarProfile, pitch_counts: Counter[int]) -> Counter[tuple[int, int]]:
    positions: Counter[tuple[int, int]] = Counter()
    used: set[tuple[int, int]] = set()
    for pitch, count in pitch_counts.items():
        valid_positions = sorted(profile.get_valid_positions(pitch), key=lambda item: (item[1], item[0]))
        for _ in range(count):
            chosen = None
            for position in valid_positions:
                if position not in used:
                    chosen = position
                    break
            if chosen is None and valid_positions:
                chosen = valid_positions[0]
            if chosen is not None:
                positions[chosen] += 1
                used.add(chosen)
    return positions


def frame_positions(tab: np.ndarray, frame_idx: int) -> Counter[tuple[int, int]]:
    positions: Counter[tuple[int, int]] = Counter()
    for string_idx in range(tab.shape[0]):
        fret = int(tab[string_idx, frame_idx])
        if fret >= 0:
            positions[(string_idx, fret)] += 1
    return positions


def pitch_counts(profile: tools.GuitarProfile, positions: Counter[tuple[int, int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for (string_idx, fret), count in positions.items():
        counts[int(profile.get_pitch(string_idx, fret))] += count
    return counts


def counter_intersection_count(left: Counter[Any], right: Counter[Any]) -> int:
    return int(sum((left & right).values()))


def normalize_tab(tab: Any) -> np.ndarray:
    arr = np.asarray(tab)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected tablature with 2 dims, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1]:
        return arr
    return arr.T


def compare_tablature(reference: np.ndarray, estimated: np.ndarray, profile: tools.GuitarProfile) -> Counts:
    ref = normalize_tab(reference)
    pred = normalize_tab(estimated)
    frames = min(ref.shape[-1], pred.shape[-1])
    counts = Counts(frames=frames)

    for frame_idx in range(frames):
        ref_positions = frame_positions(ref, frame_idx)
        pred_positions = frame_positions(pred, frame_idx)
        ref_pitches = pitch_counts(profile, ref_positions)
        pred_pitches = pitch_counts(profile, pred_positions)
        canonical_positions = canonical_positions_for_pitches(profile, pred_pitches)

        counts.ref_notes += int(sum(ref_positions.values()))
        counts.pred_notes += int(sum(pred_positions.values()))
        counts.exact_tp += counter_intersection_count(ref_positions, pred_positions)
        counts.pitch_tp += counter_intersection_count(ref_pitches, pred_pitches)
        counts.canonical_tp += counter_intersection_count(ref_positions, canonical_positions)
        counts.active_frames_ref += int(bool(ref_positions))
        counts.active_frames_pred += int(bool(pred_positions))

    return counts


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_model_for_checkpoint(config_path: Path, checkpoint_path: Path, device_override: str) -> tuple[Any, Any, Any, Any, Any]:
    cfg = load_config(config_path)
    cfg.runtime.device = device_override
    device = resolve_device(cfg.runtime)
    data_proc = build_feature_extractor(cfg)
    profile = build_profile()
    estimator, _ = build_estimators(profile)
    model = build_model(cfg, data_proc, profile, device)

    payload = safe_torch_load(checkpoint_path, device)
    if not isinstance(payload, dict) or payload.get("checkpoint_type") != "tabcnn_synthtab_training_state":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if hasattr(model, "change_device"):
        model.change_device(device)
    if "model_iter" in payload:
        model.iter = int(payload["model_iter"])
    model.eval()
    return cfg, data_proc, profile, estimator, model


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    selected_tracks: list[TrackCandidate],
    device_override: str,
) -> dict[str, Any]:
    cfg, data_proc, profile, estimator, model = load_model_for_checkpoint(config_path, checkpoint_path, device_override)
    dataset = create_synthtab_dataset(cfg, "val", data_proc, profile)
    # Full-track diagnostics avoid random 500-frame validation windows and make
    # checkpoint-to-checkpoint comparisons deterministic for the selected tracks.
    dataset.seq_length = None
    dataset.num_frames = None
    available = set(dataset.tracks)
    missing = [track.track_id for track in selected_tracks if track.track_id not in available]
    if missing:
        raise ValueError(f"{len(missing)} selected tracks are not present in the active validation set. First missing: {missing[0]}")

    aggregate_counts = Counts()
    track_reports: list[TrackReport] = []
    with torch.no_grad():
        for index, candidate in enumerate(selected_tracks, start=1):
            print(f"[{checkpoint_path.name}] {index}/{len(selected_tracks)} {candidate.bucket}: {candidate.track_id}", flush=True)
            track_data = dataset.get_track_data(candidate.track_id)
            predictions = run_offline(track_data, model, estimator)
            counts = compare_tablature(track_data[tools.KEY_TABLATURE], predictions[tools.KEY_TABLATURE], profile)
            aggregate_counts.add(counts)
            metrics = summarize_counts(counts)
            track_reports.append(
                TrackReport(
                    track_id=candidate.track_id,
                    bucket=candidate.bucket,
                    source_multi_pitch_f1=candidate.multi_pitch_f1,
                    source_tablature_f1=candidate.tablature_f1,
                    source_gap=candidate.gap,
                    source_ref_silence_ratio=candidate.ref_silence_ratio,
                    counts=counts,
                    metrics=metrics,
                )
            )

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "model_iter": int(getattr(model, "iter", -1)),
        "tracks": [track_report_to_dict(report) for report in track_reports],
        "aggregate": summarize_counts(aggregate_counts),
    }


def track_report_to_dict(report: TrackReport) -> dict[str, Any]:
    return {
        "track_id": report.track_id,
        "bucket": report.bucket,
        "source_multi_pitch_f1": report.source_multi_pitch_f1,
        "source_tablature_f1": report.source_tablature_f1,
        "source_gap": report.source_gap,
        "source_ref_silence_ratio": report.source_ref_silence_ratio,
        **report.metrics,
    }


def write_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint_report in reports:
        checkpoint_name = checkpoint_report["checkpoint_name"]
        model_iter = checkpoint_report["model_iter"]
        for track in checkpoint_report["tracks"]:
            rows.append({"checkpoint": checkpoint_name, "model_iter": model_iter, **track})
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, report: dict[str, Any], suffix: str = "") -> None:
    suffix_part = f".{suffix}" if suffix else ""
    write_json(output_dir / f"aggregate{suffix_part}.json", report)
    write_csv(output_dir / f"per_track{suffix_part}.csv", report["checkpoints"])
    write_json(output_dir / f"per_track{suffix_part}.json", report["checkpoints"])
    (output_dir / f"summary{suffix_part}.md").write_text(render_summary(report), encoding="utf-8")


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# String/Fret Frame Diagnostic",
        "",
        f"- Config: `{report['config']}`",
        f"- Track source experiment: `{report['track_source_experiment']}`",
        f"- Selected tracks: `{len(report['selected_tracks'])}`",
        "",
        "| Checkpoint | Exact F1 | Pitch F1 | Oracle Tab F1 | Canonical F1 | Wrong-position/ref | Missed/ref | Extra/pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint_report in report["checkpoints"]:
        agg = checkpoint_report["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{checkpoint_report['checkpoint_name']}`",
                    pct(agg["exact_f1"]),
                    pct(agg["pitch_f1"]),
                    pct(agg["oracle_tab_f1"]),
                    pct(agg["canonical_f1"]),
                    pct(agg["wrong_position_share_of_ref"]),
                    pct(agg["missed_share_of_ref"]),
                    pct(agg["extra_share_of_pred"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Exact F1` measures exact string/fret agreement.",
            "- `Pitch F1` ignores string/fret placement and scores only the pitch multiset.",
            "- `Oracle Tab F1` is the upper bound if every pitch-correct but position-wrong note were moved to the reference position.",
            "- `Canonical F1` applies a deterministic pitch-to-position rule without logits; if it is worse than exact F1, hard post-processing is not useful.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--track-source-experiment", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--max-ref-silence", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("generated/diagnostics/string_fret_lespaul_20tracks"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="List selected tracks without loading model/checkpoints.")
    args = parser.parse_args()

    root = Path.cwd()
    config_path = resolve_path(args.config, root)
    source_experiment = resolve_path(args.track_source_experiment, root)
    output_dir = resolve_path(args.output_dir, root)
    selected_tracks = load_track_candidates(source_experiment, args.max_tracks, args.max_ref_silence)

    if args.dry_run:
        for track in selected_tracks:
            print(
                f"{track.bucket}\t{pct(track.multi_pitch_f1)}\t{pct(track.tablature_f1)}\t"
                f"{pct(track.gap)}\tref_silence={pct(track.ref_silence_ratio)}\t{track.track_id}"
            )
        return

    if not args.checkpoint:
        raise SystemExit("At least one --checkpoint is required unless --dry-run is used.")

    checkpoints = [resolve_path(checkpoint, root) for checkpoint in args.checkpoint]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    checkpoint_reports = []
    report = {
        "config": str(config_path),
        "track_source_experiment": str(source_experiment),
        "selected_tracks": [track.__dict__ for track in selected_tracks],
        "checkpoints": checkpoint_reports,
    }

    for checkpoint in checkpoints:
        checkpoint_reports.append(evaluate_checkpoint(config_path, checkpoint, selected_tracks, args.device))
        write_outputs(output_dir, report, suffix="partial")

    write_outputs(output_dir, report)
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
