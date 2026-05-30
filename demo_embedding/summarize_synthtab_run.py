from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a TabCNN/SynthTab experiment directory.")
    parser.add_argument("experiment_dir", type=Path, help="Path to an experiment directory.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def checkpoint_iter(path: Path) -> int:
    match = re.search(r"training-state-(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def latest_training_state(experiment_dir: Path) -> str | None:
    model_dir = experiment_dir / "models"
    if not model_dir.exists():
        return None

    checkpoints = sorted(model_dir.glob("training-state-*.pt"), key=checkpoint_iter)
    return str(checkpoints[-1]) if checkpoints else None


def nested_get(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_report(experiment_dir: Path) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    run_config = load_json(experiment_dir / "run_config.json")
    summary = load_json(experiment_dir / "results" / "summary.json")

    started_at = parse_datetime(nested_get(run_config, "started_at"))
    finished_at = parse_datetime(nested_get(summary, "training_run", "finished_at"))
    runtime_seconds = None
    if started_at and finished_at:
        runtime_seconds = int((finished_at - started_at).total_seconds())

    diagnostics = nested_get(summary, "synthtab_val", "diagnostics")
    report = {
        "experiment_dir": str(experiment_dir),
        "summary_exists": summary is not None,
        "run_mode": nested_get(run_config, "run_mode"),
        "started_at": started_at.isoformat() if started_at else None,
        "finished_at": finished_at.isoformat() if finished_at else None,
        "runtime_seconds": runtime_seconds,
        "resolved_device": nested_get(run_config, "resolved_device") or nested_get(summary, "resolved_device"),
        "train_tracks": nested_get(run_config, "train_tracks"),
        "val_tracks": nested_get(run_config, "val_tracks"),
        "start_iter": nested_get(run_config, "start_iter"),
        "final_iter": nested_get(summary, "training_run", "final_iter"),
        "latest_training_state": latest_training_state(experiment_dir),
        "multi_pitch": nested_get(summary, "synthtab_val", "multi_pitch"),
        "tablature": nested_get(summary, "synthtab_val", "tablature"),
        "diagnostics": diagnostics,
        "collapse_to_silence": nested_get(diagnostics, "collapse_to_silence"),
        "pred_silence_ratio": nested_get(diagnostics, "pred_silence_ratio"),
    }
    return report


def format_seconds(seconds: int | None) -> str:
    if seconds is None:
        return "n/a"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def metric_line(name: str, payload: dict[str, Any] | None) -> str:
    if not payload:
        return f"{name}: n/a"

    parts = []
    for key in ("precision", "recall", "f1-score", "tdr", "accuracy"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.4f}")
    return f"{name}: " + (", ".join(parts) if parts else "n/a")


def print_text_report(report: dict[str, Any]) -> None:
    print(f"experiment: {report['experiment_dir']}")
    print(f"summary_exists: {report['summary_exists']}")
    print(f"run_mode: {report.get('run_mode')}")
    print(f"device: {report.get('resolved_device')}")
    print(f"tracks: train={report.get('train_tracks')} val={report.get('val_tracks')}")
    print(f"iters: start={report.get('start_iter')} final={report.get('final_iter')}")
    print(f"runtime: {format_seconds(report.get('runtime_seconds'))}")
    print(f"latest_training_state: {report.get('latest_training_state')}")

    if not report["summary_exists"]:
        print("summary: missing")
        return

    print(metric_line("multi_pitch", report.get("multi_pitch")))
    print(metric_line("tablature", report.get("tablature")))
    diagnostics = report.get("diagnostics") or {}
    print(f"collapse_to_silence: {report.get('collapse_to_silence')}")
    if isinstance(report.get("pred_silence_ratio"), (int, float)):
        print(f"pred_silence_ratio: {report['pred_silence_ratio']:.4f}")
    if isinstance(diagnostics.get("ref_silence_ratio"), (int, float)):
        print(f"ref_silence_ratio: {diagnostics['ref_silence_ratio']:.4f}")
    if isinstance(diagnostics.get("non_silent_accuracy"), (int, float)):
        print(f"non_silent_accuracy: {diagnostics['non_silent_accuracy']:.4f}")


def main() -> None:
    args = parse_args()
    report = build_report(args.experiment_dir)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
