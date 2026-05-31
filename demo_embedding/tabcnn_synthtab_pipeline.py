from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import amt_tools.tools as tools
from amt_tools.evaluate import (
    ComboEvaluator,
    LossWrapper,
    MultipitchEvaluator,
    SoftmaxAccuracy,
    TablatureEvaluator,
    validate,
)
from amt_tools.features import CQT
from amt_tools.models import TabCNN
from amt_tools.inference import run_offline
from amt_tools.transcribe import ComboEstimator, StackedMultiPitchCollapser, TablatureWrapper

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from GuitarSet import GuitarSet
from SynthTab import SynthTab, load_stacked_notes_jams
from train import train


DEFAULT_CONFIG_PATH = Path(__file__).with_name("tabcnn_synthtab_baseline.json")


@dataclass
class DatasetPaths:
    synthtab: str
    cache_dir: str
    guitarset: str | None = None
    jams_dir: str | None = None


@dataclass
class FeatureConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    n_bins: int = 192
    bins_per_octave: int = 24


@dataclass
class TrainConfig:
    num_frames: int = 500
    epochs: int = 100
    checkpoints: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_eta_min: float = 1e-6
    n_workers: int = 0
    seed: int = 0
    reset_data: bool = False
    augment_audio: bool = False
    sample_attempts: int = 5
    limit_train_tracks: int | None = None
    limit_val_tracks: int | None = None
    sanity_steps: int | None = None
    use_class_weights: bool = True
    silence_weight: float = 0.1
    note_weight: float = 1.0
    resume_from: str | None = None
    resume_strict: bool = True
    save_full_checkpoints: bool = True
    sampler: str = "shuffle"
    balance_by_group: bool = True
    balance_by_silence: bool = False
    use_amp: bool = False


@dataclass
class EvalConfig:
    run_synthtab_val: bool = True
    run_guitarset: bool = False
    guitarset_splits: list[str] = field(default_factory=lambda: ["09"])


@dataclass
class RuntimeConfig:
    device: str = "auto"
    gpu_id: int = 0
    experiment_root: str = "./generated"
    experiment_name: str = "tabcnn_synthtab_baseline"
    save_checkpoints: bool = True
    pin_memory: bool = True


@dataclass
class PipelineConfig:
    paths: DatasetPaths
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate TabCNN on SynthTab.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to JSON config.")
    parser.add_argument("--mode", choices=["train", "eval", "inspect"], default="train")
    parser.add_argument("--model-path", type=Path, help="Checkpoint to evaluate. Required for eval mode.")
    parser.add_argument("--experiment-dir", type=Path, help="Override experiment directory.")
    return parser.parse_args()


def load_config(config_path: Path) -> PipelineConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return PipelineConfig(
        paths=DatasetPaths(**raw["paths"]),
        features=FeatureConfig(**raw.get("features", {})),
        train=TrainConfig(**raw.get("train", {})),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )


def resolve_device(runtime: RuntimeConfig) -> torch.device:
    if runtime.device == "cpu":
        return torch.device("cpu")

    if runtime.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly, but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{runtime.gpu_id}")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{runtime.gpu_id}")

    return torch.device("cpu")


def build_feature_extractor(cfg: PipelineConfig) -> CQT:
    return CQT(
        sample_rate=cfg.features.sample_rate,
        hop_length=cfg.features.hop_length,
        n_bins=cfg.features.n_bins,
        bins_per_octave=cfg.features.bins_per_octave,
    )


def build_profile() -> tools.GuitarProfile:
    return tools.GuitarProfile(num_frets=19)


def build_estimators(profile: tools.GuitarProfile) -> tuple[ComboEstimator, ComboEvaluator]:
    estimator = ComboEstimator(
        [TablatureWrapper(profile=profile), StackedMultiPitchCollapser(profile=profile)]
    )
    evaluator = ComboEvaluator(
        [LossWrapper(), MultipitchEvaluator(), TablatureEvaluator(profile=profile), SoftmaxAccuracy()]
    )
    evaluator.set_patterns(["loss", "pr", "re", "f1", "tdr", "acc"])
    return estimator, evaluator


def ensure_synthtab_layout(base_dir: Path, jams_dir: str | None = None) -> None:
    standard_ok = all((base_dir / split).exists() for split in SynthTab.available_splits())
    dev_partitions = ["acoustic", "electric_clean", "electric_distortion", "electric_distortion_di", "electric_muted"]
    jams_root = Path(jams_dir) if jams_dir else base_dir / "jams"
    dev_ok = any((base_dir / name).exists() for name in dev_partitions) and jams_root.exists()

    if not standard_ok and not dev_ok:
        raise FileNotFoundError(
            f"SynthTab layout was not recognized under '{base_dir}'. "
            "Expected either train/val directories or a dev/full chunk layout "
            "with at least one audio partition and a matching JAMS root. "
            f"Resolved JAMS root: '{jams_root}'."
        )


def create_synthtab_dataset(
    cfg: PipelineConfig,
    split: str,
    data_proc: CQT,
    profile: tools.GuitarProfile,
) -> SynthTab:
    base_dir = Path(cfg.paths.synthtab)
    standard_layout = all((base_dir / name).exists() for name in SynthTab.available_splits())
    guitars = (["luthier", "martin", "taylor"] if split == "train" else ["gibson"]) if standard_layout else None
    num_frames = cfg.train.num_frames if split == "train" else cfg.train.num_frames
    dataset = SynthTab(
        base_dir=cfg.paths.synthtab,
        splits=[split],
        guitars=guitars,
        hop_length=cfg.features.hop_length,
        sample_rate=cfg.features.sample_rate,
        num_frames=num_frames,
        sample_attempts=cfg.train.sample_attempts if split == "train" else 1,
        augment_audio=cfg.train.augment_audio if split == "train" else False,
        include_onsets=False,
        data_proc=data_proc,
        profile=profile,
        reset_data=cfg.train.reset_data,
        store_data=False,
        save_data=True,
        save_loc=cfg.paths.cache_dir,
        seed=cfg.train.seed,
        jams_dir=cfg.paths.jams_dir,
    )
    limit = cfg.train.limit_train_tracks if split == "train" else cfg.train.limit_val_tracks
    if limit is not None:
        dataset.tracks = dataset.tracks[: min(limit, len(dataset.tracks))]
    return dataset


def create_guitarset_dataset(
    cfg: PipelineConfig,
    data_proc: CQT,
    profile: tools.GuitarProfile,
) -> GuitarSet:
    if not cfg.paths.guitarset:
        raise FileNotFoundError("GuitarSet path is not configured.")

    return GuitarSet(
        base_dir=cfg.paths.guitarset,
        splits=cfg.evaluation.guitarset_splits,
        hop_length=cfg.features.hop_length,
        sample_rate=cfg.features.sample_rate,
        num_frames=None,
        audio_norm=np.inf,
        data_proc=data_proc,
        profile=profile,
        store_data=False,
        reset_data=cfg.train.reset_data,
        save_loc=cfg.paths.cache_dir,
        seed=cfg.train.seed,
    )


def track_group(track: str) -> str:
    parts = Path(track).parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if parts:
        return parts[0]
    return "unknown"


def estimate_track_non_silent_ratio(dataset: Any, track: str) -> float | None:
    """Estimate track note density from cached ground truth/JAMS without touching audio."""
    try:
        jams_path = dataset.get_jams_path(track)
        stacked_notes = load_stacked_notes_jams(jams_path)
    except Exception:
        return None

    durations = []
    latest_end = 0.0
    for string_notes in stacked_notes.values():
        # amt-tools stacked notes are usually stored as (pitches, intervals).
        if isinstance(string_notes, tuple) and len(string_notes) >= 2:
            intervals = string_notes[1]
        else:
            intervals = string_notes

        for interval in np.asarray(intervals):
            if np.asarray(interval).size < 2:
                continue
            start = float(interval[0])
            end = float(interval[1])
            if end > start:
                durations.append(end - start)
                latest_end = max(latest_end, end)

    if latest_end <= 0.0:
        return 0.0

    # Sum note durations across strings. Clamp to 1.0 because overlapping notes can exceed wall-clock time.
    return min(1.0, float(sum(durations) / latest_end))


def silence_bucket(non_silent_ratio: float | None) -> str:
    if non_silent_ratio is None:
        return "unknown_density"
    if non_silent_ratio < 0.15:
        return "low_density"
    if non_silent_ratio < 0.45:
        return "mid_density"
    return "high_density"


def build_balanced_sampler(dataset: Any, cfg: PipelineConfig) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    tracks = list(getattr(dataset, "tracks", []))
    if not tracks:
        raise ValueError("Cannot build a balanced sampler for a dataset without tracks.")

    keys = []
    metadata = []
    for track in tracks:
        group = track_group(track) if cfg.train.balance_by_group else "all_groups"
        ratio = estimate_track_non_silent_ratio(dataset, track) if cfg.train.balance_by_silence else None
        density = silence_bucket(ratio) if cfg.train.balance_by_silence else "all_densities"
        key = f"{group}|{density}"
        keys.append(key)
        metadata.append({"track": track, "group": group, "non_silent_ratio": ratio, "density_bucket": density})

    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1

    weights = torch.as_tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(cfg.train.seed)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=generator)
    summary = {
        "strategy": "balanced",
        "balance_by_group": cfg.train.balance_by_group,
        "balance_by_silence": cfg.train.balance_by_silence,
        "num_tracks": len(tracks),
        "num_buckets": len(counts),
        "bucket_counts": dict(sorted(counts.items())),
        "tracks": metadata,
    }
    return sampler, summary


def build_dataloader(dataset: Any, cfg: PipelineConfig, shuffle: bool) -> tuple[DataLoader, dict[str, Any]]:
    sampler = None
    sampler_summary: dict[str, Any] = {"strategy": "shuffle" if shuffle else "sequential"}
    use_shuffle = shuffle

    if shuffle and cfg.train.sampler == "balanced":
        sampler, sampler_summary = build_balanced_sampler(dataset, cfg)
        use_shuffle = False
    elif cfg.train.sampler not in {"shuffle", "balanced"}:
        raise ValueError(f"Unsupported train.sampler='{cfg.train.sampler}'. Expected 'shuffle' or 'balanced'.")

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=use_shuffle,
        sampler=sampler,
        pin_memory=cfg.runtime.pin_memory and torch.cuda.is_available(),
        num_workers=cfg.train.n_workers,
        drop_last=shuffle,
    )
    return loader, sampler_summary



def safe_torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def build_model(cfg: PipelineConfig, data_proc: CQT, profile: tools.GuitarProfile, device: torch.device) -> TabCNN:
    model = TabCNN(
        dim_in=data_proc.get_feature_size(),
        profile=profile,
        in_channels=data_proc.get_num_channels(),
        device=device.index if device.type == "cuda" else "cpu",
    )
    if cfg.train.use_class_weights:
        output_layer = model.dense[-1]
        class_weights = np.full((output_layer.num_groups, output_layer.num_classes), cfg.train.note_weight, dtype=np.float32)
        class_weights[:, -1] = cfg.train.silence_weight
        output_layer.set_weights(class_weights.flatten(), device=device)
    model.change_device(device)
    model.train()
    return model


def build_grad_scaler(amp_enabled: bool) -> Any:
    if not amp_enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)

    return torch.cuda.amp.GradScaler(enabled=True)


def load_training_checkpoint(
    checkpoint_path: Path,
    model: TabCNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    strict: bool,
    scaler: Any = None,
) -> tuple[TabCNN, int]:
    payload = safe_torch_load(checkpoint_path, device)

    if isinstance(payload, dict) and payload.get("checkpoint_type") == "tabcnn_synthtab_training_state":
        model.load_state_dict(payload["model_state_dict"], strict=strict)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])

        if hasattr(model, "change_device"):
            model.change_device(device)
        if "model_iter" in payload:
            model.iter = int(payload["model_iter"])

        random_state = payload.get("random_state", {})
        if "python" in random_state:
            random.setstate(random_state["python"])
        if "numpy" in random_state:
            np.random.set_state(random_state["numpy"])
        if "torch" in random_state:
            torch_state = random_state["torch"]
            if isinstance(torch_state, torch.Tensor):
                torch_state = torch_state.detach().cpu()
            torch.random.set_rng_state(torch_state)
        if device.type == "cuda" and random_state.get("cuda") is not None:
            cuda_states = [
                state.detach().cpu() if isinstance(state, torch.Tensor) else state
                for state in random_state["cuda"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)

        return model, int(payload.get("next_epoch", payload.get("epoch", -1) + 1))

    # Backward compatibility note: legacy model-only files are still supported in eval mode,
    # but they do not contain optimizer/scheduler/RNG state and are unsafe for resume.
    if hasattr(payload, "run_on_batch"):
        raise ValueError(
            f"'{checkpoint_path}' is a legacy model-only checkpoint. "
            "Use it with --mode eval, or resume from training-state-*.pt."
        )

    raise ValueError(
        f"Unsupported checkpoint format in '{checkpoint_path}'. "
        "Expected a full training-state checkpoint created as training-state-*.pt."
    )


def make_experiment_dir(cfg: PipelineConfig, override: Path | None) -> Path:
    if override is not None:
        experiment_dir = override
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = Path(cfg.runtime.experiment_root) / f"{cfg.runtime.experiment_name}_{timestamp}"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir.resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_results(results: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, dict):
            normalized[key] = normalize_results(value)
        elif isinstance(value, (np.floating, np.integer)):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def compute_tablature_diagnostics(
    model: TabCNN,
    dataset: Any,
    estimator: ComboEstimator,
) -> dict[str, Any]:
    ref_silence_ratios: list[float] = []
    pred_silence_ratios: list[float] = []
    non_silent_accuracies: list[float] = []
    track_summaries: list[dict[str, Any]] = []

    with torch.no_grad():
        for track_id in dataset.tracks:
            track_data = dataset.get_track_data(track_id)
            predictions = run_offline(track_data, model, estimator)

            reference = track_data[tools.KEY_TABLATURE]
            estimated = predictions[tools.KEY_TABLATURE]

            ref_silence_mask = reference == -1
            pred_silence_mask = estimated == -1
            non_silent_mask = ~ref_silence_mask

            ref_silence_ratio = float(np.mean(ref_silence_mask))
            pred_silence_ratio = float(np.mean(pred_silence_mask))
            ref_non_silent_ratio = 1.0 - ref_silence_ratio
            pred_non_silent_ratio = 1.0 - pred_silence_ratio

            if np.any(non_silent_mask):
                non_silent_accuracy = float(np.mean(estimated[non_silent_mask] == reference[non_silent_mask]))
            else:
                non_silent_accuracy = 0.0

            ref_silence_ratios.append(ref_silence_ratio)
            pred_silence_ratios.append(pred_silence_ratio)
            non_silent_accuracies.append(non_silent_accuracy)
            track_summaries.append(
                {
                    "track_id": str(track_id),
                    "ref_silence_ratio": ref_silence_ratio,
                    "pred_silence_ratio": pred_silence_ratio,
                    "ref_non_silent_ratio": ref_non_silent_ratio,
                    "pred_non_silent_ratio": pred_non_silent_ratio,
                    "non_silent_accuracy": non_silent_accuracy,
                }
            )

    return {
        "ref_silence_ratio": float(np.mean(ref_silence_ratios)) if ref_silence_ratios else 0.0,
        "pred_silence_ratio": float(np.mean(pred_silence_ratios)) if pred_silence_ratios else 0.0,
        "ref_non_silent_ratio": 1.0 - (float(np.mean(ref_silence_ratios)) if ref_silence_ratios else 0.0),
        "pred_non_silent_ratio": 1.0 - (float(np.mean(pred_silence_ratios)) if pred_silence_ratios else 0.0),
        "non_silent_accuracy": float(np.mean(non_silent_accuracies)) if non_silent_accuracies else 0.0,
        "collapse_to_silence": bool(pred_silence_ratios and np.mean(pred_silence_ratios) >= 0.99),
        "num_tracks": len(track_summaries),
        "tracks": track_summaries,
    }


def inspect_environment(cfg: PipelineConfig) -> dict[str, Any]:
    synthtab_dir = Path(cfg.paths.synthtab)
    cache_dir = Path(cfg.paths.cache_dir)
    guitarset_dir = Path(cfg.paths.guitarset) if cfg.paths.guitarset else None
    jams_dir = Path(cfg.paths.jams_dir) if cfg.paths.jams_dir else synthtab_dir / "jams"

    ensure_synthtab_layout(synthtab_dir, cfg.paths.jams_dir)

    summary = {
        "synthtab_path": str(synthtab_dir.resolve()),
        "synthtab_exists": synthtab_dir.exists(),
        "synthtab_train_exists": (synthtab_dir / "train").exists(),
        "synthtab_val_exists": (synthtab_dir / "val").exists(),
        "synthtab_audio_partitions": [
            name
            for name in ["acoustic", "electric_clean", "electric_distortion", "electric_distortion_di", "electric_muted"]
            if (synthtab_dir / name).exists()
        ],
        "jams_dir": str(jams_dir.resolve()),
        "jams_dir_exists": jams_dir.exists(),
        "cache_dir": str(cache_dir),
        "cache_exists": cache_dir.exists(),
        "guitarset_path": str(guitarset_dir.resolve()) if guitarset_dir else None,
        "guitarset_exists": guitarset_dir.exists() if guitarset_dir else False,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():
        summary["cuda_device_name"] = torch.cuda.get_device_name(0)

    return summary


def run_train(cfg: PipelineConfig, experiment_dir: Path) -> None:
    device = resolve_device(cfg.runtime)
    tools.seed_everything(cfg.train.seed)

    ensure_synthtab_layout(Path(cfg.paths.synthtab), cfg.paths.jams_dir)
    Path(cfg.paths.cache_dir).mkdir(parents=True, exist_ok=True)

    data_proc = build_feature_extractor(cfg)
    profile = build_profile()
    estimator, evaluator = build_estimators(profile)

    train_set = create_synthtab_dataset(cfg, "train", data_proc, profile)
    val_set = create_synthtab_dataset(cfg, "val", data_proc, profile)
    train_loader, sampler_summary = build_dataloader(train_set, cfg, shuffle=True)

    model = build_model(cfg, data_proc, profile, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.scheduler_eta_min
    )
    amp_enabled = bool(cfg.train.use_amp and device.type == "cuda")
    scaler = build_grad_scaler(amp_enabled)

    start_epoch = 0
    resume_path = Path(cfg.train.resume_from) if cfg.train.resume_from else None
    if resume_path is not None:
        model, start_epoch = load_training_checkpoint(
            resume_path, model, optimizer, scheduler, device,
            strict=cfg.train.resume_strict, scaler=scaler
        )

    metadata = {
        "config": asdict(cfg),
        "resolved_device": str(device),
        "train_tracks": len(train_set),
        "val_tracks": len(val_set),
        "started_at": datetime.now().isoformat(),
        "run_mode": "resume" if resume_path is not None else "fresh",
        "resume_from": str(resume_path.resolve()) if resume_path is not None else None,
        "start_epoch": start_epoch,
        "start_iter": int(getattr(model, "iter", 0)),
        "amp_enabled": amp_enabled,
        "sampler": sampler_summary,
    }
    write_json(experiment_dir / "run_config.json", metadata)

    model_dir = experiment_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=cfg.train.epochs,
        scheduler=scheduler,
        checkpoints=cfg.train.checkpoints,
        log_dir=str(model_dir),
        val_set=val_set,
        estimator=estimator,
        evaluator=evaluator,
        start_epoch=start_epoch,
        sanity_steps=cfg.train.sanity_steps,
        save_full_checkpoints=cfg.train.save_full_checkpoints and cfg.runtime.save_checkpoints,
        config_snapshot=asdict(cfg),
        use_amp=amp_enabled,
        device=device,
        scaler=scaler,
    )

    eval_results = run_evaluation(cfg, experiment_dir, model=model, data_proc=data_proc, profile=profile)
    eval_results["training_run"] = {
        "run_mode": "resume" if resume_path is not None else "fresh",
        "resume_from": str(resume_path.resolve()) if resume_path is not None else None,
        "sampler": sampler_summary,
        "final_iter": int(getattr(model, "iter", 0)),
        "finished_at": datetime.now().isoformat(),
    }
    write_json(experiment_dir / "results" / "summary.json", eval_results)


def run_evaluation(
    cfg: PipelineConfig,
    experiment_dir: Path,
    model: TabCNN | None = None,
    data_proc: CQT | None = None,
    profile: tools.GuitarProfile | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(cfg.runtime)

    if data_proc is None:
        data_proc = build_feature_extractor(cfg)

    if profile is None:
        profile = build_profile()

    if model is None:
        if model_path is None:
            raise ValueError("model_path is required when evaluating without an in-memory model.")
        model = safe_torch_load(model_path, device)
        model.change_device(device)
        model.eval()

    estimator, evaluator = build_estimators(profile)
    evaluator.set_save_dir(str(experiment_dir / "results"))
    evaluator.set_patterns(None)

    results: dict[str, Any] = {
        "resolved_device": str(device),
        "evaluated_at": datetime.now().isoformat(),
    }

    if cfg.evaluation.run_synthtab_val:
        synthtab_val = create_synthtab_dataset(cfg, "val", data_proc, profile)
        synthtab_results = validate(model, synthtab_val, evaluator=evaluator, estimator=estimator)
        results["synthtab_val"] = normalize_results(synthtab_results)
        results["synthtab_val"]["diagnostics"] = compute_tablature_diagnostics(model, synthtab_val, estimator)
        evaluator.reset_results()

    if cfg.evaluation.run_guitarset:
        guitarset_test = create_guitarset_dataset(cfg, data_proc, profile)
        guitarset_results = validate(model, guitarset_test, evaluator=evaluator, estimator=estimator)
        results["guitarset"] = normalize_results(guitarset_results)
        results["guitarset"]["diagnostics"] = compute_tablature_diagnostics(model, guitarset_test, estimator)
        evaluator.reset_results()

    return results


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    experiment_dir = make_experiment_dir(cfg, args.experiment_dir)

    if args.mode == "inspect":
        summary = inspect_environment(cfg)
        write_json(experiment_dir / "inspect_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.mode == "eval":
        if args.model_path is None:
            raise ValueError("--model-path is required in eval mode.")
        results = run_evaluation(cfg, experiment_dir, model_path=args.model_path)
        write_json(experiment_dir / "results" / "summary.json", results)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    run_train(cfg, experiment_dir)


if __name__ == "__main__":
    main()
