# SynthTab TabCNN Baseline

This repository is a standalone snapshot of the local `TabCNN on SynthTab` baseline work.

It contains:

- the config-driven training and evaluation pipeline from `demo_embedding`
- the dataset loader changes required for `SynthTab Dev`
- the experiment configs used during tuning
- the best validation metrics from the final `28`-epoch run
- the project experiment log

It does not contain:

- the `SynthTab Dev` dataset
- generated caches
- full checkpoint directories
- virtual environments

## Best run

Best configuration:

- dataset: `SynthTab Dev`
- device: `cuda:0`
- train tracks: `134`
- val tracks: `34`
- epochs: `28`
- `silence_weight = 0.1`
- `note_weight = 1.0`

Best validation metrics:

- `multi_pitch f1 = 63.21%`
- `tablature f1 = 41.51%`
- `accuracy = 71.78%`
- `tdr = 72.02%`
- `non_silent_accuracy = 52.61%`
- `collapse_to_silence = false`

Artifacts included in this repo:

- [artifacts/baseline_dev_full_train_28ep_run/run_config.json](./artifacts/baseline_dev_full_train_28ep_run/run_config.json)
- [artifacts/baseline_dev_full_train_28ep_run/results/summary.json](./artifacts/baseline_dev_full_train_28ep_run/results/summary.json)

## Layout

- `demo_embedding/` - pipeline, loader, model code, wrappers, and configs
- `artifacts/` - selected run outputs for the best experiment
- `PROJECT_LOG.md` - chronological local research log

## Entry point

Recommended entry point:

```bash
python demo_embedding/tabcnn_synthtab_pipeline.py --mode inspect --config demo_embedding/tabcnn_synthtab_baseline_dev_full_train_28ep.json
python demo_embedding/tabcnn_synthtab_pipeline.py --mode train --config demo_embedding/tabcnn_synthtab_baseline_dev_full_train_28ep.json
python demo_embedding/tabcnn_synthtab_pipeline.py --mode eval --config demo_embedding/tabcnn_synthtab_baseline_dev_full_train_28ep.json --model-path path/to/model.pt
```


## Resume and balanced-sampler prep

The pipeline now supports long-run preparation for chunk-based full SynthTab training:

- `train.resume_from` can resume from `training-state-*.pt` full training-state checkpoints.
- `train.sampler = "balanced"` enables weighted balanced sampling by track/timbre group, with optional JAMS-based note-density buckets.
- `demo_embedding/tabcnn_synthtab_resume_balanced_smoke.json` is a small CPU smoke config for SynthTab Dev.
- `demo_embedding/tabcnn_synthtab_full_chunk_template.json` is a template for later full/chunk SynthTab runs on the desktop.

Legacy `model-*.pt` files remain usable for evaluation; use `training-state-*.pt` for resume.


## Laptop smoke verification

A CPU-only notebook smoke path was verified on SynthTab Dev:

```powershell
py -3.10 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip setuptools wheel
./.venv/Scripts/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r workspace/SynthTab/requirements.txt
./.venv/Scripts/python.exe demo_embedding/tabcnn_synthtab_pipeline.py --mode inspect --config demo_embedding/tabcnn_synthtab_resume_balanced_smoke.json --experiment-dir generated/experiments/laptop_inspect_smoke_root
```

For resume testing, use `training-state-*.pt` files, not legacy `model-*.pt` files. Generated smoke outputs are ignored under `generated/`.

## Notes

- Paths inside saved configs still point to the original local machine and must be adjusted before rerunning elsewhere.
- The original upstream repository is `SynthTab`; this repo is a focused derivative for the baseline work only.
