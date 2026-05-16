# Demo Embeddings

You can find all pretrained models in `pretrained_models` directory.

## Recommended entrypoint for TabCNN on SynthTab

Use `tabcnn_synthtab_pipeline.py` as the main entrypoint for local experiments. It replaces the older hard-coded training/evaluation flow with a config-driven pipeline.

Default config:

- `tabcnn_synthtab_baseline.json`

Main commands:

```bash
python demo_embedding/tabcnn_synthtab_pipeline.py --mode inspect
python demo_embedding/tabcnn_synthtab_pipeline.py --mode train
python demo_embedding/tabcnn_synthtab_pipeline.py --mode eval --model-path path/to/model.pt
```

What the pipeline handles:

- configurable `SynthTab` and `GuitarSet` paths
- configurable cache/log/checkpoint directories
- automatic `CUDA` vs `CPU` device selection
- one baseline `TabCNN` training flow on `SynthTab`
- validation on `SynthTab`, plus optional evaluation on `GuitarSet`

Compatibility wrappers:

- `exp_training_from_scratch.py` now forwards to the new pipeline
- `evaluate.py` now forwards to the new pipeline in `eval` mode

`exp_finetuning.py` remains available for later fine-tuning work, but it is no longer the main baseline entrypoint.
