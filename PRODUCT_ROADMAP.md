# Model MRI — Product Roadmap

## Product outcome
A local-first interpretability workbench where researchers load an open-weight model, compare hidden-state statistics across prompts and layers, export artifacts, and reproduce an analysis later.

## Users
- Interpretability researchers running activation diagnostics.
- Model developers debugging representation changes across checkpoints.
- Students learning hidden-state tensor structure.

## Current foundation
- CLI activation report and JSON/CSV exports.
- Streamlit dashboard and Colab notebook.
- CPU/GPU device selection, tests, and CI.

## Phase 1 — Reliable analysis
- [ ] Batch prompt input and side-by-side comparisons.
- [ ] Record model revision, tokenizer revision, package versions, device, seed, and timestamp in every artifact.
- [ ] Add layer/token heatmaps and a downloadable report bundle.
- [ ] Add memory guardrails and clearer out-of-memory guidance.
- [ ] Test with tiny initialized models in CI without large model downloads.

## Phase 2 — Research-grade workflow
- [ ] Activation patching with clean/corrupted prompt pairs.
- [ ] Compare checkpoints and quantify representation drift.
- [ ] Dataset mode with CSV/JSONL input and resumable jobs.
- [ ] Experiment registry and artifact manifests.
- [ ] Architecture adapters beyond GPT-2-style models.

## Phase 3 — Final product
A local web app with a model library, prompt comparison, layer explorer, activation heatmaps, reproducible experiment folders, and exportable research reports. Team hosting is a later option only after authentication and data-retention controls exist.

## Success criteria
- One-command local launch.
- Reports include exact inputs and model revision.
- Unit tests run without downloading large weights.
- Every chart traces back to a saved configuration.

## Scientific caution
Activation statistics are descriptive. They do not independently establish why a model produced an answer. Pair visualizations with controls and task-level evidence.
