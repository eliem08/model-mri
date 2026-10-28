# Model MRI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/model-mri/blob/main/colab.ipynb)

A small interpretability workbench for inspecting hidden-state statistics across transformer layers.

## Activation report

    pip install -r requirements.txt
    python activation_report.py --model-id distilgpt2 --text "The sky is blue."

Creates outputs/activation-report/report.json, layers.csv, and token-layer.csv. Reports layer-wise RMS, mean absolute activation, maximum absolute activation, zero fraction, and per-token/per-layer summaries.

    python activation_report.py --model-id distilgpt2 --device cuda --output outputs/mri-run

The device auto-option selects CUDA, Apple MPS, or CPU. In Colab, select a GPU runtime, install requirements, then run the command above.

## Quickstart

    python main.py

The original demo uses tiny random weights to validate the software path. Use a pretrained model for meaningful activation measurements.

## Interpretation cautions

Activation magnitudes are descriptive statistics, not explanations of model behavior. Compare prompts, layers, tokenization, and model versions; keep exact model IDs and inputs with results. Avoid claims about intent or deception from activation patterns alone.

## License

MIT.
