# Model MRI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/model-mri/blob/main/colab.ipynb)

A small interpretability workbench for inspecting hidden-state statistics across transformer layers.

## Interactive dashboard

    pip install -r requirements.txt
    streamlit run dashboard.py

Choose a model, enter text, run the activation profile, and inspect the layer-wise chart and table in your browser.

## Activation report

    python activation_report.py --model-id distilgpt2 --text "The sky is blue."
    python activation_report.py --model-id distilgpt2 --device cuda --output outputs/mri-run

Creates JSON and CSV artifacts with layer-wise RMS, mean absolute activation, maximum absolute activation, zero fraction, and per-token/per-layer summaries. Device auto-selection supports CUDA, Apple MPS, and CPU.

## Interpretation cautions

Activation magnitudes are descriptive statistics, not explanations of model behavior. Compare prompts, layers, tokenization, and model versions; keep exact model IDs and inputs with results. Avoid claims about intent or deception from activation patterns alone.

## License

MIT.
