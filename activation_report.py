"""Profile transformer hidden states and export a reproducible activation report."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from provenance import build_manifest

def choose_device(requested: str) -> str:
    if requested != "auto": return requested
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def profile(model_id: str, text: str, output: str = "outputs/activation-report", device: str = "auto", max_length: int = 256) -> dict:
    device = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.to(device).eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.inference_mode():
        result = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = result.hidden_states
    rows = []
    for idx, state in enumerate(hidden):
        values = state[0].float()
        rows.append({
            "layer": idx,
            "tokens": int(values.shape[0]),
            "width": int(values.shape[-1]),
            "mean_abs": float(values.abs().mean().cpu()),
            "rms": float(values.square().mean().sqrt().cpu()),
            "max_abs": float(values.abs().max().cpu()),
            "zero_fraction": float((values.abs() < 1e-8).float().mean().cpu()),
        })
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    report = {"model_id": model_id, "device": device, "input_text": text, "token_count": int(inputs["input_ids"].shape[1]), "layers": rows, "manifest": build_manifest("model-mri", model_id=model_id, device=device, config={"max_length": max_length}, inputs={"text": text})}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "layers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    token_rows = []
    for token_i, token in enumerate(tokens):
        for layer_i, state in enumerate(hidden):
            vec = state[0, token_i].float()
            token_rows.append({"token_index": token_i, "token": token, "layer": layer_i, "mean_abs": float(vec.abs().mean().cpu()), "rms": float(vec.square().mean().sqrt().cpu())})
    with (out / "token-layer.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(token_rows[0])); writer.writeheader(); writer.writerows(token_rows)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="distilgpt2"); p.add_argument("--text", default="Mechanistic interpretability studies internal representations.")
    p.add_argument("--output", default="outputs/activation-report"); p.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto"); p.add_argument("--max-length", type=int, default=256)
    a = p.parse_args(); report = profile(a.model_id, a.text, a.output, a.device, a.max_length)
    print(json.dumps({"model_id":report["model_id"],"device":report["device"],"layers":len(report["layers"]),"output":a.output}, indent=2))
if __name__ == "__main__": main()
