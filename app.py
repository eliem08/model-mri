from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from runtime import load_bundle, forward_text, OUT
from provenance import build_manifest

def run(text="A short sentence to inspect.", real_model=False, model_id=None):
    b=load_bundle(real_model=real_model, model_id=model_id)
    batch,out=forward_text(b,text)
    rows=[]; mat=[]
    for i,h in enumerate(out.hidden_states or []):
        x=h[0,-1].detach().float().cpu()
        rows.append({"layer":i,"shape":list(x.shape),"mean":float(x.mean()),"std":float(x.std())})
        mat.append(x.numpy())
    fig,ax=plt.subplots(figsize=(10,4)); ax.imshow(mat,aspect="auto",interpolation="nearest")
    ax.set(title="Hidden-state activation heatmap",xlabel="hidden dimension",ylabel="embedding / transformer layer"); fig.tight_layout()
    plot=OUT/"activation_heatmap.png"; fig.savefig(plot,dpi=130); plt.close(fig)
    report={"model":b.model_id,"device":b.device,"text":text,"sequence_length":int(batch["input_ids"].shape[-1]),"layers":rows,"plot":str(plot),"manifest":build_manifest("model-mri",model_id=b.model_id,device=b.device,config={"real_model":real_model},inputs={"text":text})}
    (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
