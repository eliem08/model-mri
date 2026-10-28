"""Streamlit dashboard for Model MRI. Run: streamlit run dashboard.py"""
import streamlit as st
import pandas as pd
from activation_report import profile
st.set_page_config(page_title="Model MRI",layout="wide")
st.title("Model MRI")
st.caption("Transformer activation profiling · descriptive diagnostics, not causal explanations")
model=st.text_input("Hugging Face model ID","distilgpt2")
prompt=st.text_area("Input text","The sky is blue.")
device=st.selectbox("Device",["auto","cuda","mps","cpu"])
if st.button("Profile activations",type="primary"):
    with st.spinner("Loading model and profiling hidden states…"):
        report=profile(model,prompt,device=device)
    st.success(f"Profiled {len(report['layers'])} hidden-state tensors on {report['device']}.")
    df=pd.DataFrame(report["layers"])
    st.line_chart(df.set_index("layer")[["mean_abs","rms","max_abs"]])
    st.dataframe(df,use_container_width=True)
    st.json({"model_id":report["model_id"],"device":report["device"],"token_count":report["token_count"]})
    st.download_button("Download report JSON",pd.Series([__import__("json").dumps(report,indent=2)]).iloc[0],file_name="activation-report.json")
