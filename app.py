# app.py
import os
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"


import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
import altair as alt
import traceback
from FuncionesAuxiliares import select_top2_seeds_scale, select_topK_model_configs_scale, select_topN_models_scale, collect_filter_values
from FuncionesAuxiliares import apply_filters, apply_top2_seeds, apply_topK_configs, apply_topN_models, get_metrics_filtered_and_metric, load_data_safe
ROOT = Path(__file__).parent



def test_read_one(name):
    p = ROOT / f"{name}.parquet"
    st.write(f"🔎 Reading {p.name} ...")
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        st.success(f"OK: {name}  shape={df.shape}")
        return df
    except Exception as e:
        import traceback
        st.error(f"💥 Failed reading {name}.parquet")
        st.code("".join(traceback.format_exception(e)), language="python")
        st.stop()

dfs = {}
for name in ["metrics", "leaderboards", "preds", "fi"]:
    dfs[name] = test_read_one(name)

st.success("✅ All parquet files loaded")

