import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
import altair as alt


def select_top2_seeds_scale(df, metric):
    s = df[['Seed', metric]].dropna()
    g = s.groupby('Seed')[metric]
    means = g.mean()
    stds = g.std(ddof=1).replace(0, np.nan)  
    mu = s[metric].mean()
    score = (means.sub(mu).abs() / stds).fillna(np.inf)
    stats = score.rename('score').reset_index().sort_values('score').reset_index(drop=True)
    return stats, stats.head(2)['Seed'].tolist()



def select_topN_models_scale(df, metric, k, gamma=0.5, eta=0.25, min_n=2, eps=1e-12):
    s = df[['ModelFamily', metric]].dropna()
    g = s.groupby('ModelFamily')[metric]

    means = g.mean()
    stds  = g.std(ddof=1)
    ns    = g.size()

    denom = stds.replace(0, np.nan).pow(gamma) + eta
    denom = denom.replace([0, np.nan], eps)       

    score = means / denom

    stats = (pd.DataFrame({
                'ModelFamily': means.index,
                'n': ns.values,
                'mean': means.values,
                'std': stds.values,
                'score': score.values
            })
            .sort_values('score', ascending=False)
            .reset_index(drop=True))

    return stats, stats.head(k)['ModelFamily'].tolist()

def select_topK_model_configs_scale(df, metric, k, gamma=0.01, eta=1, eps=1e-12):
    s = df[['ModelName', metric]].dropna()
    g = s.groupby('ModelName')[metric]

    means = g.mean()
    stds  = g.std(ddof=1)
    ns    = g.size()

    denom = stds.replace(0, np.nan).pow(gamma) + eta
    denom = denom.replace([0, np.nan], eps)        

    score = means / denom

    stats = (pd.DataFrame({
                'ModelName': means.index,
                'n': ns.values,
                'mean': means.values,
                'std': stds.values,
                'score': score.values
            })
            .sort_values('score', ascending=False)
            .reset_index(drop=True))

    return stats, stats.head(k)['ModelName'].tolist()


def load_data():
    datasets = {}
    files = {"metrics": "metrics", "preds": "preds", "fi": "fi", "leaderboards":"leaderboards"}
    for name, base in files.items():
        pq_path  = Path(f"{base}.parquet")
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
        else:
            st.error(f"No encontrado {pq_path}")
            df = pd.DataFrame()
        datasets[name] = df
    return datasets


def collect_filter_values(dfs):
    def union(col):
        vals = []
        for df in dfs.values():
            if col in df.columns:
                vals.extend(df[col].dropna().unique())
        if not vals: return []
        return sorted(np.unique(vals).tolist())
    return {"seed": union("seed"), "nF": union("nF"), "nV": union("nV")}



def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros globales seed/nF/nV a cualquier DF que tenga esas columnas."""
    out = df
    seeds = st.session_state.get("seed_sel", [])
    nfs   = st.session_state.get("nF_sel", [])
    nvs   = st.session_state.get("nV_sel", [])

    if "seed" in out.columns and seeds:
        out = out[out["seed"].isin(seeds)]
    if "nF" in out.columns and nfs:
        out = out[out["nF"].isin(nfs)]
    if "nV" in out.columns and nvs:
        out = out[out["nV"].isin(nvs)]
    return out


def get_metrics_filtered_and_metric(dfs: dict) -> tuple[pd.DataFrame, str]:
    """
    Devuelve (metrics_filtrado, metric_col) donde:
      - metrics_filtrado: aplica filtros globales seed/nF/nV
      - metric_col: métrica global elegida en session_state['metric_sel']
    Lanza aviso si la métrica no existe.
    """
    metric_col = st.session_state.get("metric_sel")
    m = dfs.get("metrics", pd.DataFrame()).copy()

    if m.empty:
        st.warning("El dataset 'metrics' está vacío o no se cargó.")
        return m, metric_col

    m = apply_filters(m)

    if metric_col not in m.columns:
        st.warning(f"La métrica seleccionada `{metric_col}` no existe en 'metrics'.")
    return m, metric_col


def _find_col(df: pd.DataFrame, base: str):
    """Devuelve el nombre real de la columna si existe con variaciones ('seed','Seed', etc.)."""
    candidates = [base, base.lower(), base.upper(), base.capitalize()]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def apply_top2_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el filtro global 'top2_seeds' si está presente en session_state.
    No hace nada si la columna seed/Seed no existe o si no hay top2 definidos.
    """
    top2 = st.session_state.get("top2_seeds")
    if df is None or df.empty or not top2:
        return df
    seed_col = _find_col(df, "seed")
    if seed_col:
        return df[df[seed_col].isin(top2)]
    return df

def apply_topN_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el filtro global 'topN_models' si está presente en session_state.
    No hace nada si la columna ModelFamily no existe o si no hay topN definidos.
    """
    topN = st.session_state.get("topN_models")
    if df is None or df.empty or not topN:
        return df
    model_col = _find_col(df, "ModelFamily")
    if model_col:
        return df[df[model_col].isin(topN)]
    return df

def apply_topK_configs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el filtro global 'topK_configs' si está presente en session_state.
    No hace nada si la columna ModelName no existe o si no hay topN definidos.
    """
    topK = st.session_state.get("topK_configs")
    if df is None or df.empty or not topK:
        return df
    config_col = _find_col(df, "ModelName") or _find_col(df, "model")

    if config_col:
        return df[df[config_col].isin(topK)]
    return df