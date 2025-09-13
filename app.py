# app.py
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"


import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
import altair as alt
import os
import traceback
from FuncionesAuxiliares import select_top2_seeds_scale, select_topK_model_configs_scale, select_topN_models_scale, collect_filter_values
from FuncionesAuxiliares import apply_filters, apply_top2_seeds, apply_topK_configs, apply_topN_models, get_metrics_filtered_and_metric, load_data_safe


try:
    st.set_page_config(page_title="ML Logs — Dashboard", layout="wide")
    st.title("Visualización de datos - Machine learning")


    METRIC_OPTIONS = [
        'Precision_macro', 'Precision_micro', 'Precision_weighted',
        'Recall_macro', 'Recall_micro', 'Recall_weighted',
        'F1_weighted', 'F1_macro', 'Precision_0', 'Recall_0',
        'Precision_1', 'Recall_1', 'Roc_auc', 'PR_auc'
    ]
    DEFAULT_METRIC = 'Roc_auc'


    ROOT = Path(__file__).parent
    st.title("🔍 Debugging startup")

    # Show current working dir and files available
    st.write("**Working directory:**", os.getcwd())
    st.write("**Files in repo root:**", [p.name for p in Path(".").iterdir()])
    dfs = load_data_safe(ROOT)


    # ----------------------------
    # Filtros
    # ----------------------------


    all_vals = collect_filter_values(dfs)

    # Estado global
    if "seed_sel" not in st.session_state: st.session_state["seed_sel"] = all_vals["seed"]
    if "nF_sel"  not in st.session_state: st.session_state["nF_sel"]  = all_vals["nF"]
    if "nV_sel"  not in st.session_state: st.session_state["nV_sel"]  = all_vals["nV"]
    if "metric_sel" not in st.session_state: st.session_state["metric_sel"] = DEFAULT_METRIC

    st.sidebar.header("Filtros")
    st.sidebar.multiselect("Semilla", options=all_vals["seed"], key="seed_sel")
    st.sidebar.multiselect("Número de folds", options=all_vals["nF"], key="nF_sel")
    st.sidebar.multiselect("Número de variables", options=all_vals["nV"], key="nV_sel")
    st.sidebar.selectbox("Métrica", options=METRIC_OPTIONS, key="metric_sel")



    with st.expander("Inspeccionar conjuntos de datos"):
        cols = st.columns(3)
        for i, (name, df) in enumerate(dfs.items()):
            with cols[i % 3]:
                st.caption(f"**{name}**")
                st.dataframe(apply_filters(df).head(10), use_container_width=True)

    # ----------------------------
    # Tabs por objetivo
    # ----------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["SO1", "SO2", "SO3", "SO4/1", "SO4/2", "SO5", "SO6"]
    )

    # ----------------------------
    # Tab1
    # ----------------------------
    with tab1:
        st.header("SO1 — Selección de las mejores semillas")
        m, metric_col = get_metrics_filtered_and_metric(dfs)
        seed_stats, top2_seeds = select_top2_seeds_scale(m, metric_col)
        st.session_state["top2_seeds"] = top2_seeds

        metric_sel = st.session_state.get("metric_sel")
        if metric_sel not in METRIC_OPTIONS:
            metric_sel = METRIC_OPTIONS[0]  


        base_fold = (
            alt.Chart(m)  
            .transform_fold(METRIC_OPTIONS, as_=['metric','value'])
        )

        base_filt = base_fold.transform_filter(alt.datum.metric == metric_sel)

        base_ord = base_filt.transform_joinaggregate(
            metric_median='median(value)', groupby=['Seed']
        )

        x_enc = alt.X('Seed:N',
                    sort=alt.SortField(field='metric_median', order='descending'),
                    title='Semilla')
        y_enc = alt.Y('value:Q', title='Valor', scale=alt.Scale(zero=False))

        box = base_ord.mark_boxplot(size=30).encode(
            x=x_enc, y=y_enc, color=alt.Color('Seed:N', legend=None),
            tooltip=['metric:N','Seed:N','metric_median:Q']
        ).properties(width=max(480, 40*9), height=260)

        points = base_ord.mark_circle(size=60, opacity=0.35).encode(
            x=x_enc, y=y_enc, color='Seed:N',
            tooltip=['metric:N','Seed:N','value:Q']
        )

        rule = base_filt.transform_aggregate(mean='mean(value)').mark_rule().encode(y='mean:Q')

        seed_means_dot = (
            base_filt
            .transform_aggregate(seed_mean='mean(value)', groupby=['Seed'])
            .mark_point(size=50, color='black')
            .encode(
                x=x_enc,
                y=alt.Y('seed_mean:Q', title='Valor'),
                tooltip=['metric:N', 'Seed:N', 'seed_mean:Q']
            )
        )

        chart = alt.layer(points, box, rule, seed_means_dot).resolve_scale(color='independent')
        chart



        st.write(f"Semillas candidatas según `{metric_col}` (con filtros nF/nV aplicados):`{', '.join(map(str, top2_seeds))}`")

    # ----------------------------
    # Tab2
    # ----------------------------
    with tab2:
        st.header("SO2 — Selección de los mejores modelos")

        m=apply_top2_seeds(m)

        st.markdown(f"""
        **Estado de filtros activos:**
        - **Semillas:** {', '.join(map(str, top2_seeds))}
        """)

        

        topN = st.number_input(
            "Cantidad de modelos a mostrar:",
            min_value=1,
            max_value=50,
            value=10, 
            step=1
        )

        cols = ['ModelFamily']
        if 'ModelName' in m.columns:
            cols.append('ModelName')

        dfm = (m[cols + [metric_col]]
            .rename(columns={metric_col: 'value'})
            .dropna(subset=['value']))

        family_rank = (dfm.groupby('ModelFamily', as_index=False)
                        .agg(family_center=('value', 'median')))

        order = (family_rank.sort_values('family_center', ascending=False)
                        .head(topN)['ModelFamily'].tolist())

        df_top = dfm[dfm['ModelFamily'].isin(order)].copy()


        y_enc = alt.Y(
            'value:Q',
            title=metric_col,
            scale=alt.Scale(domain=[0.9 * dfm['value'].min(), 1.0])
        )

        box = (alt.Chart(df_top)
            .mark_boxplot(extent="min-max", size=30)
            .encode(
                x=alt.X('ModelFamily:N', sort=order, title='Modelo'),
                y=y_enc,
                color=alt.Color('ModelFamily:N', legend=None)
            )
            .properties(width=max(480, 40*len(order)), height=360))

        points_tooltip = ['ModelFamily:N']
        if 'ModelName' in df_top.columns:
            points_tooltip.append('ModelName:N')
        points_tooltip.append(alt.Tooltip('value:Q', title='value', format='.4f'))

        points = (alt.Chart(df_top)
                .mark_circle(size=60, opacity=0.25)
                .encode(
                    x=alt.X('ModelFamily:N', sort=order, title=''),
                    y=y_enc,
                    color=alt.Color('ModelFamily:N', legend=None),
                    tooltip=points_tooltip
                ))

        means = (alt.Chart(df_top)
                .transform_aggregate(family_mean='mean(value)', groupby=['ModelFamily'])
                .mark_point(color='black', size=70)
                .encode(
                    x=alt.X('ModelFamily:N', sort=order, title=''),
                    y='family_mean:Q'
                ))

        chart = (points+box+ means).resolve_scale(color='independent')
        chart




        dfw = (m[['ModelFamily','ModelName', metric_col]]
            .rename(columns={metric_col: 'value'})
            .dropna(subset=['value']))

        family_stats = (
            dfw.groupby('ModelFamily', as_index=False)
            .agg(
                family_mean=('value', 'mean'),     
                family_std =('value', 'std'),      
                n_configs  =('ModelName', 'nunique'),
                n_rows     =('value', 'size')
            )
        )

        family_stats['family_std'] = family_stats['family_std'].fillna(0.0)

        order_scatter = (family_stats.sort_values('family_mean', ascending=False)
                                    .head(topN)['ModelFamily'].tolist())
        fst = family_stats[family_stats['ModelFamily'].isin(order_scatter)].copy()

        x_enc = alt.X(
            'family_mean:Q',
            title=f'{metric_col} medio',
            scale=alt.Scale(domain=[0.5, 1.0])
        )

        scatter = (
            alt.Chart(fst)
            .mark_circle(size=140)
            .encode(
                x=x_enc,
                y=alt.Y('family_std:Q', title='Desviación estándar'),
                color=alt.Color('ModelFamily:N', title='Modelo'),
                size=alt.Size('n_configs:Q', title='nº configs',
                                legend=alt.Legend(values=sorted(fst['n_configs'].unique()))),
                tooltip=[
                    'ModelFamily:N',
                    alt.Tooltip('family_mean:Q', title='media', format='.4f'),
                    alt.Tooltip('family_std:Q',  title='std',   format='.4f'),
                    alt.Tooltip('n_configs:Q',   title='nº configs'),
                    alt.Tooltip('n_rows:Q',      title='nº filas')
                ]
            )
            .properties(width=520, height=420)
        )
        scatter

        _, topN_models = select_topN_models_scale(m, metric_col, topN)
        st.session_state["topN_models"] = topN_models


    # ----------------------------
    # Tab3
    # ----------------------------
    with tab3:
        st.header("SO3 — Selección de mejores configuraciones")
        m=apply_topN_models(m)
        st.markdown(f"""
        **Estado de filtros activos:**
        - **Semillas:** {', '.join(map(str, top2_seeds))}
        - **Modelos:** {', '.join(map(str, topN_models))}
        """)

        
        topK = st.number_input(
            "Cantidad de configuraciones por modelo:",
            min_value=1,
            max_value=50,
            value=5, 
            step=1
        )

        pivot_df = (
        m.groupby("ModelName", as_index=False)[metric_col]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "mean", "std": "std"})
        )
        families = m[["ModelName", "ModelFamily"]].drop_duplicates()
        pivot_df = pivot_df.merge(families, on="ModelName", how="left")

        cfg=pivot_df.copy()
        cfg = (cfg.sort_values(['ModelFamily','mean','std'], ascending=[True, False, True])
            .assign(rank_cfg=lambda d: d.groupby('ModelFamily').cumcount() + 1))

        default_family = topN_models[0] if len(topN_models) else None
        family_param = alt.param(
            name='family_sel',
            value=default_family,
            bind=alt.binding_select(options=topN_models, name='Modelo: ')
        )

        x_enc = alt.X(
            'mean:Q',
            title=f'{metric_col} medio',
            scale=alt.Scale(domain=[0.5, 1.0])
        )
        y_enc = alt.Y('std:Q', title='Desviación estandar')

        base = alt.Chart(cfg).add_params(family_param).transform_filter(
            alt.datum.ModelFamily == family_param
        ).properties(width=620, height=380, title='Configuraciones dentro del modelo seleccionado')

        others = base.transform_filter(
            alt.datum.rank_cfg > topK
        ).mark_circle(size=80, opacity=0.4, color='lightgrey').encode(
            x=x_enc, y=y_enc,
            tooltip=['ModelFamily:N','ModelName:N',
                    alt.Tooltip('mean:Q', format='.4f'),
                    alt.Tooltip('std:Q',  format='.4f'),
                    alt.Tooltip('rank_cfg:Q', title='rank')]
        )

        highlight = base.transform_filter(
            alt.datum.rank_cfg <= topK
        ).mark_circle(size=140).encode(
            x=x_enc, y=y_enc,
            color=alt.Color('ModelName:N', title='Mejores configuraciones'),
            tooltip=['ModelFamily:N','ModelName:N',
                    alt.Tooltip('mean:Q', format='.4f'),
                    alt.Tooltip('std:Q',  format='.4f'),
                    alt.Tooltip('rank_cfg:Q', title='rank')]
        )

        chart = (others + highlight).resolve_scale(color='independent')
        chart
        _, topK_configs = select_topK_model_configs_scale(m, metric_col, topK)
        st.session_state["topK_configs"] = topK_configs

        st.markdown(f"**Las mejores configuraciones globalmente son:**`'{', '.join(map(str, topK_configs))}`")


    # ----------------------------
    # Tab4
    # ----------------------------

    with tab4:
        st.header("SO4/1 — Ranking de mejores variables en mejores modelos")
        st.markdown(f"""
        **Estado de filtros activos:**
        - **Semillas:** {', '.join(map(str, top2_seeds))}
        - **Modelos:** {', '.join(map(str, topN_models))}
        """)

        fi=apply_topN_models(apply_top2_seeds(apply_filters(dfs["fi"])))

        topF = 5  # nº de features a mostrar en el heatmap

        agg = (
            fi.groupby(['ModelFamily','feature'], as_index=False)
            .agg(
                median_importance=('importance', 'median'),
                iqr=('importance', lambda x: x.quantile(0.75) - x.quantile(0.25)),
                n=('importance', 'size')
            )
        )

        family_order = topN_models

        feature_rank = (agg.groupby('feature')['median_importance']
                        .mean().sort_values(ascending=False))
        top_features = feature_rank.head(topF).index.tolist()

        agg_top = agg[agg['feature'].isin(top_features)].copy()

        eps = 1e-12  

        feature_max = agg_top.groupby('feature')['median_importance'].max()
        valid_features = feature_max[feature_max.abs() > eps].index
        agg_top_nz = agg_top[agg_top['feature'].isin(valid_features)].copy()


        top_features_nz = (agg_top_nz.groupby('feature')['median_importance']
                                    .mean()
                                    .sort_values(ascending=True)
                                    .index.tolist())

        max_val = float(agg_top_nz['median_importance'].max())


        y_enc = alt.X('ModelFamily:N',   sort=family_order,       title='Modelo')
        x_enc = alt.Y('feature:N', sort=top_features_nz[::-1], title='Característica')

        base_hm = (
                alt.Chart(agg_top_nz)
                .transform_calculate(
                    is_high=f"datum.median_importance / {max_val if max_val>0 else 1} >= 0.6"
                )
            )
        heatmap = (
                base_hm
                .mark_rect(stroke='rgba(0,0,0,0.12)', strokeWidth=0.5)
                .encode(
                    x=x_enc,
                    y=y_enc,
                    color=alt.Color(
                        'median_importance:Q',
                        title='Importancia (mediana)',
                        scale=alt.Scale(
                            domain=[0, max_val],
                            range=['#FFFFFF', '#D7301F']  
                        ),
                        legend=alt.Legend(format='.3f')
                    ),
                    tooltip=[
                        'ModelFamily:N','feature:N',
                        alt.Tooltip('median_importance:Q', title='Median', format='.4f'),
                        alt.Tooltip('iqr:Q',              title='IQR',    format='.4f'),
                        alt.Tooltip('n:Q',                title='n rows')
                    ]
                )
                .properties(
                    width=max(520, 40*len(family_order)),
                    height=max(320, 16*len(top_features_nz))
                )
            )

        labels = (
                base_hm
                .mark_text(baseline='middle', fontSize=10, fontWeight=600)
                .encode(
                    x=x_enc,
                    y=y_enc,
                    text=alt.Text('median_importance:Q', format='.3f'),
                    color=alt.condition(
                        alt.datum.is_high,
                        alt.value('white'),   # sobre rojo intenso
                        alt.value('#222')     # sobre claro
                    )
                )
            )

        heatmap_chart = (heatmap + labels).resolve_scale(color='independent')
        heatmap_chart


        raw_top = fi[fi['feature'].isin(top_features)].copy()
        raw_top = fi[fi['feature'].isin(top_features)].copy()
        raw_top['importance'] = pd.to_numeric(raw_top['importance'], errors='coerce')
        raw_top = raw_top.dropna(subset=['importance'])


        family_options = sorted(raw_top['ModelFamily'].unique().tolist())
        
        
        
        if not family_options:
            st.info("No families available with current filters.")

        fam_param = alt.param(
            name='fam',
            bind=alt.binding_select(options=family_options, name='Family: '),
            value=family_options[0]
        )

        feat_order = top_features

        base = (
            alt.Chart(raw_top)
            .add_params(fam_param)
            .transform_calculate(
                famSel=f"indexof({family_options!r}, fam) >= 0 ? fam : {family_options[0]!r}"
            )
            .transform_filter("datum.ModelFamily == datum.famSel")
            .encode(
                x=alt.X('feature:N', sort=feat_order, title='Característica'),
                y=alt.Y('importance:Q', title='Importancia'),
                color=alt.Color('feature:N', legend=None)
            )
            .transform_joinaggregate(n='count()', groupby=['feature'])
        )

        box = (
            base
            .mark_boxplot()
            .encode(
                tooltip=[
                    alt.Tooltip('feature:N'),
                    alt.Tooltip('n:Q', title='n')
                ]
            )
            .properties(width=max(520, 20*len(feat_order)), height=260)
            .resolve_scale(color='independent')
        )

        st.altair_chart(box, use_container_width=True)


    # ----------------------------
    # Tab5
    # ----------------------------
    with tab5:
        st.header("SO4/2 — Ranking de mejores variables en mejores configuraciones")
        st.markdown(f"""
        **Estado de filtros activos:**
        - **Semillas:** {', '.join(map(str, top2_seeds))}
        - **Modelos:** {', '.join(map(str, topN_models))}
        - **Configuraciones:** {', '.join(map(str, topK_configs))}
        """)

        fi=apply_topK_configs(apply_top2_seeds(apply_filters(dfs["fi"])))
        topF = 5  

        agg = (
            fi.groupby(['model','feature'], as_index=False)
            .agg(
                median_importance=('importance', 'median'),
                iqr=('importance', lambda x: x.quantile(0.75) - x.quantile(0.25)),
                n=('importance', 'size')
            )
        )
        family_order = topK_configs

        feature_rank = (agg.groupby('feature')['median_importance']
                        .mean().sort_values(ascending=False))
        top_features = feature_rank.head(topF).index.tolist()

        agg_top = agg[agg['feature'].isin(top_features)].copy()

        eps = 1e-12  

        feature_max = agg_top.groupby('feature')['median_importance'].max()
        valid_features = feature_max[feature_max.abs() > eps].index
        agg_top_nz = agg_top[agg_top['feature'].isin(valid_features)].copy()

        top_features_nz = (agg_top_nz.groupby('feature')['median_importance']
                                    .mean()
                                    .sort_values(ascending=True)
                                    .index.tolist())

        max_val = float(agg_top_nz['median_importance'].max())

        y_enc = alt.X('model:N',   sort=family_order,       title='Modelo')
        x_enc = alt.Y('feature:N', sort=top_features_nz[::-1], title='Característica')

        base_hm = (
            alt.Chart(agg_top_nz)
            .transform_calculate(
                is_high=f"datum.median_importance / {max_val if max_val>0 else 1} >= 0.6"
            )
        )
        heatmap = (
            base_hm
            .mark_rect(stroke='rgba(0,0,0,0.12)', strokeWidth=0.5)
            .encode(
                x=x_enc,
                y=y_enc,
                color=alt.Color(
                    'median_importance:Q',
                    title='Importancia (mediana)',
                    scale=alt.Scale(
                        domain=[0, max_val],
                        range=['#FFFFFF', '#D7301F']  
                    ),
                    legend=alt.Legend(format='.3f')
                ),
                tooltip=[
                    'model:N','feature:N',
                    alt.Tooltip('median_importance:Q', title='Median', format='.4f'),
                    alt.Tooltip('iqr:Q',              title='IQR',    format='.4f'),
                    alt.Tooltip('n:Q',                title='n rows')
                ]
            )
            .properties(
                width=max(520, 40*len(family_order)),
                height=max(320, 16*len(top_features_nz))
            )
        )

        labels = (
            base_hm
            .mark_text(baseline='middle', fontSize=10, fontWeight=600)
            .encode(
                x=x_enc,
                y=y_enc,
                text=alt.Text('median_importance:Q', format='.3f'),
                color=alt.condition(
                    alt.datum.is_high,
                    alt.value('white'),   
                    alt.value('#222')     
                )
            )
        )

        heatmap_chart = (heatmap + labels).resolve_scale(color='independent')
        heatmap_chart

        raw_top = fi[fi['feature'].isin(top_features)].copy()
        raw_top = fi[fi['feature'].isin(top_features)].copy()
        raw_top['importance'] = pd.to_numeric(raw_top['importance'], errors='coerce')
        raw_top = raw_top.dropna(subset=['importance'])

        family_options = sorted(raw_top['model'].unique().tolist())
        
        
        if not family_options:
            st.info("No families available with current filters.")

        fam_param = alt.param(
            name='fam',
            bind=alt.binding_select(options=family_options, name='Configuración: '),
            value=family_options[0]
        )

        feat_order = top_features

        base = (
            alt.Chart(raw_top)
            .add_params(fam_param)
            .transform_calculate(
                famSel=f"indexof({family_options!r}, fam) >= 0 ? fam : {family_options[0]!r}"
            )
            .transform_filter("datum.model == datum.famSel")
            .encode(
                x=alt.X('feature:N', sort=feat_order, title='Característica'),
                y=alt.Y('importance:Q', title='Importancia'),
                color=alt.Color('feature:N', legend=None)
            )
            .transform_joinaggregate(n='count()', groupby=['feature'])
        )

        box = (
            base
            .mark_boxplot()
            .encode(
                tooltip=[
                    alt.Tooltip('feature:N'),
                    alt.Tooltip('n:Q', title='n')
                ]
            )
            .properties(width=max(520, 20*len(feat_order)), height=260)
            .resolve_scale(color='independent')
        )

        st.altair_chart(box, use_container_width=True)


    # ----------------------------
    # Tab6
    # ----------------------------

    with tab6:
        st.header("SO5 — Análisis de Métrica por folds")

        st.markdown(f"""
        **Estado de filtros activos:**
        - **Semillas:** {', '.join(map(str, top2_seeds))}
        - **Configuraciones:** {', '.join(map(str, topK_configs))}
        """)

        data,metric_col=get_metrics_filtered_and_metric(dfs)


        model_lb_col   = 'model'
        model_mx_col   = 'ModelName'
        metric_lb_col  = metric_col.lower()  
        metric_mx_col  = metric_col

        lb=apply_topK_configs(apply_top2_seeds(apply_filters(dfs["leaderboards"])))
        m=apply_topK_configs(apply_top2_seeds(data))
        if metric_lb_col not in lb.columns:
            st.warning(f"Visualization not available for this metric: {metric_col}")
        else:

            base_all = (
                lb[['nF','seed','fold', model_lb_col, metric_lb_col]]
                .merge(
                    m[['nF','seed', model_mx_col, metric_mx_col]]
                    .rename(columns={metric_mx_col:'auc_oof_global'}),
                    left_on=['nF','seed', model_lb_col],
                    right_on=['nF','seed', model_mx_col],
                    how='left'
                )
            )

            model_param = alt.param(
                name='Modelo',
                bind=alt.binding_select(options=list(topK_configs), name='Modelo: '),
                value=topK_configs[0]
            )

            filtered = (
                alt.Chart(base_all)
                .add_params(model_param)
                .transform_filter(alt.datum[model_lb_col] == model_param)
            )


            pts = (
                filtered
                .mark_circle(size=70, opacity=0.9)
                .encode(
                    x=alt.X('fold:O', title='Fold'),
                    y=alt.Y(f'{metric_lb_col}:Q', title='ROC AUC (fold)'),
                    color=alt.Color('seed:N', title='Semilla'),
                    tooltip=[
                        alt.Tooltip('nF:N', title='nF'),
                        alt.Tooltip('seed:N', title='Semilla'),
                        alt.Tooltip('fold:N', title='Fold'),
                        alt.Tooltip(f'{metric_lb_col}:Q', title='AUC fold', format='.3f'),
                        alt.Tooltip('auc_oof_global:Q', title='AUC OOF', format='.3f')
                    ]
                )
            )

            rules = (
                filtered
                .transform_aggregate(
                    auc_oof_global='mean(auc_oof_global)',
                    groupby=['nF']
                )
                .mark_rule(strokeDash=[6,3])
                .encode(
                    y=alt.Y('auc_oof_global:Q', title='')
                    )
            )

            chart1 = alt.layer(pts, rules).properties(width=350, height=250).facet(
                column=alt.Column('nF:O', title='nF', sort='ascending')
            ).properties(
                title=f'{metric_col} por capa (puntos) vs OOF (línea)'
            )


            base2 = filtered 

            pts_db = (
                base2
                .transform_aggregate(
                    auc_mean=f"mean({metric_lb_col})",
                    groupby=['nF','seed','fold']
                )
                .mark_point(size=100, filled=True, opacity=0.95)
                .encode(
                    y=alt.Y('fold:O', title='Fold', sort='ascending', axis=alt.Axis(labelPadding=8)),
                    x=alt.X('auc_mean:Q', title=f'{metric_mx_col} medio (fold, seed)', scale=alt.Scale(domain=[0.5, 1.0])),
                    color=alt.Color('seed:N', title='Semilla'),
                    shape=alt.Shape('seed:N'),
                    tooltip=[
                        alt.Tooltip('nF:N', title='nF'),
                        alt.Tooltip('seed:N', title='Semilla'),
                        alt.Tooltip('fold:N', title='Fold'),
                        alt.Tooltip('auc_mean:Q', title=f'{metric_mx_col} medio', format='.3f')
                    ]
                )
            )

            connectors = (
                base2
                .transform_aggregate(
                    auc_mean=f"mean({metric_lb_col})",
                    groupby=['nF','seed','fold']
                )
                .transform_aggregate(
                    auc_min='min(auc_mean)',
                    auc_max='max(auc_mean)',
                    groupby=['nF','fold']
                )
                .mark_rule(strokeWidth=2, opacity=0.7)
                .encode(
                    y=alt.Y('fold:O', sort='ascending'),
                    x=alt.X('auc_min:Q', scale=alt.Scale(domain=[0.5, 1.0])),
                    x2='auc_max:Q'
                )
            )

            rule_oof = (
                base2
                .transform_aggregate(
                    auc_oof_global='mean(auc_oof_global)',
                    groupby=['nF']
                )
                .mark_rule(strokeDash=[6,3], strokeWidth=2)
                .encode(
                    x=alt.X('auc_oof_global:Q', title=f'{metric_mx_col} OOF', scale=alt.Scale(domain=[0.5, 1.0])),
                    tooltip=[
                        alt.Tooltip('nF:N', title='nF'),
                        alt.Tooltip('auc_oof_global:Q', title='AUC OOF', format='.3f')
                    ]
                )
            )

            chart_dumbbell = alt.layer(connectors, pts_db, rule_oof).properties(
                width=350, height=250
            ).facet(
                column=alt.Column('nF:O', title='nF', sort='ascending')
            ).properties(
                title=f'{metric_mx_col} medio por semilla vs OOF'
            )

            chart1
            chart_dumbbell

    # ----------------------------
    # Tab7
    # ----------------------------


    with tab7:
        st.header("SO6 — Análisis de errores OOF")

        st.markdown(f"""
        **Estado de filtros activos:**
        - **Semillas:** {', '.join(map(str, top2_seeds))}
        - **Configuraciones:** {', '.join(map(str, topK_configs))}
        """)
        

        preds=dfs["preds"]
        preds=apply_topK_configs(apply_top2_seeds(apply_filters(preds)))


        df = preds.loc[ preds['proba'].notna()].copy()
        df['ED_2Clases'] = df['ED_2Clases'].astype(int)
        df['pred_label'] = (df['proba'].astype(float) >= 0.5).astype(int)
        df['wrong']      = (df['pred_label'] != df['ED_2Clases']).astype(int)

        counts = (df.groupby(['etiq_id','model'], as_index=False)
                    .agg(n_pred=('wrong','size'), n_wrong=('wrong','sum')))
        counts['n_right']    = counts['n_pred'] - counts['n_wrong']
        counts['prop_wrong'] = counts['n_wrong'] / counts['n_pred']

        totals = (counts.groupby('etiq_id', as_index=False)
                        .agg(total_wrong=('n_wrong','sum'), total_pred=('n_pred','sum')))
        
        top_ids = (totals.sort_values(['total_wrong','total_pred'], ascending=[False, False])
                            .head(15)['etiq_id'].tolist())

        hm = counts[counts['etiq_id'].isin(top_ids)].copy()

        hm['model']   = pd.Categorical(hm['model'],   categories=topK_configs, ordered=True)
        hm['etiq_id'] = pd.Categorical(hm['etiq_id'], categories=top_ids,   ordered=True)

        totals_top = totals[totals['etiq_id'].isin(top_ids)].copy()
        totals_top['etiq_id'] = pd.Categorical(totals_top['etiq_id'],
                                            categories=top_ids, ordered=True)

        label_thresh = 0.6  
        max_prop = float(max(1.0, hm['prop_wrong'].max()))  
        min_prop = float(max(0, hm['prop_wrong'].min()))  

        y_enc = alt.Y(
        'etiq_id:N',
        title=None,
        scale=alt.Scale(domain=top_ids),
        sort=None,
        axis=alt.Axis(labelAlign='left', labelPadding=130, labelLimit=0)
    )

        x_models = alt.X('model:N', title='Modelo',
                        scale=alt.Scale(domain=topK_configs), sort=None)

        bars_left = (
        alt.Chart(totals_top)
        .mark_bar(color="#C7AB31")   
        .encode(
            y=y_enc,
            x=alt.X('total_wrong:Q', title='Errores totales'),
            tooltip=[
                    alt.Tooltip('etiq_id:N', title='Instance'),
                    alt.Tooltip('total_wrong:Q', title='Mistakes'),
                    alt.Tooltip('total_pred:Q',  title='Preds')
            ]
        )
        .properties(width=260, height=26*len(top_ids))
        )



        bars_labels = (
            alt.Chart(totals_top)
            .mark_text(align='left', dx=3, baseline='middle')
            .encode(
                y=y_enc,
                x=alt.X('total_wrong:Q'),
                text=alt.Text('total_wrong:Q'),
            )
        )

        base_hm = (
            alt.Chart(hm)
            .transform_calculate(is_high=f"datum.prop_wrong / {max_prop} >= {label_thresh}")
        )

        heatmap = (
            base_hm
            .mark_rect(stroke='rgba(0,0,0,0.12)', strokeWidth=0.5)
            .encode(
                x=x_models,
                y=y_enc,
                color=alt.Color('prop_wrong:Q',
                                title='Proporción de fallos',
                                scale=alt.Scale(domain=[min_prop, 1], range=['#FFFFFF', '#B30000'], clamp=True),
                                legend=alt.Legend(format='.2f')),
                tooltip=[
                    alt.Tooltip('etiq_id:N', title='Instance'),
                    alt.Tooltip('model:N',   title='Model'),
                    alt.Tooltip('n_wrong:Q', title='Mistakes'),
                    alt.Tooltip('n_right:Q', title='Correct'),
                    alt.Tooltip('n_pred:Q',  title='Preds'),
                    alt.Tooltip('prop_wrong:Q', title='Error %', format='.2f')
                ]
            )
            .properties(width=48*len(topK_configs), height=26*len(top_ids))
        )


        labels = (
            base_hm
            .mark_text(baseline='middle', fontSize=10)
            .encode(
                x=x_models,
                y=y_enc,
                text=alt.Text('prop_wrong:Q', format='.2f'),
                color=alt.condition('datum.is_high', alt.value('white'), alt.value('#222'))
            )
        )

        barsandheatmap = alt.hconcat(
            (bars_left + bars_labels).properties(title='Errores totales'),
            (heatmap + labels).properties(title='Proporción de errores por modelo')
        ).resolve_scale(y='shared').properties(
        title='Instancias difíciles en los mejores modelos'
        )

        barsandheatmap

except Exception as e:
    st.error("💥 The app crashed with an exception:")
    st.code("".join(traceback.format_exception(e)), language="python")







