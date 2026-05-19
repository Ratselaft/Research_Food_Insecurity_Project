# dashboard.py
#
# PURPOSE:
#   Interactive 5-page research dashboard for the food insecurity dissertation.
#   Displays NLP findings, empirical model results, a country choropleth map,
#   and robustness checks.
#
# HOW TO RUN:
#   streamlit run dashboard.py
#
# PAGES:
#   1. Overview           — model performance + corpus summary
#   2. NLP Findings       — TF-IDF keywords, NMF topics, theme-variable map
#   3. Empirical Results  — Model F coefficients, bootstrap CIs, nested F-test
#   4. Country Map        — choropleth of cereal availability across 160 countries
#   5. Robustness Checks  — 7 specifications + Cook's Distance influential countries
#

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Insecurity Research Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "outputs", "powerbi")

# ── Helper: load a CSV with a clear error message ─────────────────────────────
def load_csv(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        st.error("Data file not found: " + file_path)
        st.stop()
    df = pd.read_csv(file_path)
    return df

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page_options = [
    "1. Overview",
    "2. NLP Findings",
    "3. Empirical Results",
    "4. Country Map",
    "5. Robustness Checks",
]

selected_page = st.sidebar.radio("Go to page:", page_options)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sheffield Hallam University**")
st.sidebar.markdown("MSc Data Science and Artificial Intelligence")
st.sidebar.markdown("Dissertation: Food Insecurity Pipeline")
st.sidebar.markdown("Year: 2026")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if selected_page == "1. Overview":

    st.title("Overview: Model Performance and Research Corpus")
    st.markdown(
        "This page summarises how well each nested model explains cross-country cereal food "
        "availability (kg/person/yr) and how the 127-paper corpus was assembled."
    )

    # ── Load data ────────────────────────────────────────────────────────────
    df_perf   = load_csv("page1_model_performance.csv")
    df_corpus = load_csv("page1_corpus_summary.csv")

    # ── Section 1: Model Performance ─────────────────────────────────────────
    st.subheader("Model Performance Comparison")
    st.markdown(
        "Five nested OLS models (A → B → C → F and A★ as the honest comparator). "
        "RF and XGB CV R² are 5-fold cross-validated."
    )

    col1, col2, col3 = st.columns(3)

    # Find Model F row for headline metrics
    f_row = None
    for i in range(len(df_perf)):
        label = str(df_perf.loc[i, "Model Label"])
        if "Model F" in label or "NLP" in label:
            f_row = df_perf.loc[i]
            break
    if f_row is None:
        f_row = df_perf.iloc[-1]

    a_row = df_perf.iloc[0]

    col1.metric("Model A  OLS R²",  str(round(float(a_row["OLS R²"]), 3)))
    col2.metric("Model F  OLS R²",  str(round(float(f_row["OLS R²"]), 3)))

    # Delta R² column may have blank for baseline
    delta_val = f_row["Delta R² vs A*"]
    if pd.notna(delta_val) and str(delta_val).strip() != "":
        col3.metric("ΔR² (F vs A★)", str(round(float(delta_val), 3)))
    else:
        col3.metric("Nested F-test", "p = 0.004 (***)")

    # Bar chart — OLS R² for each model
    model_labels = list(df_perf["Model Label"])
    ols_r2_vals  = []
    rf_r2_vals   = []
    xgb_r2_vals  = []

    for i in range(len(df_perf)):
        ols_r2_vals.append(float(df_perf.loc[i, "OLS R²"]))
        rf_r2_vals.append(float(df_perf.loc[i, "RF 5-fold CV R²"]))
        xgb_r2_vals.append(float(df_perf.loc[i, "XGB 5-fold CV R²"]))

    fig_perf = go.Figure()

    fig_perf.add_trace(go.Bar(
        name="OLS R²",
        x=model_labels,
        y=ols_r2_vals,
        marker_color="#1f77b4",
    ))

    fig_perf.add_trace(go.Bar(
        name="RF CV R²",
        x=model_labels,
        y=rf_r2_vals,
        marker_color="#2ca02c",
    ))

    fig_perf.add_trace(go.Bar(
        name="XGB CV R²",
        x=model_labels,
        y=xgb_r2_vals,
        marker_color="#ff7f0e",
    ))

    fig_perf.update_layout(
        barmode="group",
        title="R² by Model and Estimator",
        xaxis_title="Model",
        yaxis_title="R²",
        legend_title="Estimator",
        height=420,
        xaxis_tickangle=-20,
    )

    st.plotly_chart(fig_perf, use_container_width=True)

    # Expandable detail table
    with st.expander("Show full model performance table"):
        show_cols = ["Model Label", "N (countries)", "Predictors used",
                     "OLS R²", "OLS Adj R²", "OLS F-stat p",
                     "RF 5-fold CV R²", "XGB 5-fold CV R²", "Delta R² vs A*"]
        display_cols = []
        for col in show_cols:
            if col in df_perf.columns:
                display_cols.append(col)
        st.dataframe(df_perf[display_cols], use_container_width=True)

    st.divider()

    # ── Section 2: Corpus Summary ─────────────────────────────────────────────
    st.subheader("Research Corpus Summary")
    st.markdown(
        "Papers were retrieved via OpenAlex and Scopus, then scored for alignment "
        "with the food insecurity theme. Only 'Strict' alignment papers feed the NLP analysis."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_corpus = px.bar(
            df_corpus,
            x="Alignment Level",
            y="Paper Count",
            color="Source",
            barmode="group",
            title="Papers by Alignment Level and Source",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_corpus.update_layout(height=380, xaxis_tickangle=-10)
        st.plotly_chart(fig_corpus, use_container_width=True)

    with col_b:
        # Aggregate totals for a pie chart
        totals_by_level = df_corpus.groupby("Alignment Level")["Paper Count"].sum().reset_index()
        fig_pie = px.pie(
            totals_by_level,
            names="Alignment Level",
            values="Paper Count",
            title="Share of Papers by Alignment Level",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("Show corpus table"):
        st.dataframe(df_corpus, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — NLP FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "2. NLP Findings":

    st.title("NLP Findings: Topics, Keywords, and Theme-Variable Mapping")
    st.markdown(
        "TF-IDF identifies the most distinctive terms across 127 strictly aligned papers. "
        "NMF (k=7) groups those terms into interpretable topics. Each topic is then mapped "
        "to an empirical proxy variable used in the regression models."
    )

    df_tfidf    = load_csv("page2_tfidf_keywords.csv")
    df_topics   = load_csv("page2_nmf_topics.csv")
    df_map      = load_csv("page2_theme_variable_map.csv")

    # ── Section 1: Top TF-IDF Keywords ───────────────────────────────────────
    st.subheader("Top 20 TF-IDF Keywords")

    df_top20 = df_tfidf.head(20).copy()

    fig_tfidf = px.bar(
        df_top20,
        x="TF-IDF Score",
        y="Keyword",
        orientation="h",
        title="Top 20 Keywords by TF-IDF Score (127-paper corpus)",
        color="TF-IDF Score",
        color_continuous_scale="Blues",
    )
    fig_tfidf.update_layout(
        height=520,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_tfidf, use_container_width=True)

    st.divider()

    # ── Section 2: NMF Topics ─────────────────────────────────────────────────
    st.subheader("NMF Topic Model — 7 Topics")
    st.markdown(
        "Each topic represents a cluster of co-occurring terms. "
        "The label is assigned by reading the top keywords."
    )

    for i in range(len(df_topics)):
        row = df_topics.iloc[i]
        topic_label = str(row["Topic Label"])
        topic_id    = str(row["Topic ID"])
        keywords    = str(row["Top Keywords"])
        dom_papers  = str(row["Dominant Papers"])

        with st.expander("Topic " + topic_id + " — " + topic_label):
            st.markdown("**Top keywords:** " + keywords)
            st.markdown("**Dominant papers:** " + dom_papers)

    st.divider()

    # ── Section 3: Theme → Variable Mapping ──────────────────────────────────
    st.subheader("NLP Theme to Model Variable Mapping")
    st.markdown(
        "Each NLP topic is translated into one empirical proxy variable. "
        "Rows with no proxy were not included in Model F."
    )

    display_cols = []
    priority_cols = ["Topic ID", "NLP Theme", "Proxy Variable (Model F)", "Data Source", "Top Words"]
    for col in priority_cols:
        if col in df_map.columns:
            display_cols.append(col)
    for col in df_map.columns:
        if col not in display_cols:
            display_cols.append(col)

    st.dataframe(df_map[display_cols], use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EMPIRICAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "3. Empirical Results":

    st.title("Empirical Results: Model F Coefficients and Bootstrap CIs")
    st.markdown(
        "Model F adds five NLP-discovered variables to the baseline production model (A). "
        "A nested F-test confirms the NLP block adds genuine predictive power."
    )

    df_ci      = load_csv("page3_bootstrap_cis.csv")
    df_ftest   = load_csv("page3_ftest_summary.csv")
    df_synth   = load_csv("page3_nlp_synthesis.csv")

    # ── Section 1: Nested F-test Result ──────────────────────────────────────
    st.subheader("Nested F-test: Does the NLP Block Add Explanatory Power?")

    frow = df_ftest.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F statistic",  str(round(float(frow["F statistic"]), 3)))
    col2.metric("p-value",      str(round(float(frow["p-value"]), 4)))
    col3.metric("Significance", str(frow["Significance"]))
    col4.metric("Partial R²",   str(round(float(frow["Partial R²"]), 3)))

    st.markdown(
        "**Interpretation:** F(" + str(int(frow["df1 (extra vars)"])) + ", " +
        str(int(frow["df2"])) + ") = " + str(round(float(frow["F statistic"]), 3)) +
        ", p = " + str(round(float(frow["p-value"]), 4)) +
        " — the five NLP-identified variables jointly improve model fit at the 1% significance level."
    )

    st.divider()

    # ── Section 2: Bootstrap CIs ──────────────────────────────────────────────
    st.subheader("Bootstrap Confidence Intervals — Model F Predictors")
    st.markdown(
        "1,000-iteration bootstrap (RANDOM_SEED=42). "
        "Error bars show 95% CIs. Variables whose CI excludes zero are starred."
    )

    fig_ci = go.Figure()

    for i in range(len(df_ci)):
        row = df_ci.iloc[i]
        var_name   = str(row["Variable"])
        mean_val   = float(row["Bootstrap Mean"])
        ci_lower   = float(row["95% CI Lower"])
        ci_upper   = float(row["95% CI Upper"])
        ci_excl    = str(row["CI Excludes Zero"])

        if ci_excl == "Yes":
            bar_color = "#1a9641"
            label     = var_name + " *"
        else:
            bar_color = "#888888"
            label     = var_name

        error_minus = mean_val - ci_lower
        error_plus  = ci_upper - mean_val

        fig_ci.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[mean_val],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[error_plus],
                arrayminus=[error_minus],
                visible=True,
            ),
            marker_color=bar_color,
            showlegend=False,
        ))

    fig_ci.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
    fig_ci.update_layout(
        title="Bootstrap Mean Coefficients with 95% CIs (Model F, 1,000 iterations)",
        xaxis_title="Variable",
        yaxis_title="Coefficient (Bootstrap Mean)",
        height=430,
    )

    st.plotly_chart(fig_ci, use_container_width=True)
    st.caption("Green bars = CI excludes zero. Red dashed line = zero effect. * marks variables with CI ≠ 0.")

    st.divider()

    # ── Section 3: NLP Synthesis Table ───────────────────────────────────────
    st.subheader("Model F — OLS Coefficients Summary")

    # Color-code by significance
    def style_significance(val):
        if str(val) in ("***", "**", "*"):
            return "background-color: #d4edda; font-weight: bold"
        if str(val) == "n.s.":
            return "color: #888888"
        return ""

    styled_df = df_synth.style.applymap(style_significance, subset=["Significance"])
    st.dataframe(styled_df, use_container_width=True, height=280)

    st.markdown(
        "**Note:** All OLS coefficients use HC3 heteroskedasticity-robust standard errors. "
        "Coefficients are on log-transformed predictors where skewness > 1."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COUNTRY MAP
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "4. Country Map":

    st.title("Country Map: Cereal Food Availability, 160 Countries")
    st.markdown(
        "Dependent variable: cereal food supply for human consumption "
        "(FAO Food Balance Sheet, Element 664, Item 2905, 2021 data). "
        "Values are kg of cereal per person per year."
    )

    df_map = load_csv("page4_country_map.csv")

    # ── Metric strip ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Countries in sample",  str(len(df_map)))
    col2.metric("Mean availability",    str(round(df_map["Cereal Availability (kg/person/yr)"].mean(), 1)) + " kg/pc/yr")
    col3.metric("Median availability",  str(round(df_map["Cereal Availability (kg/person/yr)"].median(), 1)) + " kg/pc/yr")

    # ── Choropleth map ────────────────────────────────────────────────────────
    st.subheader("Choropleth: Availability Bands (Quintiles)")

    band_colour_map = {
        "Very Low":  "#d7191c",
        "Low":       "#fdae61",
        "Medium":    "#ffffbf",
        "High":      "#a6d96a",
        "Very High": "#1a9641",
    }

    band_order = ["Very Low", "Low", "Medium", "High", "Very High"]

    fig_choro = px.choropleth(
        df_map,
        locations="ISO3 Code",
        color="Availability Band",
        hover_name="Country",
        hover_data={
            "Cereal Availability (kg/person/yr)": True,
            "GDP per Capita (USD)": True,
            "Rural Electricity Access (%)": True,
            "Post-Harvest Loss (%)": True,
        },
        color_discrete_map=band_colour_map,
        category_orders={"Availability Band": band_order},
        title="Cereal Food Availability by Country, 2021 (Quintile Bands)",
        projection="natural earth",
    )

    fig_choro.update_layout(
        height=520,
        legend_title_text="Availability Band",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="grey",
            showland=True,
            landcolor="#f0f0f0",
            showocean=True,
            oceancolor="#cce5ff",
        ),
    )

    st.plotly_chart(fig_choro, use_container_width=True)

    # ── Continuous value map ──────────────────────────────────────────────────
    st.subheader("Choropleth: Actual Cereal Availability (kg/person/yr)")

    fig_cont = px.choropleth(
        df_map,
        locations="ISO3 Code",
        color="Cereal Availability (kg/person/yr)",
        hover_name="Country",
        hover_data={
            "Cereal Availability (kg/person/yr)": True,
            "Availability Band": True,
        },
        color_continuous_scale=[
            [0.0,  "#d7191c"],
            [0.25, "#fdae61"],
            [0.5,  "#ffffbf"],
            [0.75, "#a6d96a"],
            [1.0,  "#1a9641"],
        ],
        title="Cereal Food Availability — Continuous Scale",
        projection="natural earth",
    )

    fig_cont.update_layout(
        height=480,
        coloraxis_colorbar_title="kg/person/yr",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="grey",
            showland=True,
            landcolor="#f0f0f0",
            showocean=True,
            oceancolor="#cce5ff",
        ),
    )

    st.plotly_chart(fig_cont, use_container_width=True)

    st.caption(
        "Note: FAO FBS Element 664 measures cereal available for human consumption after "
        "production + imports − exports ± stocks. High-income wheat-exporting countries may "
        "show moderate values because much of their production is exported or used as animal feed."
    )

    st.divider()

    # ── Top and bottom 10 table ───────────────────────────────────────────────
    st.subheader("Top 10 and Bottom 10 Countries by Availability")

    df_sorted = df_map.sort_values("Cereal Availability (kg/person/yr)", ascending=False).reset_index(drop=True)
    df_top10 = df_sorted.head(10).copy()
    df_bot10 = df_sorted.tail(10).copy()

    col_top, col_bot = st.columns(2)

    show_cols = ["Country", "Cereal Availability (kg/person/yr)", "Availability Band"]

    with col_top:
        st.markdown("**Highest availability**")
        st.dataframe(df_top10[show_cols], use_container_width=True, hide_index=True)

    with col_bot:
        st.markdown("**Lowest availability**")
        st.dataframe(df_bot10[show_cols], use_container_width=True, hide_index=True)

    st.divider()

    # ── Country explorer ──────────────────────────────────────────────────────
    st.subheader("Country Explorer")

    country_list = list(df_map["Country"].sort_values())
    selected_country = st.selectbox("Select a country:", country_list)

    selected_row = df_map[df_map["Country"] == selected_country]

    if len(selected_row) > 0:
        r = selected_row.iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Cereal Availability", str(round(float(r["Cereal Availability (kg/person/yr)"]), 1)) + " kg/pc/yr")
        m2.metric("Availability Band",   str(r["Availability Band"]))
        m3.metric("GDP per Capita",      "$" + str(round(float(r["GDP per Capita (USD)"]), 0)))

        m4, m5, m6 = st.columns(3)
        m4.metric("Rural Electricity",   str(round(float(r["Rural Electricity Access (%)"]), 1)) + "%")
        m5.metric("Post-Harvest Loss",   str(round(float(r["Post-Harvest Loss (%)"]), 1)) + "%")
        m6.metric("Trade % GDP",         str(round(float(r["Trade % GDP"]), 1)) + "%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ROBUSTNESS CHECKS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "5. Robustness Checks":

    st.title("Robustness Checks: Specifications and Influential Observations")
    st.markdown(
        "Seven robustness specifications test the stability of the baseline OLS model (A) "
        "across different samples, variable sets, and outlier treatments. "
        "Cook's Distance flags influential observations."
    )

    df_specs    = load_csv("page5_robustness_specs.csv")
    df_cooks    = load_csv("page5_cooks_distance.csv")
    df_frobust  = load_csv("page5_model_f_robustness.csv")

    # ── Section 1: Robustness Specifications ─────────────────────────────────
    st.subheader("Robustness Specifications — Baseline Model A")

    spec_names   = list(df_specs["Specification"])
    spec_n_vals  = list(df_specs["N"])
    spec_r2_vals = list(df_specs["R²"])

    fig_specs = go.Figure()

    fig_specs.add_trace(go.Bar(
        x=spec_names,
        y=spec_r2_vals,
        name="R²",
        marker_color="#1f77b4",
        text=[str(round(v, 3)) for v in spec_r2_vals],
        textposition="outside",
    ))

    fig_specs.update_layout(
        title="R² Across 7 Robustness Specifications",
        xaxis_title="Specification",
        yaxis_title="R²",
        height=430,
        xaxis_tickangle=-25,
        yaxis_range=[0, max(spec_r2_vals) * 1.25],
    )

    st.plotly_chart(fig_specs, use_container_width=True)

    st.markdown(
        "**Key finding:** Removing the 8 Cook-flagged countries (Spec 4) raises R² from 0.176 to 0.311 — "
        "a 77% relative improvement and the largest single-specification gain in the table."
    )

    with st.expander("Show full robustness specifications table"):
        st.dataframe(df_specs, use_container_width=True)

    st.divider()

    # ── Section 2: Cook's Distance ────────────────────────────────────────────
    st.subheader("Cook's Distance — Influential Observations")
    st.markdown(
        "Threshold = 4 / N. Countries exceeding the threshold are flagged as influential. "
        "They represent structural outliers, not data errors."
    )

    n_flagged = 0
    for i in range(len(df_cooks)):
        if str(df_cooks.loc[i, "Flagged"]).strip() == "Yes":
            n_flagged = n_flagged + 1
    n_total   = len(df_cooks)
    threshold = round(float(df_cooks["Threshold"].iloc[0]), 4)

    col1, col2, col3 = st.columns(3)
    col1.metric("Countries analysed", str(n_total))
    col2.metric("Flagged (Cook's D > threshold)", str(n_flagged))
    col3.metric("Threshold (4/N)", str(threshold))

    # Sort descending for the scatter
    df_cooks_sorted = df_cooks.sort_values("Cooks D", ascending=False).reset_index(drop=True)

    # Build colour list
    colour_list = []
    for i in range(len(df_cooks_sorted)):
        if str(df_cooks_sorted.loc[i, "Flagged"]).strip() == "Yes":
            colour_list.append("#d7191c")
        else:
            colour_list.append("#1f77b4")
    df_cooks_sorted["Colour"] = colour_list

    fig_cooks = go.Figure()

    # Non-flagged countries
    mask_normal = []
    mask_flagged = []
    for i in range(len(df_cooks_sorted)):
        if str(df_cooks_sorted.loc[i, "Flagged"]).strip() == "Yes":
            mask_flagged.append(i)
        else:
            mask_normal.append(i)

    normal_rows  = df_cooks_sorted.iloc[mask_normal]
    flagged_rows = df_cooks_sorted.iloc[mask_flagged]

    fig_cooks.add_trace(go.Scatter(
        x=list(range(len(mask_normal))),
        y=list(normal_rows["Cooks D"]),
        mode="markers",
        name="Not flagged",
        marker=dict(color="#1f77b4", size=5),
        text=list(normal_rows["Country"]),
        hovertemplate="%{text}: Cook's D = %{y:.4f}<extra></extra>",
    ))

    fig_cooks.add_trace(go.Scatter(
        x=list(range(len(mask_normal), len(mask_normal) + len(mask_flagged))),
        y=list(flagged_rows["Cooks D"]),
        mode="markers+text",
        name="Flagged",
        marker=dict(color="#d7191c", size=9),
        text=list(flagged_rows["ISO3 Code"]),
        textposition="top center",
        hovertemplate="%{text}: Cook's D = %{y:.4f}<extra></extra>",
    ))

    fig_cooks.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=1.5,
        annotation_text="Threshold = " + str(threshold),
        annotation_position="top right",
    )

    fig_cooks.update_layout(
        title="Cook's Distance for All Countries (sorted descending)",
        xaxis_title="Country rank (high Cook's D first)",
        yaxis_title="Cook's D",
        height=430,
    )

    st.plotly_chart(fig_cooks, use_container_width=True)

    # Flagged country table
    st.markdown("**Flagged countries:**")
    flagged_display = df_cooks_sorted[df_cooks_sorted["Flagged"].astype(str).str.strip() == "Yes"][["ISO3 Code", "Country", "Cooks D"]].copy()
    flagged_display = flagged_display.reset_index(drop=True)
    flagged_display["Cooks D"] = flagged_display["Cooks D"].round(4)
    st.dataframe(flagged_display, use_container_width=True, hide_index=True)

    st.caption(
        "These 8 countries represent three structural types: (1) extreme post-conflict states "
        "(very low cereal availability), (2) small island nations (atypical trade dependency), "
        "and (3) major grain-exporting nations (very high availability from domestic production). "
        "Removing them raises R² from 0.176 to 0.311."
    )

    st.divider()

    # ── Section 3: Model F Robustness ─────────────────────────────────────────
    st.subheader("Model F Robustness Specifications")
    st.markdown(
        "Four additional specifications test the NLP-enhanced Model F across "
        "different sub-samples and variable transformations."
    )

    f_spec_names   = list(df_frobust["Specification"])
    f_spec_r2_vals = []
    for i in range(len(df_frobust)):
        f_spec_r2_vals.append(float(df_frobust.loc[i, "R²"]))

    fig_frobust = go.Figure()

    fig_frobust.add_trace(go.Bar(
        x=f_spec_names,
        y=f_spec_r2_vals,
        name="R²",
        marker_color="#2ca02c",
        text=[str(round(v, 3)) for v in f_spec_r2_vals],
        textposition="outside",
    ))

    fig_frobust.update_layout(
        title="Model F R² Across Robustness Specifications",
        xaxis_title="Specification",
        yaxis_title="R²",
        height=380,
        xaxis_tickangle=-20,
        yaxis_range=[0, max(f_spec_r2_vals) * 1.25],
    )

    st.plotly_chart(fig_frobust, use_container_width=True)

    with st.expander("Show Model F robustness table"):
        st.dataframe(df_frobust, use_container_width=True)
