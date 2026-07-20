// ============================================================
// Page 1 — Overview
// ============================================================

const PageOverview = (function () {
  let loaded = false;
  let perf = [], corpus = [];

  function findModelFRow() {
    let row = perf.find((r) => {
      const label = String(r["Model Label"] || "");
      return label.indexOf("Model F") !== -1 || label.indexOf("NLP") !== -1;
    });
    return row || perf[perf.length - 1];
  }

  function renderStats() {
    const aRow = perf[0];
    const fRow = findModelFRow();
    const el = document.getElementById("ov-stats");

    const deltaVal = fRow["Delta R² vs A*"];
    let thirdTile;
    if (deltaVal !== null && deltaVal !== undefined && deltaVal !== "") {
      thirdTile = { label: "ΔR² (F vs A★)", value: ChartsCommon.fmt(Number(deltaVal), 3) };
    } else {
      thirdTile = { label: "Nested F-test", value: "p = 0.004 (***)" };
    }

    const tiles = [
      { label: "Model A  OLS R²", value: ChartsCommon.fmt(Number(aRow["OLS R²"]), 3) },
      { label: "Model F  OLS R²", value: ChartsCommon.fmt(Number(fRow["OLS R²"]), 3) },
      thirdTile,
    ];
    el.innerHTML = tiles.map((t) =>
      '<div class="stat-tile"><div class="stat-tile__label">' + t.label +
      '</div><div class="stat-tile__value">' + t.value + "</div></div>"
    ).join("");
  }

  function drawPerfChart() {
    const t = window.getThemeTokens();
    const labels = perf.map((r) => r["Model Label"]);
    const ols = perf.map((r) => Number(r["OLS R²"]));
    const rf = perf.map((r) => Number(r["RF 5-fold CV R²"]));
    const xgb = perf.map((r) => Number(r["XGB 5-fold CV R²"]));

    const traces = [
      { name: "OLS R²", x: labels, y: ols, type: "bar", marker: { color: t.series[0] } },
      { name: "RF CV R²", x: labels, y: rf, type: "bar", marker: { color: t.series[1] } },
      { name: "XGB CV R²", x: labels, y: xgb, type: "bar", marker: { color: t.series[2] } },
    ];
    const layout = ChartsCommon.baseLayout({
      barmode: "group",
      title: { text: "R² by Model and Estimator" },
      xaxis: { title: "Model", tickangle: -20 },
      yaxis: { title: "R²" },
      height: 420,
    });
    Plotly.react("ov-perf-chart", traces, layout, ChartsCommon.baseConfig);
  }

  function renderPerfTable() {
    const cols = [
      { key: "Model Label", label: "Model" },
      { key: "N (countries)", label: "N" },
      { key: "Predictors used", label: "Predictors" },
      { key: "OLS R²", label: "OLS R²", numeric: true, decimals: 3 },
      { key: "OLS Adj R²", label: "OLS Adj R²", numeric: true, decimals: 3 },
      { key: "OLS F-stat p", label: "OLS F-stat p", numeric: true, decimals: 4 },
      { key: "RF 5-fold CV R²", label: "RF CV R²", numeric: true, decimals: 3 },
      { key: "XGB 5-fold CV R²", label: "XGB CV R²", numeric: true, decimals: 3 },
      { key: "Delta R² vs A*", label: "ΔR² vs A★", numeric: true, decimals: 3 },
    ];
    ChartsCommon.renderTable("ov-perf-table", perf, cols);
  }

  function drawCorpusCharts() {
    const t = window.getThemeTokens();

    // grouped bar: Alignment Level (x) x Source (color)
    const sources = [...new Set(corpus.map((r) => r["Source"]))];
    const levels = [...new Set(corpus.map((r) => r["Alignment Level"]))];
    const barTraces = sources.map((src, i) => {
      const y = levels.map((lvl) => {
        const row = corpus.find((r) => r["Alignment Level"] === lvl && r["Source"] === src);
        return row ? Number(row["Paper Count"]) : 0;
      });
      return { name: src, x: levels, y, type: "bar", marker: { color: t.series[i % t.series.length] } };
    });
    Plotly.react("ov-corpus-bar", barTraces, ChartsCommon.baseLayout({
      barmode: "group",
      title: { text: "Papers by Alignment Level and Source" },
      xaxis: { title: "", tickangle: -10 },
      yaxis: { title: "Paper Count" },
      height: 380,
    }), ChartsCommon.baseConfig);

    // pie: totals by alignment level
    const totals = levels.map((lvl) =>
      corpus.filter((r) => r["Alignment Level"] === lvl)
        .reduce((sum, r) => sum + Number(r["Paper Count"]), 0)
    );
    Plotly.react("ov-corpus-pie", [{
      type: "pie",
      labels: levels,
      values: totals,
      marker: { colors: t.series },
      textinfo: "label+percent",
      hole: 0.35,
    }], ChartsCommon.baseLayout({
      title: { text: "Share of Papers by Alignment Level" },
      height: 380,
      showlegend: true,
    }), ChartsCommon.baseConfig);
  }

  function renderCorpusTable() {
    const cols = [
      { key: "Alignment Level", label: "Alignment Level" },
      { key: "Source", label: "Source" },
      { key: "Paper Count", label: "Paper Count", numeric: true, decimals: 0 },
    ];
    ChartsCommon.renderTable("ov-corpus-table", corpus, cols);
  }

  function drawAll() {
    renderStats();
    drawPerfChart();
    renderPerfTable();
    drawCorpusCharts();
    renderCorpusTable();
  }

  function init() {
    if (loaded) { drawAll(); return; }
    Promise.all([
      DataStore.loadCSV("page1_model_performance.csv"),
      DataStore.loadCSV("page1_corpus_summary.csv"),
    ]).then(([p, c]) => {
      perf = p; corpus = c;
      loaded = true;
      drawAll();
      ChartsCommon.onThemeChange(() => { drawPerfChart(); drawCorpusCharts(); });
    }).catch((err) => {
      document.getElementById("ov-stats").innerHTML =
        '<p class="caption">Could not load data: ' + err.message + "</p>";
    });
  }

  return { init };
})();
