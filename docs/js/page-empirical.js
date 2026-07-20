// ============================================================
// Page 3 — Empirical Results
// ============================================================

const PageEmpirical = (function () {
  let loaded = false;
  let ci = [], ftest = [], synth = [];

  function renderFtestStats() {
    const row = ftest[0];
    const el = document.getElementById("emp-ftest-stats");
    const tiles = [
      { label: "F statistic", value: ChartsCommon.fmt(Number(row["F statistic"]), 3) },
      { label: "p-value", value: ChartsCommon.fmt(Number(row["p-value"]), 4) },
      { label: "Significance", value: row["Significance"] },
      { label: "Partial R²", value: ChartsCommon.fmt(Number(row["Partial R²"]), 3) },
    ];
    el.innerHTML = tiles.map((t) =>
      '<div class="stat-tile"><div class="stat-tile__label">' + t.label +
      '</div><div class="stat-tile__value">' + t.value + "</div></div>"
    ).join("");

    document.getElementById("emp-ftest-interp").innerHTML =
      "<strong>Interpretation:</strong> F(" + row["df1 (extra vars)"] + ", " + row["df2"] + ") = " +
      ChartsCommon.fmt(Number(row["F statistic"]), 3) + ", p = " + ChartsCommon.fmt(Number(row["p-value"]), 4) +
      " — the five NLP-identified variables jointly improve model fit at the 1% significance level.";
  }

  function drawCiChart() {
    const t = window.getThemeTokens();
    const vars = ci.map((r) => r["Variable"]);
    const mean = ci.map((r) => Number(r["Bootstrap Mean"]));
    const lower = ci.map((r) => Number(r["95% CI Lower"]));
    const upper = ci.map((r) => Number(r["95% CI Upper"]));
    const excl = ci.map((r) => String(r["CI Excludes Zero"]).trim().indexOf("Yes") === 0);
    const colors = excl.map((e) => (e ? t.statusGood : t.textMuted));
    const labels = vars.map((v, i) => (excl[i] ? v + " *" : v));

    const trace = {
      type: "bar",
      x: labels,
      y: mean,
      marker: { color: colors },
      error_y: {
        type: "data",
        symmetric: false,
        array: upper.map((u, i) => u - mean[i]),
        arrayminus: mean.map((m, i) => m - lower[i]),
        visible: true,
        color: t.textMuted,
      },
    };

    const layout = ChartsCommon.baseLayout({
      title: { text: "Bootstrap Mean Coefficients with 95% CIs (Model F, 1,000 iterations)" },
      xaxis: { title: "Variable" },
      yaxis: { title: "Coefficient (Bootstrap Mean)" },
      height: 430,
      shapes: [{
        type: "line", x0: -0.5, x1: vars.length - 0.5, y0: 0, y1: 0,
        line: { color: t.statusCritical, dash: "dash", width: 1 },
      }],
      showlegend: false,
    });

    Plotly.react("emp-ci-chart", [trace], layout, ChartsCommon.baseConfig);
  }

  function renderSynthTable() {
    const cols = [
      { key: "NLP Theme", label: "NLP Theme" },
      { key: "Model F Variable", label: "Model F Variable" },
      { key: "OLS Coefficient", label: "OLS Coefficient", numeric: true, decimals: 4 },
      { key: "p-value", label: "p-value", numeric: true, decimals: 3 },
      { key: "Significance", label: "Significance", render: (v) => ChartsCommon.sigBadge(v) },
      { key: "Direction", label: "Direction" },
    ];
    ChartsCommon.renderTable("emp-synth-table", synth, cols);
  }

  function drawAll() {
    renderFtestStats();
    drawCiChart();
    renderSynthTable();
  }

  function init() {
    if (loaded) { drawAll(); return; }
    Promise.all([
      DataStore.loadCSV("page3_bootstrap_cis.csv"),
      DataStore.loadCSV("page3_ftest_summary.csv"),
      DataStore.loadCSV("page3_nlp_synthesis.csv"),
    ]).then(([a, b, c]) => {
      ci = a; ftest = b; synth = c;
      loaded = true;
      drawAll();
      ChartsCommon.onThemeChange(drawCiChart);
    }).catch((err) => {
      document.getElementById("emp-ftest-stats").innerHTML =
        '<p class="caption">Could not load data: ' + err.message + "</p>";
    });
  }

  return { init };
})();
