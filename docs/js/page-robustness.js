// ============================================================
// Page 5 — Robustness Checks
// ============================================================

const PageRobustness = (function () {
  let loaded = false;
  let specs = [], cooks = [], frobust = [];

  function drawSpecsChart() {
    const t = window.getThemeTokens();
    const names = specs.map((r) => r["Specification"]);
    const r2 = specs.map((r) => Number(r["R²"]));
    Plotly.react("rob-specs-chart", [{
      type: "bar",
      x: names,
      y: r2,
      marker: { color: t.series[0] },
      text: r2.map((v) => v.toFixed(3)),
      textposition: "outside",
    }], ChartsCommon.baseLayout({
      title: { text: "R² Across Robustness Specifications" },
      xaxis: { title: "Specification", tickangle: -25 },
      yaxis: { title: "R²", range: [0, Math.max(...r2) * 1.25] },
      height: 430,
    }), ChartsCommon.baseConfig);
  }

  function renderKeyFinding() {
    const baseline = specs[0];
    const cookSpec = specs.find((r) => String(r["Specification"]).toLowerCase().indexOf("cook") !== -1);
    const el = document.getElementById("rob-specs-key-finding");
    if (baseline && cookSpec) {
      const r0 = Number(baseline["R²"]);
      const r1 = Number(cookSpec["R²"]);
      const pct = (((r1 - r0) / r0) * 100).toFixed(0);
      el.innerHTML = "<strong>Key finding:</strong> Removing the Cook-flagged countries (" + cookSpec["Specification"] +
        ") raises R² from " + r0.toFixed(3) + " to " + r1.toFixed(3) +
        " — a " + pct + "% relative improvement.";
    } else {
      el.innerHTML = "";
    }
  }

  function renderSpecsTable() {
    const cols = Object.keys(specs[0] || {}).map((k) => ({
      key: k, label: k, numeric: typeof specs[0][k] === "number",
      decimals: k.toLowerCase().indexOf("coef") !== -1 || k === "R²" || k === "Adj R²" ? 3 : 0,
    }));
    ChartsCommon.renderTable("rob-specs-table", specs, cols);
  }

  function renderCooksStats() {
    const flagged = cooks.filter((r) => String(r["Flagged"]).trim() === "Yes");
    const threshold = cooks.length ? Number(cooks[0]["Threshold"]) : 0;
    const el = document.getElementById("rob-cooks-stats");
    const tiles = [
      { label: "Countries analysed", value: String(cooks.length) },
      { label: "Flagged (Cook's D > threshold)", value: String(flagged.length) },
      { label: "Threshold (4/N)", value: ChartsCommon.fmt(threshold, 4) },
    ];
    el.innerHTML = tiles.map((t) =>
      '<div class="stat-tile"><div class="stat-tile__label">' + t.label +
      '</div><div class="stat-tile__value">' + t.value + "</div></div>"
    ).join("");
  }

  function drawCooksChart() {
    const t = window.getThemeTokens();
    const sorted = [...cooks].sort((a, b) => Number(b["Cooks D"]) - Number(a["Cooks D"]));
    const threshold = sorted.length ? Number(sorted[0]["Threshold"]) : 0;
    const normal = sorted.filter((r) => String(r["Flagged"]).trim() !== "Yes");
    const flagged = sorted.filter((r) => String(r["Flagged"]).trim() === "Yes");

    const traces = [
      {
        type: "scatter", mode: "markers", name: "Not flagged",
        x: normal.map((_, i) => i),
        y: normal.map((r) => Number(r["Cooks D"])),
        marker: { color: t.series[0], size: 6 },
        text: normal.map((r) => r["Country"]),
        hovertemplate: "%{text}: Cook's D = %{y:.4f}<extra></extra>",
      },
      {
        type: "scatter", mode: "markers+text", name: "Flagged",
        x: flagged.map((_, i) => normal.length + i),
        y: flagged.map((r) => Number(r["Cooks D"])),
        marker: { color: t.statusCritical, size: 10 },
        text: flagged.map((r) => r["ISO3 Code"]),
        textposition: "top center",
        textfont: { color: t.textSecondary },
        hovertemplate: "%{text}: Cook's D = %{y:.4f}<extra></extra>",
      },
    ];

    const layout = ChartsCommon.baseLayout({
      title: { text: "Cook's Distance for All Countries (sorted descending)" },
      xaxis: { title: "Country rank (high Cook's D first)" },
      yaxis: { title: "Cook's D" },
      height: 430,
      shapes: [{
        type: "line", x0: 0, x1: sorted.length - 1, y0: threshold, y1: threshold,
        line: { color: t.statusCritical, dash: "dash", width: 1.5 },
      }],
      annotations: [{
        x: sorted.length - 1, y: threshold, xanchor: "right", yanchor: "bottom",
        text: "Threshold = " + threshold.toFixed(4), showarrow: false,
        font: { color: t.textSecondary, size: 11 },
      }],
    });
    Plotly.react("rob-cooks-chart", traces, layout, ChartsCommon.baseConfig);
  }

  function renderCooksTable() {
    const flagged = cooks.filter((r) => String(r["Flagged"]).trim() === "Yes")
      .sort((a, b) => Number(b["Cooks D"]) - Number(a["Cooks D"]));
    const cols = [
      { key: "ISO3 Code", label: "ISO3" },
      { key: "Country", label: "Country" },
      { key: "Cooks D", label: "Cook's D", numeric: true, decimals: 4 },
    ];
    ChartsCommon.renderTable("rob-cooks-table", flagged, cols);
  }

  function drawFrobustChart() {
    const t = window.getThemeTokens();
    const names = frobust.map((r) => r["Specification"]);
    const r2 = frobust.map((r) => Number(r["R²"]));
    Plotly.react("rob-frobust-chart", [{
      type: "bar",
      x: names,
      y: r2,
      marker: { color: t.series[1] },
      text: r2.map((v) => v.toFixed(3)),
      textposition: "outside",
    }], ChartsCommon.baseLayout({
      title: { text: "Model F R² Across Robustness Specifications" },
      xaxis: { title: "Specification", tickangle: -20 },
      yaxis: { title: "R²", range: [0, Math.max(...r2) * 1.25] },
      height: 380,
    }), ChartsCommon.baseConfig);
  }

  function renderFrobustTable() {
    const cols = Object.keys(frobust[0] || {}).map((k) => ({
      key: k, label: k, numeric: typeof frobust[0][k] === "number",
      decimals: k.toLowerCase().indexOf("coef") !== -1 || k === "R²" || k === "Adj R²" ? 3 : 0,
    }));
    ChartsCommon.renderTable("rob-frobust-table", frobust, cols);
  }

  function drawAll() {
    drawSpecsChart();
    renderKeyFinding();
    renderSpecsTable();
    renderCooksStats();
    drawCooksChart();
    renderCooksTable();
    drawFrobustChart();
    renderFrobustTable();
  }

  function init() {
    if (loaded) { drawSpecsChart(); drawCooksChart(); drawFrobustChart(); return; }
    Promise.all([
      DataStore.loadCSV("page5_robustness_specs.csv"),
      DataStore.loadCSV("page5_cooks_distance.csv"),
      DataStore.loadCSV("page5_model_f_robustness.csv"),
    ]).then(([a, b, c]) => {
      specs = a; cooks = b; frobust = c;
      loaded = true;
      drawAll();
      ChartsCommon.onThemeChange(() => { drawSpecsChart(); drawCooksChart(); drawFrobustChart(); });
    }).catch((err) => {
      document.getElementById("rob-cooks-stats").innerHTML =
        '<p class="caption">Could not load data: ' + err.message + "</p>";
    });
  }

  return { init };
})();
