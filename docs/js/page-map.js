// ============================================================
// Page 4 — Country Map
// ============================================================

const PageMap = (function () {
  let loaded = false;
  let rows = [];
  const BAND_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"];

  function renderStats() {
    const el = document.getElementById("map-stats");
    const vals = rows.map((r) => Number(r["Cereal Availability (kg/person/yr)"])).sort((a, b) => a - b);
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    const median = vals.length % 2 === 0
      ? (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2
      : vals[Math.floor(vals.length / 2)];

    const tiles = [
      { label: "Countries in sample", value: String(rows.length) },
      { label: "Mean availability", value: ChartsCommon.fmt(mean, 1) + " kg/pc/yr" },
      { label: "Median availability", value: ChartsCommon.fmt(median, 1) + " kg/pc/yr" },
    ];
    el.innerHTML = tiles.map((t) =>
      '<div class="stat-tile"><div class="stat-tile__label">' + t.label +
      '</div><div class="stat-tile__value">' + t.value + "</div></div>"
    ).join("");
  }

  function drawBandChoropleth() {
    const t = window.getThemeTokens();
    const bandColors = {
      "Very Low": t.seq[0], "Low": t.seq[1], "Medium": t.seq[2], "High": t.seq[3], "Very High": t.seq[4],
    };
    const traces = BAND_ORDER.filter((b) => rows.some((r) => r["Availability Band"] === b)).map((band) => {
      const subset = rows.filter((r) => r["Availability Band"] === band);
      return {
        type: "choropleth",
        locationmode: "ISO-3",
        locations: subset.map((r) => r["ISO3 Code"]),
        z: subset.map(() => 1),
        text: subset.map((r) =>
          r["Country"] + "<br>Availability: " + ChartsCommon.fmt(Number(r["Cereal Availability (kg/person/yr)"]), 1) +
          " kg/pc/yr<br>GDP/capita: $" + ChartsCommon.fmt(Number(r["GDP per Capita (USD)"]), 0) +
          "<br>Rural electricity: " + ChartsCommon.fmt(Number(r["Rural Electricity Access (%)"]), 1) + "%" +
          "<br>Post-harvest loss: " + ChartsCommon.fmt(Number(r["Post-Harvest Loss (%)"]), 1) + "%"
        ),
        hoverinfo: "text",
        name: band,
        showscale: false,
        colorscale: [[0, bandColors[band]], [1, bandColors[band]]],
        marker: { line: { color: t.grid, width: 0.4 } },
      };
    });

    const layout = ChartsCommon.baseLayout({
      title: { text: "Cereal Food Availability by Country, 2021 (Quintile Bands)" },
      height: 520,
      showlegend: true,
      legend: { title: { text: "Availability Band" }, font: { color: t.textSecondary } },
      geo: {
        showframe: false, showcoastlines: true, coastlinecolor: t.grid,
        showland: true, landcolor: t.grid, showocean: true, oceancolor: t.surface1,
        bgcolor: "transparent", projection: { type: "natural earth" },
      },
    });
    Plotly.react("map-choro-band", traces, layout, ChartsCommon.baseConfig);
  }

  function drawContinuousChoropleth() {
    const t = window.getThemeTokens();
    const values = rows.map((r) => Number(r["Cereal Availability (kg/person/yr)"]));
    const colorscale = t.seqFull.map((hex, i) => [i / (t.seqFull.length - 1), hex]);

    const trace = {
      type: "choropleth",
      locationmode: "ISO-3",
      locations: rows.map((r) => r["ISO3 Code"]),
      z: values,
      text: rows.map((r) => r["Country"] + " (" + r["Availability Band"] + ")"),
      hoverinfo: "text+z",
      colorscale,
      colorbar: { title: { text: "kg/person/yr", side: "right" }, tickfont: { color: t.textMuted } },
      marker: { line: { color: t.grid, width: 0.4 } },
    };
    const layout = ChartsCommon.baseLayout({
      title: { text: "Cereal Food Availability — Continuous Scale" },
      height: 480,
      geo: {
        showframe: false, showcoastlines: true, coastlinecolor: t.grid,
        showland: true, landcolor: t.grid, showocean: true, oceancolor: t.surface1,
        bgcolor: "transparent", projection: { type: "natural earth" },
      },
    });
    Plotly.react("map-choro-cont", [trace], layout, ChartsCommon.baseConfig);
  }

  function renderTopBottomTables() {
    const sorted = [...rows].sort((a, b) =>
      Number(b["Cereal Availability (kg/person/yr)"]) - Number(a["Cereal Availability (kg/person/yr)"])
    );
    const cols = [
      { key: "Country", label: "Country" },
      { key: "Cereal Availability (kg/person/yr)", label: "kg/pc/yr", numeric: true, decimals: 1 },
      { key: "Availability Band", label: "Band" },
    ];
    ChartsCommon.renderTable("map-top10", sorted.slice(0, 10), cols);
    ChartsCommon.renderTable("map-bot10", sorted.slice(-10).reverse(), cols);
  }

  function renderCountryExplorer() {
    const select = document.getElementById("map-country-select");
    const countries = [...rows].sort((a, b) => String(a["Country"]).localeCompare(String(b["Country"])));
    select.innerHTML = countries.map((r) => '<option value="' + r["Country"] + '">' + r["Country"] + "</option>").join("");

    function renderFor(name) {
      const r = rows.find((row) => row["Country"] === name);
      if (!r) return;
      const el = document.getElementById("map-country-stats");
      const tiles = [
        { label: "Cereal Availability", value: ChartsCommon.fmt(Number(r["Cereal Availability (kg/person/yr)"]), 1) + " kg/pc/yr" },
        { label: "Availability Band", value: r["Availability Band"] },
        { label: "GDP per Capita", value: "$" + ChartsCommon.fmt(Number(r["GDP per Capita (USD)"]), 0) },
        { label: "Rural Electricity", value: ChartsCommon.fmt(Number(r["Rural Electricity Access (%)"]), 1) + "%" },
        { label: "Post-Harvest Loss", value: ChartsCommon.fmt(Number(r["Post-Harvest Loss (%)"]), 1) + "%" },
        { label: "Trade % GDP", value: ChartsCommon.fmt(Number(r["Trade % GDP"]), 1) + "%" },
      ];
      el.innerHTML = tiles.map((t) =>
        '<div class="stat-tile"><div class="stat-tile__label">' + t.label +
        '</div><div class="stat-tile__value small">' + t.value + "</div></div>"
      ).join("");
    }

    select.addEventListener("change", () => renderFor(select.value));
    if (countries.length > 0) {
      select.value = countries[0]["Country"];
      renderFor(countries[0]["Country"]);
    }
  }

  function drawAll() {
    renderStats();
    drawBandChoropleth();
    drawContinuousChoropleth();
    renderTopBottomTables();
    renderCountryExplorer();
  }

  function init() {
    if (loaded) { drawBandChoropleth(); drawContinuousChoropleth(); return; }
    DataStore.loadCSV("page4_country_map.csv").then((data) => {
      rows = data.filter((r) => r["ISO3 Code"]);
      loaded = true;
      drawAll();
      ChartsCommon.onThemeChange(() => { drawBandChoropleth(); drawContinuousChoropleth(); });
    }).catch((err) => {
      document.getElementById("map-stats").innerHTML =
        '<p class="caption">Could not load data: ' + err.message + "</p>";
    });
  }

  return { init };
})();
