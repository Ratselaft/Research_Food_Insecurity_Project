// ============================================================
// Page 2 — NLP Findings
// ============================================================

const PageNlp = (function () {
  let loaded = false;
  let tfidf = [], topics = [], map = [];

  function drawTfidfChart() {
    const t = window.getThemeTokens();
    const top20 = [...tfidf].sort((a, b) => Number(a["Rank"]) - Number(b["Rank"])).slice(0, 20).reverse();
    const scores = top20.map((r) => Number(r["TF-IDF Score"]));
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);

    const colors = scores.map((s) => {
      const frac = maxScore > minScore ? (s - minScore) / (maxScore - minScore) : 1;
      const idx = Math.min(t.seq.length - 1, Math.floor(frac * t.seq.length));
      return t.seq[idx];
    });

    Plotly.react("nlp-tfidf-chart", [{
      type: "bar",
      orientation: "h",
      x: scores,
      y: top20.map((r) => r["Keyword"]),
      marker: { color: colors },
    }], ChartsCommon.baseLayout({
      title: { text: "Top 20 Keywords by TF-IDF Score" },
      xaxis: { title: "TF-IDF Score" },
      yaxis: { title: "" },
      height: 540,
      margin: { t: 44, r: 24, l: 140, b: 48 },
    }), ChartsCommon.baseConfig);
  }

  function renderTopics() {
    const el = document.getElementById("nlp-topics");
    el.innerHTML = topics.map((row) => {
      const id = row["Topic ID"];
      const label = row["Topic Label"];
      const kw = row["Top Keywords"];
      const dom = row["Dominant Papers"];
      return (
        '<div class="topic-card">' +
        '<div class="topic-card__head"><span class="topic-card__id">Topic ' + id + '</span>' + label + '</div>' +
        '<div class="topic-card__body"><b>Top keywords:</b> ' + kw + '<br><b>Dominant papers:</b> ' + dom + '</div>' +
        '</div>'
      );
    }).join("");
  }

  function renderMapTable() {
    const priority = ["Topic ID", "NLP Theme", "Proxy Variable (Model F)", "Data Source", "Top Words"];
    const present = priority.filter((c) => map.length > 0 && Object.prototype.hasOwnProperty.call(map[0], c));
    const cols = present.map((c) => ({ key: c, label: c }));
    ChartsCommon.renderTable("nlp-map-table", map, cols);
  }

  function drawAll() {
    drawTfidfChart();
    renderTopics();
    renderMapTable();
  }

  function init() {
    if (loaded) { drawAll(); return; }
    Promise.all([
      DataStore.loadCSV("page2_tfidf_keywords.csv"),
      DataStore.loadCSV("page2_nmf_topics.csv"),
      DataStore.loadCSV("page2_theme_variable_map.csv"),
    ]).then(([a, b, c]) => {
      tfidf = a; topics = b; map = c;
      loaded = true;
      drawAll();
      ChartsCommon.onThemeChange(drawTfidfChart);
    }).catch((err) => {
      document.getElementById("nlp-topics").innerHTML =
        '<p class="caption">Could not load data: ' + err.message + "</p>";
    });
  }

  return { init };
})();
