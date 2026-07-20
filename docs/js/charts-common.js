// ============================================================
// Shared Plotly theming + table rendering + redraw registry.
// ============================================================

const ChartsCommon = (function () {
  const redrawCallbacks = [];

  function onThemeChange(fn) {
    redrawCallbacks.push(fn);
  }

  window.addEventListener("themechange", function () {
    redrawCallbacks.forEach((fn) => {
      try { fn(); } catch (e) { console.error(e); }
    });
  });

  function baseLayout(rawOverrides) {
    const overrides = rawOverrides || {};
    const t = window.getThemeTokens();
    const layout = {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", color: t.textSecondary, size: 12.5 },
      margin: { t: 44, r: 24, l: 56, b: 90 },
      xaxis: { gridcolor: t.grid, zerolinecolor: t.baseline, linecolor: t.baseline, tickcolor: t.baseline, color: t.textMuted, automargin: true },
      yaxis: { gridcolor: t.grid, zerolinecolor: t.baseline, linecolor: t.baseline, tickcolor: t.baseline, color: t.textMuted, automargin: true },
      legend: { font: { color: t.textSecondary } },
      title: { font: { color: t.textPrimary, size: 15 } },
      hoverlabel: { bgcolor: t.surface1, bordercolor: t.baseline, font: { color: t.textPrimary } },
      colorway: t.series,
    };
    // one-level-deep merge for xaxis/yaxis/font/legend/title so page-level
    // overrides (e.g. { xaxis: { title, tickangle } }) add to the themed
    // base instead of replacing it and losing gridcolor/automargin/etc.
    const merged = Object.assign({}, layout, overrides);
    ["xaxis", "yaxis", "font", "legend", "title", "hoverlabel"].forEach((key) => {
      if (overrides[key]) merged[key] = Object.assign({}, layout[key], overrides[key]);
    });
    return merged;
  }

  const baseConfig = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  };

  function fmt(x, decimals) {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    if (typeof x !== "number") return String(x);
    return x.toFixed(decimals === undefined ? 2 : decimals);
  }

  function sigBadge(sig) {
    const s = (sig || "").toString().trim();
    if (s === "***" || s === "**" || s === "*") {
      return '<span class="sig-badge sig-yes">' + s + "</span>";
    }
    if (s === "n.s." || s === "") {
      return '<span class="sig-badge sig-no">' + (s || "n.s.") + "</span>";
    }
    return s;
  }

  /**
   * columns: [{ key, label, numeric?, decimals?, render?(value,row) }]
   */
  function renderTable(elId, rows, columns) {
    const el = document.getElementById(elId);
    if (!el) return;
    if (!rows || rows.length === 0) {
      el.innerHTML = '<p class="caption" style="padding:14px;">No data.</p>';
      return;
    }
    let html = '<table class="data-table"><thead><tr>';
    columns.forEach((c) => { html += "<th>" + c.label + "</th>"; });
    html += "</tr></thead><tbody>";
    rows.forEach((row) => {
      html += "<tr>";
      columns.forEach((c) => {
        let val = row[c.key];
        let out;
        if (c.render) {
          out = c.render(val, row);
        } else if (c.numeric && typeof val === "number") {
          out = fmt(val, c.decimals);
        } else if (val === null || val === undefined || val === "") {
          out = "—";
        } else {
          out = val;
        }
        html += "<td>" + out + "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    el.innerHTML = html;
  }

  return { onThemeChange, baseLayout, baseConfig, fmt, sigBadge, renderTable };
})();
