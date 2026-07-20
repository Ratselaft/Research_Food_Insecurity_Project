// ============================================================
// Theme toggle: light / dark, persisted, dispatches 'themechange'
// so chart modules can re-render with the right surface colors.
// ============================================================

(function () {
  const STORAGE_KEY = "fi-dashboard-theme";

  function applyTheme(theme) {
    if (theme === "dark" || theme === "light") {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }

  function currentEffectiveTheme() {
    const stamped = document.documentElement.getAttribute("data-theme");
    if (stamped) return stamped;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function updateToggleUI() {
    const effective = currentEffectiveTheme();
    const icon = document.getElementById("themeToggleIcon");
    const label = document.getElementById("themeToggleLabel");
    if (icon && label) {
      if (effective === "dark") {
        icon.textContent = "☀️";
        label.textContent = "Light mode";
      } else {
        icon.textContent = "🌙";
        label.textContent = "Dark mode";
      }
    }
  }

  const saved = localStorage.getItem(STORAGE_KEY);
  applyTheme(saved);
  updateToggleUI();

  document.addEventListener("DOMContentLoaded", function () {
    updateToggleUI();
    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", function () {
        const next = currentEffectiveTheme() === "dark" ? "light" : "dark";
        applyTheme(next);
        localStorage.setItem(STORAGE_KEY, next);
        updateToggleUI();
        window.dispatchEvent(new CustomEvent("themechange", { detail: { theme: next } }));
      });
    }
  });

  window.getThemeTokens = function () {
    const styles = getComputedStyle(document.documentElement);
    const get = (name) => styles.getPropertyValue(name).trim();
    return {
      surface1: get("--surface-1"),
      textPrimary: get("--text-primary"),
      textSecondary: get("--text-secondary"),
      textMuted: get("--text-muted"),
      grid: get("--grid-hairline"),
      baseline: get("--baseline"),
      series: [
        get("--series-1"), get("--series-2"), get("--series-3"), get("--series-4"),
        get("--series-5"), get("--series-6"), get("--series-7"), get("--series-8"),
      ],
      seq: [
        get("--seq-250"), get("--seq-350"), get("--seq-450"), get("--seq-550"), get("--seq-650"),
      ],
      seqFull: [
        get("--seq-100"), get("--seq-150"), get("--seq-200"), get("--seq-250"), get("--seq-300"),
        get("--seq-350"), get("--seq-400"), get("--seq-450"), get("--seq-500"), get("--seq-550"),
        get("--seq-600"), get("--seq-650"), get("--seq-700"),
      ],
      statusGood: get("--status-good"),
      statusCritical: get("--status-critical"),
      statusWarning: get("--status-warning"),
    };
  };
})();
