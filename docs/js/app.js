// ============================================================
// App shell: page navigation + first-load dispatch per page.
// ============================================================

(function () {
  const pageInitters = {
    overview: PageOverview.init,
    nlp: PageNlp.init,
    empirical: PageEmpirical.init,
    map: PageMap.init,
    robustness: PageRobustness.init,
  };

  function showPage(name) {
    document.querySelectorAll(".page").forEach((el) => el.classList.remove("is-active"));
    const target = document.getElementById("page-" + name);
    if (target) target.classList.add("is-active");

    document.querySelectorAll(".sidebar__nav-item").forEach((btn) => {
      if (btn.dataset.page === name) {
        btn.setAttribute("aria-current", "page");
      } else {
        btn.removeAttribute("aria-current");
      }
    });

    if (pageInitters[name]) pageInitters[name]();

    const nav = document.getElementById("sidebarNav");
    if (nav) nav.classList.remove("is-open");

    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }

  document.querySelectorAll(".sidebar__nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      window.location.hash = btn.dataset.page;
      showPage(btn.dataset.page);
    });
  });

  const navToggle = document.getElementById("navToggle");
  if (navToggle) {
    navToggle.addEventListener("click", () => {
      document.getElementById("sidebarNav").classList.toggle("is-open");
    });
  }

  // deep-link support: #nlp, #map, etc.
  const initial = (window.location.hash || "").replace("#", "");
  showPage(pageInitters[initial] ? initial : "overview");

  window.addEventListener("hashchange", () => {
    const name = (window.location.hash || "").replace("#", "");
    if (pageInitters[name]) showPage(name);
  });
})();
