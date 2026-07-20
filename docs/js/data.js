// ============================================================
// CSV loading helper — fetch + PapaParse, with in-memory cache
// so switching pages doesn't re-fetch data already loaded.
// ============================================================

const DataStore = (function () {
  const cache = {};

  function loadCSV(filename) {
    if (cache[filename]) return cache[filename];
    const promise = fetch("data/" + filename)
      .then((resp) => {
        if (!resp.ok) throw new Error("Data file not found: data/" + filename);
        return resp.text();
      })
      .then((text) => {
        const parsed = Papa.parse(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
        });
        return parsed.data;
      });
    cache[filename] = promise;
    return promise;
  }

  return { loadCSV };
})();
