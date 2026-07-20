# Live web dashboard

This folder is a self-contained static website — the same 5 pages as
`dashboard.py` (Streamlit), rebuilt in plain HTML/CSS/JS so it can be hosted
for free on **GitHub Pages** instead of needing a running Python server.

- `index.html` — page shell + sidebar navigation (single-page app, no reloads)
- `css/style.css` — design system (light/dark mode, both fully supported)
- `js/` — one file per dashboard page, plus shared theme/data/chart helpers
- `data/*.csv` — the exact data `dashboard.py` reads, copied from `outputs/powerbi/`

Charts are drawn with [Plotly.js](https://plotly.com/javascript/) (loaded via
CDN) and CSVs are parsed with [PapaParse](https://www.papaparse.com/) (also
CDN) — no build step, no bundler, nothing to install.

## Enabling GitHub Pages (one-time setup)

1. Push this repo to GitHub (already done if you're reading this from GitHub).
2. On GitHub: **Settings → Pages**.
3. Under "Build and deployment", set **Source** to "Deploy from a branch".
4. Set **Branch** to `main` and the folder to **`/docs`**, then **Save**.
5. Wait ~1 minute. Your dashboard will be live at:
   `https://<your-github-username>.github.io/<repo-name>/`

Any time you push a change to `docs/`, the live site updates automatically
within a minute or two — no rebuild step required.

## Keeping the data in sync

`src/step10_export_for_dashboard.py` writes the dashboard's source CSVs to
`outputs/powerbi/` **and** copies them into `docs/data/` automatically. So
re-running the pipeline (`bash run_pipeline.sh`, or just `python
src/step10_export_for_dashboard.py`) and pushing the updated `docs/data/*.csv`
files is all that's needed to refresh the live site with new results.

## Running it locally before pushing

```bash
cd docs
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a browser. (Opening `index.html`
directly via `file://` won't work — browsers block `fetch()` on CSVs from
the local filesystem, so it needs to be served over HTTP, even just locally.)
