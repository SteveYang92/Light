# ruff: noqa: E501

import json
from pathlib import Path


class DashboardGenerator:
    def generate(self, snapshots_dir: Path, output_path: Path) -> None:
        cases = self._load_cases(snapshots_dir)
        html = self._render_html(cases)
        output_path.write_text(html, encoding="utf-8")

    def _load_cases(self, snapshots_dir: Path) -> list[dict]:
        cases = []
        if not snapshots_dir.exists():
            return cases

        for case_dir in snapshots_dir.iterdir():
            if not case_dir.is_dir():
                continue

            manifest_path = case_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs = []

            for run_id in manifest.get("runs", []):
                run_dir = case_dir / "runs" / run_id
                if not run_dir.exists():
                    continue

                diff_path = run_dir / "diff.json"
                diff = json.loads(diff_path.read_text(encoding="utf-8")) if diff_path.exists() else {}
                report_path = run_dir / "report.json"
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

                e = report.get("errors", 0)
                w = report.get("warnings", 0)
                s_val = report.get("suggestions", 0)

                issues = report.get("issues", [])
                rules = {}
                for i in issues:
                    rule = i.get("rule", "unknown")
                    sev = i.get("severity", "error")
                    if rule not in rules:
                        rules[rule] = {"error": 0, "warning": 0, "suggestion": 0, "total": 0}
                    rules[rule][sev] = rules[rule].get(sev, 0) + 1
                    rules[rule]["total"] += 1

                runs.append(
                    {
                        "run_id": run_id,
                        "degraded": diff.get("degraded", False),
                        "errors": e,
                        "warnings": w,
                        "suggestions": s_val,
                        "total": e + w + s_val,
                        "rules": rules,
                        "errors_delta": diff.get("errors_delta", 0),
                        "warnings_delta": diff.get("warnings_delta", 0),
                        "suggestions_delta": diff.get("suggestions_delta", 0),
                        "reasons": diff.get("reasons", []),
                    }
                )

            cases.append(
                {
                    "name": manifest["case_name"],
                    "baseline_run_id": manifest.get("baseline_run_id"),
                    "latest_run_id": manifest.get("latest_run_id"),
                    "runs": runs,
                }
            )

        return cases

    def _render_html(self, cases: list[dict]) -> str:
        cases_json = json.dumps(cases, ensure_ascii=False)

        return (
            """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>light-subtitle Regression Dashboard</title>
<style>
:root {
  --bg: #f8f9fb;
  --surface: #ffffff;
  --border: #e5e7eb;
  --text-primary: #111827;
  --text-secondary: #6b7280;
  --text-muted: #9ca3af;
  --error: #ef4444;
  --error-bg: #fef2f2;
  --warning: #f59e0b;
  --suggestion: #8b5cf6;
  --success: #10b981;
  --success-bg: #f0fdf4;
  --accent: #3b82f6;
  --radius: 10px;
  --shadow: 0 1px 3px rgba(0,0,0,0.1);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  background: var(--bg);
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.6;
  padding: 32px 24px;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 28px; font-weight: 700; margin-bottom: 24px; letter-spacing: -0.025em; }
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}
.summary-card {
  background: var(--surface);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  padding: 20px;
  box-shadow: var(--shadow);
  text-align: center;
}
.summary-card .number { font-size: 32px; font-weight: 700; display: block; margin-bottom: 4px; }
.summary-card .label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.cases-grid { display: grid; gap: 24px; grid-template-columns: 1fr; margin-bottom: 32px; }
.case-card {
  background: var(--surface);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-left: 4px solid var(--success);
  overflow: hidden;
}
.case-card.degraded { border-left-color: var(--error); }
.case-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px 0;
  flex-wrap: wrap;
  gap: 12px;
}
.case-card-header h3 { font-size: 16px; font-weight: 600; }
.case-card-header .status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 100px;
}
.status.pass { background: var(--success-bg); color: #166534; }
.status.degraded { background: var(--error-bg); color: #991b1b; }
.case-card-body { padding: 16px 24px; }
.trend-section {
  margin-bottom: 8px;
}
.trend-chart {
  width: 100%;
  height: 160px;
  display: block;
}
.trend-legend {
  display: flex;
  gap: 20px;
  font-size: 12px;
  color: var(--text-secondary);
  padding: 8px 0 4px;
  user-select: none;
}
.trend-legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}
.trend-legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.trend-legend-count {
  font-weight: 600;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}
.trend-legend-label {
  text-transform: capitalize;
}
.trend-tooltip {
  position: fixed;
  background: var(--text-primary);
  color: #fff;
  font-size: 12px;
  padding: 8px 12px;
  border-radius: 6px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.15s;
  z-index: 100;
  white-space: nowrap;
  line-height: 1.5;
}
.trend-tooltip.visible { opacity: 1; }
.runs-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.runs-table th {
  text-align: left;
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  padding: 8px 12px 6px;
  border-bottom: 1px solid var(--border);
}
.runs-table td {
  padding: 7px 12px;
  border-bottom: 1px solid var(--border);
  font-variant-numeric: tabular-nums;
}
.runs-table tr:last-child td { border-bottom: none; }
.runs-table .run-id { color: var(--text-secondary); font-family: "SF Mono", monospace; font-size: 12px; }
.runs-table .val-errors { color: var(--error); font-weight: 600; }
.runs-table .val-warnings { color: var(--warning); font-weight: 600; }
.runs-table .val-suggestions { color: var(--suggestion); font-weight: 600; }
.runs-table .val-total { color: var(--text-primary); font-weight: 600; }
.runs-table .delta-pos { color: var(--error); }
.runs-table .delta-neg { color: var(--success); }
.runs-table .delta-zero { color: var(--text-muted); }
.alert {
  background: var(--error-bg);
  border: 1px solid #fecaca;
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 16px;
  color: #991b1b;
}
.alert h4 { font-size: 14px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.alert ul { margin-left: 16px; font-size: 13px; }
.alert li { margin: 4px 0; }
.empty {
  text-align: center;
  padding: 64px 24px;
  color: var(--text-muted);
}
.empty h3 { font-size: 16px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; }
/* ── Tag Cloud ─────────────────────────────────────── */
.tag-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
  align-items: center;
}
.tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px 14px;
  border-radius: 100px;
  font-size: 13px;
  font-weight: 500;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  background: var(--surface);
  cursor: pointer;
  transition: all 0.15s ease;
  user-select: none;
}
.tag:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.tag.selected {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.tag.selected:hover {
  opacity: 0.85;
}
.tag-clear {
  border-color: transparent;
  color: var(--text-muted);
  font-size: 12px;
  margin-left: 4px;
}
.tag-clear:hover {
  color: var(--error);
  border-color: transparent;
}
.selected-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 12px;
  font-weight: 500;
  background: var(--accent);
  color: #fff;
}
.chip-remove {
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  opacity: 0.7;
  transition: opacity 0.15s;
}
.chip-remove:hover {
  opacity: 1;
}
.empty-filtered {
  text-align: center;
  padding: 64px 24px;
  color: var(--text-muted);
}
.empty-filtered h3 { font-size: 16px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; }

@media (max-width: 640px) {
  .trend-chart { height: 120px; }
}

/* ── Modal ──────────────────────────────────────────── */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.4);
  z-index: 200;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.25s ease;
}
.modal-overlay.open {
  opacity: 1;
  pointer-events: auto;
}
.modal-box {
  background: var(--surface);
  border-radius: var(--radius);
  box-shadow: 0 20px 60px rgba(0,0,0,0.15);
  width: 90%;
  max-width: 640px;
  max-height: 85vh;
  overflow: auto;
  padding: 0;
}
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
}
.modal-header h2 {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}
.modal-close {
  width: 32px;
  height: 32px;
  border: none;
  background: var(--bg);
  border-radius: 50%;
  cursor: pointer;
  font-size: 18px;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: var(--transition);
}
.modal-close:hover {
  background: var(--border);
  color: var(--text-primary);
}
.modal-body {
  padding: 24px;
  display: flex;
  align-items: flex-start;
  gap: 32px;
  flex-wrap: wrap;
}
.modal-pie-wrap {
  position: relative;
  width: 220px;
  height: 220px;
  flex-shrink: 0;
  margin: 0 auto;
}
.modal-pie-wrap svg {
  width: 100%;
  height: 100%;
  overflow: visible;
}
.modal-pie-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  pointer-events: none;
}
.modal-pie-center .num {
  display: block;
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1;
}
.modal-pie-center .lbl {
  display: block;
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 2px;
}
.modal-rule-list {
  flex: 1;
  min-width: 200px;
}
.modal-rule-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-light);
  font-size: 13px;
}
.modal-rule-item:last-child {
  border-bottom: none;
}
.modal-rule-color {
  width: 12px;
  height: 12px;
  border-radius: 3px;
  flex-shrink: 0;
}
.modal-rule-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-primary);
}
.modal-rule-stats {
  display: flex;
  gap: 8px;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
}
.modal-rule-stats .e { color: var(--error); font-weight: 600; }
.modal-rule-stats .w { color: var(--warning); font-weight: 600; }
.modal-rule-stats .s { color: var(--suggestion); font-weight: 600; }
.modal-rule-pct {
  min-width: 40px;
  text-align: right;
  color: var(--text-muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.pie-slice {
  cursor: pointer;
  transition: opacity 0.15s ease;
}
.pie-slice:hover {
  opacity: 0.75;
}
</style>
</head>
<body>
<div class="container">
  <h1>light-subtitle Regression Dashboard</h1>

  <div class="tag-cloud" id="tag-cloud"></div>
  <div class="selected-chips" id="selected-chips" style="display:none"></div>
  <div id="alerts"></div>
  <div id="summary"></div>
  <div class="cases-grid" id="cases-grid"></div>
</div>
<div class="trend-tooltip" id="tooltip"></div>

<!-- Run Detail Modal -->
<div class="modal-overlay" id="modal" onclick="closeModal(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-header">
      <h2 id="modal-title">Run Detail</h2>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <div class="modal-pie-wrap">
        <svg id="modal-pie-svg" viewBox="0 0 200 200"></svg>
        <div class="modal-pie-center">
          <span class="num" id="modal-pie-num">0</span>
          <span class="lbl" id="modal-pie-lbl">issues</span>
        </div>
      </div>
      <div class="modal-rule-list" id="modal-rule-list"></div>
    </div>
  </div>
</div>

<script>
const CASES = """
            + cases_json
            + """;

const selectedCases = new Set();

// ── Tag Cloud & Filtering ─────────────────────────────
function buildTagCloud() {
  const cloud = document.getElementById("tag-cloud");
  if (!cloud) return;
  const allNames = CASES.map(c => c.name);
  cloud.innerHTML = allNames.map(name =>
    `<span class="tag${selectedCases.has(name) ? " selected" : ""}" data-case-name="${name}">${name}</span>`
  ).join("");
  if (selectedCases.size > 0) {
    cloud.innerHTML += `<span class="tag tag-clear" data-action="clear">✕ Clear</span>`;
  }
}

function toggleCase(name) {
  if (selectedCases.has(name)) {
    selectedCases.delete(name);
  } else {
    selectedCases.add(name);
  }
  buildTagCloud();
  applyFilters();
}

function applyFilters() {
  const hasSelection = selectedCases.size > 0;

  let visibleCount = 0;
  let visibleDegraded = 0;
  let visibleRuns = 0;

  document.querySelectorAll(".case-card").forEach(card => {
    const rawName = card.dataset.caseName || "";
    const matchesSelection = !hasSelection || selectedCases.has(rawName);
    card.style.display = matchesSelection ? "" : "none";
    if (matchesSelection) {
      visibleCount++;
      if (card.classList.contains("degraded")) visibleDegraded++;
      visibleRuns += card.querySelectorAll(".runs-table tbody tr").length;
    }
  });

  document.querySelectorAll(".alert").forEach(a => {
    const name = a.dataset.caseName;
    a.style.display = (!hasSelection || selectedCases.has(name)) ? "" : "none";
  });

  // Update summary
  document.getElementById("summary").innerHTML = `
    <div class="summary">
      <div class="summary-card"><span class="number">${visibleCount}</span><span class="label">Test Cases${hasSelection ? " (selected)" : ""}</span></div>
      <div class="summary-card"><span class="number">${visibleRuns}</span><span class="label">Total Runs</span></div>
      <div class="summary-card"><span class="number" style="color:${visibleDegraded > 0 ? "var(--error)" : "var(--success)"}">${visibleDegraded}</span><span class="label">Degraded</span></div>
    </div>
  `;

  // Update chips
  const chipsEl = document.getElementById("selected-chips");
  if (selectedCases.size === 0) {
    chipsEl.style.display = "none";
    chipsEl.innerHTML = "";
  } else {
    chipsEl.style.display = "flex";
    chipsEl.innerHTML = Array.from(selectedCases).map(name =>
      `<span class="chip">${name} <span class="chip-remove" data-name="${name}" data-action="remove-chip">✕</span></span>`
    ).join("");
  }

  // Empty state
  const gridEl = document.getElementById("cases-grid");
  let emptyEl = document.querySelector(".empty-filtered");
  if (visibleCount === 0 && CASES.length > 0) {
    if (!emptyEl) {
      emptyEl = document.createElement("div");
      emptyEl.className = "empty empty-filtered";
      emptyEl.innerHTML = "<h3>No cases match current selection</h3><p>Click a case tag above to select it</p>";
      gridEl.appendChild(emptyEl);
    }
    emptyEl.style.display = "";
  } else if (emptyEl) {
    emptyEl.style.display = "none";
  }
}

function initFilterListeners() {
  document.getElementById("tag-cloud").addEventListener("click", function(e) {
    const tag = e.target.closest(".tag");
    if (!tag) return;
    if (tag.dataset.action === "clear") {
      selectedCases.clear();
      buildTagCloud();
      applyFilters();
      return;
    }
    toggleCase(tag.dataset.caseName);
  });
  document.getElementById("selected-chips").addEventListener("click", function(e) {
    const removeBtn = e.target.closest("[data-action=remove-chip]");
    if (!removeBtn) return;
    selectedCases.delete(removeBtn.dataset.name);
    buildTagCloud();
    applyFilters();
  });
}

function render() {
  const alertsEl = document.getElementById('alerts');
  const summaryEl = document.getElementById('summary');
  const gridEl = document.getElementById('cases-grid');

  if (CASES.length === 0) {
    gridEl.innerHTML = '<div class="empty"><h3>No test cases found</h3><p>Run some tests to see results here</p></div>';
    return;
  }

  summaryEl.innerHTML = '<div class="summary"></div>';

  CASES.forEach((c, ci) => {
    const latest = c.runs[c.runs.length - 1];
    const isDegraded = latest && latest.degraded;

    if (isDegraded) {
      const a = document.createElement('div');
      a.className = 'alert';
      a.dataset.caseName = c.name;
      a.innerHTML = `<h4>⚠️ ${c.name}</h4><ul>${latest.reasons.map(r => `<li>${r}</li>`).join('')}</ul>`;
      alertsEl.appendChild(a);
    }

    const lastErrors = latest ? latest.errors : 0;
    const lastWarnings = latest ? latest.warnings : 0;
    const lastSuggestions = latest ? latest.suggestions : 0;
    const lastTotal = latest ? latest.total : 0;

    const card = document.createElement('div');
    card.className = 'case-card ' + (isDegraded ? 'degraded' : '');
    card.dataset.caseName = c.name;

    card.innerHTML = `
      <div class="case-card-header">
        <h3>${c.name}</h3>
        <div class="status ${isDegraded ? 'degraded' : 'pass'}">${isDegraded ? '✗ Degraded' : '✓ Passing'}</div>
      </div>
      <div class="case-card-body">
        <div class="trend-section">
          <svg class="trend-chart" id="chart-${ci}" viewBox="0 0 400 160"></svg>
          <div class="trend-legend">
            <div class="trend-legend-item" data-series="errors">
              <span class="trend-legend-dot" style="background:var(--error)"></span>
              <span class="trend-legend-label">Errors</span>
              <span class="trend-legend-count">${lastErrors}</span>
            </div>
            <div class="trend-legend-item" data-series="warnings">
              <span class="trend-legend-dot" style="background:var(--warning)"></span>
              <span class="trend-legend-label">Warnings</span>
              <span class="trend-legend-count">${lastWarnings}</span>
            </div>
            <div class="trend-legend-item" data-series="suggestions">
              <span class="trend-legend-dot" style="background:var(--suggestion)"></span>
              <span class="trend-legend-label">Suggestions</span>
              <span class="trend-legend-count">${lastSuggestions}</span>
            </div>
            <div class="trend-legend-item" data-series="total">
              <span class="trend-legend-dot" style="background:#6b7280"></span>
              <span class="trend-legend-label">Total</span>
              <span class="trend-legend-count">${lastTotal}</span>
            </div>
            <div style="margin-left:auto;color:var(--text-muted);font-size:12px;">上一轮: ${c.runs.length > 1 ? c.runs[c.runs.length - 2].run_id : '—'}</div>
          </div>
        </div>
        <table class="runs-table">
          <thead><tr>
            <th>Run</th>
            <th>Errors</th>
            <th>Warnings</th>
            <th>Suggestions</th>
            <th>Total</th>
            <th>Δ Errors</th>
          </tr></thead>
          <tbody>
            ${c.runs.slice(-10).reverse().map(r => `
              <tr>
                <td class="run-id">${r.run_id}</td>
                <td class="val-errors">${r.errors}</td>
                <td class="val-warnings">${r.warnings}</td>
                <td class="val-suggestions">${r.suggestions}</td>
                <td class="val-total">${r.total}</td>
                <td class="${r.errors_delta > 0 ? 'delta-pos' : (r.errors_delta < 0 ? 'delta-neg' : 'delta-zero')}">${r.errors_delta > 0 ? '+' : ''}${r.errors_delta}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;

    gridEl.appendChild(card);

    // Render trend chart
    renderTrend(document.getElementById('chart-' + ci), c.runs, ci);
  });

  buildTagCloud();
  initFilterListeners();
  applyFilters();
}

function renderTrend(svg, runs, caseIdx) {
  const W = 400, H = 160;
  const pad = { top: 12, right: 16, bottom: 24, left: 36 };
  const x0 = pad.left, y0 = pad.top;
  const x1 = W - pad.right, y1 = H - pad.bottom;
  const cw = x1 - x0, ch = y1 - y0;

  const maxVal = Math.max(1, ...runs.map(r => Math.max(r.errors, r.warnings, r.suggestions, r.total)));
  const maxValRounded = Math.ceil(maxVal / 5) * 5 || 5;
  const scaleY = ch / maxValRounded;

  const n = runs.length;
  const stepX = n > 1 ? cw / (n - 1) : 0;

  // Grid lines
  let g = '';
  const gridCount = 4;
  for (let i = 0; i <= gridCount; i++) {
    const v = (i / gridCount) * maxValRounded;
    const y = y1 - v * scaleY;
    g += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="#e5e7eb" stroke-width="1"/>`;
    g += `<text x="${x0 - 4}" y="${y + 4}" text-anchor="end" font-size="9" fill="#9ca3af">${Math.round(v)}</text>`;
  }

  // X tick labels: show some, not all
  const tickStep = Math.max(1, Math.floor(n / 8));
  for (let i = 0; i < n; i += tickStep) {
    const x = x0 + i * stepX;
    g += `<text x="${x}" y="${y1 + 16}" text-anchor="middle" font-size="9" fill="#9ca3af">${i + 1}</text>`;
  }

  // Series definitions
  const series = [
    { key: 'errors', color: '#ef4444', label: 'Errors' },
    { key: 'warnings', color: '#f59e0b', label: 'Warnings' },
    { key: 'suggestions', color: '#8b5cf6', label: 'Suggestions' },
    { key: 'total', color: '#6b7280', label: 'Total' },
  ];

  // Build polyline + dots for each series
  series.forEach(s => {
    if (n === 0) return;

    const pts = runs.map((r, i) => ({ x: x0 + i * stepX, y: y1 - r[s.key] * scaleY }));
    const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p.x + ' ' + p.y).join(' ');
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;

    pts.forEach((p, i) => {
      g += `<circle cx="${p.x}" cy="${p.y}" r="3.5" fill="${s.color}" stroke="#fff" stroke-width="1.5"
        data-run="${i}" data-series="${s.key}" data-errors="${runs[i].errors}"
        data-warnings="${runs[i].warnings}" data-suggestions="${runs[i].suggestions}"
        data-total="${runs[i].total}"
        onmouseenter="onDotEnter(event, '${runs[i].run_id}')" onmouseleave="onDotLeave()"
        onclick="openRunDetail(${caseIdx}, ${i})" style="cursor:pointer"/>`;
    });
  });

  svg.innerHTML = g;
}

function onDotEnter(e, runId) {
  const dot = e.target;
  const errors = dot.getAttribute('data-errors');
  const warnings = dot.getAttribute('data-warnings');
  const suggestions = dot.getAttribute('data-suggestions');
  const total = dot.getAttribute('data-total');

  const tooltip = document.getElementById('tooltip');
  tooltip.innerHTML = `<b>${runId}</b><br>Errors: ${errors} | Warnings: ${warnings} | Suggestions: ${suggestions} | Total: ${total}`;
  tooltip.classList.add('visible');

  const rect = dot.getBoundingClientRect();
  tooltip.style.left = (rect.left + rect.width / 2 - tooltip.offsetWidth / 2) + 'px';
  tooltip.style.top = (rect.top - tooltip.offsetHeight - 8) + 'px';
}

function onDotLeave() {
  document.getElementById('tooltip').classList.remove('visible');
}

// ── Run Detail Modal / Pie Chart ──────────────────────
const PIE_COLORS = [
  '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
  '#06b6d4', '#f97316', '#6366f1', '#14b8a6', '#e11d48',
  '#84cc16', '#d946ef', '#0ea5e9', '#a855f7', '#22c55e'
];

function openRunDetail(caseIdx, runIdx) {
  const run = CASES[caseIdx].runs[runIdx];
  if (!run || !run.rules) return;

  document.getElementById('modal-title').textContent =
    CASES[caseIdx].name + ' — ' + run.run_id;
  document.getElementById('modal').classList.add('open');

  renderPieChart(run.rules, run.total);
}

function closeModal(e) {
  if (e && e.target !== e.currentTarget) return;
  document.getElementById('modal').classList.remove('open');
}

function renderPieChart(rules, total) {
  const svg = document.getElementById('modal-pie-svg');
  const list = document.getElementById('modal-rule-list');
  const entries = Object.entries(rules).sort((a, b) => b[1].total - a[1].total);

  document.getElementById('modal-pie-num').textContent = total;
  document.getElementById('modal-pie-lbl').textContent = total === 1 ? 'issue' : 'issues';

  if (entries.length === 0) {
    svg.innerHTML = '';
    list.innerHTML = '<div style="color:var(--text-muted);text-align:center;">No issues</div>';
    return;
  }

  const cx = 100, cy = 100, outerR = 88, innerR = 52;
  let currentAngle = 0;
  let paths = '';

  entries.forEach(([rule, counts], idx) => {
    const sweep = (counts.total / total) * 360;
    const endAngle = currentAngle + sweep;
    const color = PIE_COLORS[idx % PIE_COLORS.length];
    const start = polarToCartesian(cx, cy, outerR, endAngle);
    const end = polarToCartesian(cx, cy, outerR, currentAngle);
    const iStart = polarToCartesian(cx, cy, innerR, endAngle);
    const iEnd = polarToCartesian(cx, cy, innerR, currentAngle);
    const large = sweep > 180 ? 1 : 0;

    paths += `<path d="M ${start.x} ${start.y} A ${outerR} ${outerR} 0 ${large} 0 ${end.x} ${end.y} L ${iEnd.x} ${iEnd.y} A ${innerR} ${innerR} 0 ${large} 1 ${iStart.x} ${iStart.y} Z"
      fill="${color}" class="pie-slice" data-rule="${rule}"/>`;
    currentAngle = endAngle;
  });

  svg.innerHTML = paths;

  let listHtml = '';
  entries.forEach(([rule, counts], idx) => {
    const color = PIE_COLORS[idx % PIE_COLORS.length];
    const pct = ((counts.total / total) * 100).toFixed(1);
    listHtml += `
      <div class="modal-rule-item">
        <span class="modal-rule-color" style="background:${color}"></span>
        <span class="modal-rule-name" title="${rule}">${rule}</span>
        <span class="modal-rule-stats">
          ${counts.error ? '<span class="e">E' + counts.error + '</span>' : ''}
          ${counts.warning ? '<span class="w">W' + counts.warning + '</span>' : ''}
          ${counts.suggestion ? '<span class="s">S' + counts.suggestion + '</span>' : ''}
        </span>
        <span class="modal-rule-pct">${pct}%</span>
      </div>`;
  });
  list.innerHTML = listHtml;
}

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg - 90) * Math.PI / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

// Close modal on Escape key
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.getElementById('modal').classList.remove('open');
  }
});

render();
</script>
</body>
</html>"""
        )
