#!/usr/bin/env python3
"""Browser dashboard for durable GoTrack progress files.

This dashboard is file-based. It reads the JSON files produced by
``run_gotrack_session.py`` and does not need to share a process with the
tracker.

Usage:
    python scripts/gotrack_progress_dashboard.py --trial-dir <trial>
    python scripts/gotrack_progress_dashboard.py --output-dir <trial>/object_tracking/gotrack_output --open-browser
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.tracking.progress import default_run_index_dir, load_run_index  # noqa: E402

RUN_HISTORY_ENRICH_LIMIT = 100


def _out_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    if args.trial_dir:
        return Path(args.trial_dir).expanduser() / "object_tracking/gotrack_output"
    raise SystemExit("Provide --trial-dir or --output-dir")


def _run_index_dir(args: argparse.Namespace) -> Path:
    return Path(args.run_index_dir).expanduser() if args.run_index_dir else default_run_index_dir()


def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": repr(exc), "_path": str(path)}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_events(path: Path, limit: int = 30) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    out: List[Dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"event": "corrupt_event", "raw": line})
    return out


def _default_overlay_dir_from_output(output_dir: Any) -> Path | None:
    if not output_dir:
        return None
    out = Path(str(output_dir)).expanduser()
    if out.name == "gotrack_output" and out.parent.name == "object_tracking":
        return out.parent / "overlay_check"
    return None


def _overlay_files_from_dir(path: Any) -> List[Dict[str, Any]]:
    if not path:
        return []
    root = Path(str(path)).expanduser()
    try:
        root = root.resolve()
    except OSError:
        return []
    if not root.is_dir():
        return []
    files: List[Dict[str, Any]] = []
    for video in sorted(root.glob("overlay_*.mp4")):
        try:
            stat = video.stat()
            resolved = video.resolve()
        except OSError:
            continue
        files.append({
            "name": video.name,
            "path": str(resolved),
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        })
    return files


def _load_overlay_status(path: Any) -> Dict[str, Any]:
    if not path:
        return {}
    return _load_json(Path(str(path)).expanduser() / "overlay_status.json", {})


def _overlay_dirs_for_record(record: Dict[str, Any]) -> List[Path]:
    dirs: List[Path] = []
    explicit = record.get("overlay_output_dir")
    if explicit:
        dirs.append(Path(str(explicit)).expanduser())
    inferred = _default_overlay_dir_from_output(record.get("output_dir"))
    if inferred is not None:
        dirs.append(inferred)
    seen = set()
    unique: List[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _enrich_overlay(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    overlay_files: List[Dict[str, Any]] = []
    for item in out.get("overlay_files") or []:
        if isinstance(item, dict) and item.get("path"):
            overlay_files.append(dict(item))
        elif isinstance(item, str):
            path = Path(item).expanduser()
            try:
                stat = path.stat()
                resolved = path.resolve()
            except OSError:
                continue
            overlay_files.append({
                "name": path.name,
                "path": str(resolved),
                "size_bytes": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            })
    for overlay_dir in _overlay_dirs_for_record(out):
        for item in _overlay_files_from_dir(overlay_dir):
            if item["path"] not in {f.get("path") for f in overlay_files if isinstance(f, dict)}:
                overlay_files.append(item)
        status = _load_overlay_status(overlay_dir)
        if status and not out.get("overlay_status"):
            out["overlay_status"] = status.get("status")
        if status and not out.get("overlay_output_dir"):
            out["overlay_output_dir"] = status.get("output_dir")
    if overlay_files:
        out["overlay_files"] = overlay_files
        out["overlay_n_outputs"] = len(overlay_files)
    return out


def _enrich_run_index(run_index: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(run_index)
    runs = data.get("runs") or {}
    if isinstance(runs, dict):
        items = list(runs.items())
        items.sort(
            key=lambda item: float(item[1].get("updated_at", 0))
            if isinstance(item[1], dict) else 0.0,
            reverse=True,
        )
        enriched = {}
        for idx, (run_id, run) in enumerate(items):
            if isinstance(run, dict) and idx < RUN_HISTORY_ENRICH_LIMIT:
                enriched[run_id] = _enrich_overlay(run)
            else:
                enriched[run_id] = run
        data["runs"] = enriched
    return data


def collect(out_dir: Path, run_index_dir: Path) -> Dict[str, Any]:
    state = _load_json(out_dir / "state.json", {})
    pc_status = _load_json(out_dir / "capture_pc_status.json", {"pcs": {}})
    summary = _load_json(out_dir / "summary.json", {})
    manifest = _load_json(out_dir / "run_manifest.json", {})
    events = _load_events(out_dir / "events.jsonl")
    run_index = _enrich_run_index(load_run_index(run_index_dir))
    poses_live = out_dir / "world_pose_records.jsonl"
    poses_final = out_dir / "world_pose_records.json"
    state_overlay = _as_dict(state.get("overlay_check"))
    summary_overlay = _as_dict(summary.get("overlay_check"))
    current_overlay = _enrich_overlay({
        "output_dir": str(out_dir),
        "overlay_output_dir": (
            state_overlay.get("output_dir")
            or summary_overlay.get("output_dir")
            or state.get("overlay_output_dir")
            or summary.get("overlay_output_dir")
        ),
        "overlay_status": (
            state_overlay.get("status")
            or summary_overlay.get("status")
            or state.get("overlay_status")
            or summary.get("overlay_status")
        ),
    })
    return {
        "now": time.time(),
        "out_dir": str(out_dir),
        "state": state,
        "pc_status": pc_status,
        "summary": summary,
        "manifest": manifest,
        "events": events,
        "run_index": run_index,
        "files": {
            "poses_live": str(poses_live),
            "poses_live_exists": poses_live.exists(),
            "poses_final": str(poses_final),
            "poses_final_exists": poses_final.exists(),
        },
        "overlay": current_overlay,
    }


def allowed_overlay_paths(out_dir: Path, run_index_dir: Path) -> Dict[str, Path]:
    data = collect(out_dir, run_index_dir)
    allowed: Dict[str, Path] = {}
    records: List[Dict[str, Any]] = []
    overlay = data.get("overlay")
    if isinstance(overlay, dict):
        records.append(overlay)
    runs = ((data.get("run_index") or {}).get("runs") or {})
    if isinstance(runs, dict):
        records.extend(run for run in runs.values() if isinstance(run, dict))
    for record in records:
        for item in record.get("overlay_files") or []:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            try:
                path = Path(str(item["path"])).expanduser().resolve()
            except OSError:
                continue
            if path.name.startswith("overlay_") and path.suffix.lower() == ".mp4" and path.is_file():
                allowed[str(path)] = path
    return allowed


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoTrack Progress</title>
<style>
:root {
  color-scheme: dark;
  --bg: #111315;
  --panel: #1b1f22;
  --panel2: #22272b;
  --line: #31383e;
  --text: #e6edf3;
  --muted: #8b949e;
  --green: #3fb950;
  --yellow: #d29922;
  --red: #f85149;
  --blue: #58a6ff;
  --gray: #6e7681;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 13px/1.4 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  background: #15191c;
}
h1 { font-size: 17px; margin: 0; font-weight: 650; letter-spacing: 0; }
.sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
main { padding: 16px 18px 22px; max-width: 1280px; margin: 0 auto; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.metric-label { color: var(--muted); font-size: 12px; }
.metric-value { margin-top: 4px; font-size: 24px; font-weight: 700; letter-spacing: 0; }
.metric-small { color: var(--muted); margin-top: 4px; font-size: 12px; overflow-wrap: anywhere; }
.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 18px 0 8px;
}
h2 { font-size: 14px; margin: 0; color: #c9d1d9; font-weight: 650; }
.lifecycle {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 8px;
}
.step {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 42px;
  padding: 9px 10px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.step .dot { width: 10px; height: 10px; }
.step .name { font-weight: 650; }
.step .meta { color: var(--muted); font-size: 12px; }
.step.complete .dot { background: var(--green); }
.step.running .dot { background: var(--yellow); }
.step.failed .dot { background: var(--red); }
.step.pending .dot { background: var(--gray); }
table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-size: 12px; font-weight: 500; background: var(--panel2); }
tr:last-child td { border-bottom: 0; }
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 7px;
  border-radius: 999px;
  background: #2a3035;
  color: var(--muted);
  font-weight: 600;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--gray); flex: none; }
.running .dot, .complete .dot, .done .dot, .ok .dot, .skipped_done .dot, .dry_run_done .dot { background: var(--green); }
.stale .dot, .init_sent .dot, .start_sent .dot, .stopping .dot { background: var(--yellow); }
.failed .dot, .daemon_missing .dot { background: var(--red); }
.not_started .dot { background: var(--gray); }
.bar {
  width: 100%;
  height: 8px;
  background: #30363d;
  border-radius: 99px;
  overflow: hidden;
}
.bar > div { height: 100%; width: 0%; background: var(--green); transition: width .2s ease; }
.bar.slim { width: 96px; height: 6px; display: inline-block; vertical-align: middle; margin-right: 8px; }
.bar.warn > div { background: var(--yellow); }
.bar.bad > div { background: var(--red); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.quality-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 10px;
}
.forecast-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 10px;
}
.forecast-item {
  min-height: 76px;
  padding: 10px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.quality-item {
  min-height: 76px;
  padding: 10px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.quality-label, .forecast-label { color: var(--muted); font-size: 12px; }
.quality-value, .forecast-value { margin-top: 6px; font-size: 16px; font-weight: 650; overflow-wrap: anywhere; }
.quality-detail, .forecast-detail { color: var(--muted); margin-top: 5px; font-size: 12px; overflow-wrap: anywhere; }
.events {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 4px 0;
  max-height: 280px;
  overflow: auto;
}
.event { display: grid; grid-template-columns: 82px 180px 1fr; gap: 10px; padding: 6px 10px; border-bottom: 1px solid var(--line); }
.event:last-child { border-bottom: 0; }
.muted { color: var(--muted); }
.bad { color: var(--red); }
.ok { color: var(--green); }
.warn { color: var(--yellow); }
.overlay-links { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.overlay-button {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--blue);
  cursor: pointer;
  padding: 0;
  font: inherit;
}
.overlay-button:hover { text-decoration: underline; }
.viewer-backdrop {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(0, 0, 0, 0.72);
}
.viewer-backdrop.open { display: flex; }
.viewer {
  width: min(1120px, 96vw);
  max-height: 92vh;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55);
}
.viewer-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  background: var(--panel2);
}
.viewer-title { min-width: 0; }
.viewer-title h2 { margin-bottom: 2px; }
.viewer-close {
  flex: none;
  width: 30px;
  height: 30px;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: #30363d;
  color: var(--text);
  cursor: pointer;
  font: inherit;
}
.viewer-close:hover { background: #3b434a; }
.viewer video {
  width: 100%;
  max-height: 74vh;
  display: block;
  background: #000;
}
.viewer-meta {
  padding: 8px 12px 10px;
  color: var(--muted);
  border-top: 1px solid var(--line);
  overflow-wrap: anywhere;
}
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
@media (max-width: 900px) {
  .grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
  .lifecycle, .quality-grid, .forecast-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
  table { font-size: 12px; }
  th, td { padding: 7px 8px; }
  .event { grid-template-columns: 74px 1fr; }
  .event .detail { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>GoTrack Progress</h1>
    <div class="sub" id="subtitle">loading</div>
  </div>
  <div class="sub mono" id="clock">-</div>
</header>
<main>
  <div class="section-title">
    <h2>Current Run Snapshot</h2>
    <span class="sub" id="snapshotMeta">state.json</span>
  </div>
  <div class="grid">
    <div class="panel">
      <div class="metric-label">Run State</div>
      <div class="metric-value" id="phase">-</div>
      <div class="metric-small" id="objtrial">-</div>
    </div>
    <div class="panel">
      <div class="metric-label">Pose Records</div>
      <div class="metric-value" id="frames">-</div>
      <div class="metric-small" id="received">-</div>
    </div>
    <div class="panel">
      <div class="metric-label">Tracker FPS</div>
      <div class="metric-value" id="fps">0.00</div>
      <div class="metric-small" id="lastframe">last frame -</div>
    </div>
    <div class="panel">
      <div class="metric-label">Fit Success</div>
      <div class="metric-value" id="ratio">-</div>
      <div class="bar" style="margin-top:8px"><div id="ratioBar"></div></div>
    </div>
  </div>

  <div class="section-title">
    <h2>Runtime Forecast</h2>
    <span class="sub" id="forecastMeta">requires max-frames or max-seconds for ETA</span>
  </div>
  <div class="forecast-grid" id="runtimeForecast"></div>

  <div class="section-title">
    <h2>Lifecycle</h2>
    <span class="sub" id="lifecycleMeta">phase order</span>
  </div>
  <div class="lifecycle" id="lifecycleSteps"></div>

  <div class="section-title">
    <h2>Distributed Capture Health</h2>
    <span class="sub" id="pcUpdated">-</span>
  </div>
  <table>
    <thead>
      <tr>
        <th>PC</th><th>Status</th><th>IP</th><th>Last Frame</th><th>Age</th><th>Frames</th><th>Frame Share</th><th>Daemon</th><th>Command</th>
      </tr>
    </thead>
    <tbody id="pcRows"><tr><td colspan="9" class="muted">loading</td></tr></tbody>
  </table>

  <div class="section-title">
    <h2>Tracking Quality Signals</h2>
    <span class="sub" id="files">-</span>
  </div>
  <div class="quality-grid" id="qualitySignals"></div>

  <div class="section-title">
    <h2>Accumulated Run History</h2>
    <span class="sub" id="historyMeta">runs_latest.json</span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Status</th><th>Object</th><th>Trial</th><th>Frames</th><th>Overlay</th><th>PCs</th><th>Updated</th><th>Output</th>
      </tr>
    </thead>
    <tbody id="runRows"><tr><td colspan="8" class="muted">loading</td></tr></tbody>
  </table>

  <div class="section-title">
    <h2>Current Run Event Log</h2>
    <span class="sub">events.jsonl tail</span>
  </div>
  <div class="events mono" id="events"></div>
</main>
<div class="viewer-backdrop" id="overlayViewer" aria-hidden="true">
  <div class="viewer" role="dialog" aria-modal="true" aria-labelledby="overlayViewerTitle">
    <div class="viewer-head">
      <div class="viewer-title">
        <h2 id="overlayViewerTitle">Overlay Playback</h2>
        <div class="sub mono" id="overlayViewerSub">-</div>
      </div>
      <button class="viewer-close" type="button" id="overlayClose" aria-label="Close overlay video">x</button>
    </div>
    <video id="overlayVideo" controls playsinline></video>
    <div class="viewer-meta mono" id="overlayMeta">-</div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const safe = v => (v === undefined || v === null || v === '') ? '-' : String(v);
const cls = s => safe(s).replaceAll('_', '-').replace(/[^a-zA-Z0-9-]/g, '');
const statusClass = s => safe(s).replace(/[^a-zA-Z0-9_]/g, '_');
const fmtAge = v => (v === undefined || v === null) ? '-' : Number(v).toFixed(2) + 's';
const fmtBytes = v => {
  const n = Number(v || 0);
  if (!n) return '';
  if (n < 1024 * 1024) return Math.ceil(n / 1024) + ' KB';
  return (n / 1024 / 1024).toFixed(1) + ' MB';
};
const fmtTime = ts => {
  if (!ts) return '-';
  const d = new Date(Number(ts) * 1000);
  return d.toLocaleTimeString();
};
const numberOrNull = v => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const fmtDuration = seconds => {
  const s0 = numberOrNull(seconds);
  if (s0 === null) return '-';
  const s = Math.max(0, Math.round(s0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(r).padStart(2, '0')}s`;
  return `${r}s`;
};
const esc = s => safe(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const donePhases = new Set(['done', 'ok', 'complete', 'skipped_done', 'dry_run_done']);
const failedPhases = new Set(['failed', 'failed_no_pose_records', 'daemon_missing']);
const phaseOrder = ['preflight', 'preflight_done', 'daemon_init', 'daemon_start', 'tracking', 'stopping', 'overlay_check', 'done'];

function phaseIndex(phase) {
  const idx = phaseOrder.indexOf(String(phase || ''));
  if (donePhases.has(String(phase || ''))) return phaseOrder.length;
  if (failedPhases.has(String(phase || ''))) return phaseOrder.length;
  return idx < 0 ? -1 : idx;
}

function stepState(step, phase, overlayStatus) {
  const p = String(phase || '');
  const idx = phaseIndex(p);
  if (failedPhases.has(p)) return step === 'Complete' ? 'failed' : 'complete';
  if (step === 'Preflight') {
    if (idx < 0) return 'pending';
    if (p === 'preflight') return 'running';
    return 'complete';
  }
  if (step === 'Daemon Init') {
    if (['daemon_init', 'daemon_start'].includes(p)) return 'running';
    return idx > phaseIndex('daemon_start') || donePhases.has(p) ? 'complete' : 'pending';
  }
  if (step === 'Tracking') {
    if (p === 'tracking' || p === 'stopping') return 'running';
    return idx > phaseIndex('stopping') || donePhases.has(p) ? 'complete' : 'pending';
  }
  if (step === 'Overlay') {
    if (overlayStatus === 'failed') return 'failed';
    if (overlayStatus === 'ok') return 'complete';
    if (p === 'overlay_check') return 'running';
    return 'pending';
  }
  if (step === 'Complete') {
    if (failedPhases.has(p)) return 'failed';
    if (donePhases.has(p)) return 'complete';
    return 'pending';
  }
  return 'pending';
}

function renderLifecycle(phase, overlayStatus) {
  const steps = ['Preflight', 'Daemon Init', 'Tracking', 'Overlay', 'Complete'];
  const meta = {
    'Preflight': 'inputs',
    'Daemon Init': '5 PCs',
    'Tracking': 'poses',
    'Overlay': overlayStatus || 'check',
    'Complete': 'summary',
  };
  $('lifecycleSteps').innerHTML = steps.map(step => {
    const state = stepState(step, phase, overlayStatus);
    return `<div class="step ${state}">
      <span class="dot"></span>
      <div><div class="name">${esc(step)}</div><div class="meta">${esc(state)} · ${esc(meta[step])}</div></div>
    </div>`;
  }).join('');
  $('lifecycleMeta').textContent = 'current phase ' + safe(phase);
}

function qualityItem(label, value, detail, clsName='') {
  return `<div class="quality-item ${clsName}">
    <div class="quality-label">${esc(label)}</div>
    <div class="quality-value">${esc(value)}</div>
    <div class="quality-detail">${esc(detail)}</div>
  </div>`;
}

function forecastItem(label, value, detail, clsName='') {
  return `<div class="forecast-item ${clsName}">
    <div class="forecast-label">${esc(label)}</div>
    <div class="forecast-value">${esc(value)}</div>
    <div class="forecast-detail">${esc(detail)}</div>
  </div>`;
}

function targetText(limits) {
  const maxFrames = numberOrNull(limits.max_frames);
  const maxSeconds = numberOrNull(limits.max_seconds);
  const parts = [];
  if (maxFrames !== null && maxFrames > 0) parts.push(`${maxFrames} frames`);
  if (maxSeconds !== null && maxSeconds > 0) parts.push(fmtDuration(maxSeconds));
  return parts.length ? parts.join(' or ') : 'external stop';
}

function computeForecast(data, ok, phase) {
  const st = data.state || {};
  const summary = data.summary || {};
  const manifest = data.manifest || {};
  const limits = st.limits || manifest.limits || {};
  const now = numberOrNull(data.now) ?? Date.now() / 1000;
  const startedAt = numberOrNull(st.started_at) ?? numberOrNull(manifest.created_at) ?? numberOrNull(summary.started_at);
  const finishedAt = numberOrNull(summary.finished_at) ?? numberOrNull(st.finished_at);
  const terminal = donePhases.has(String(phase || '')) || failedPhases.has(String(phase || ''));
  const effectiveNow = terminal && finishedAt !== null ? finishedAt : now;
  const elapsed = startedAt !== null ? Math.max(0, effectiveNow - startedAt) : null;
  const maxFrames = numberOrNull(limits.max_frames);
  const maxSeconds = numberOrNull(limits.max_seconds);
  const frameTarget = maxFrames !== null && maxFrames > 0 ? maxFrames : null;
  const timeTarget = maxSeconds !== null && maxSeconds > 0 ? maxSeconds : null;
  const fps = numberOrNull(st.fps) ?? numberOrNull(summary.throughput_success_fps) ?? null;
  const measuredRate = elapsed !== null && elapsed > 0 && ok > 0 ? ok / elapsed : null;
  const rate = fps !== null && fps > 0 ? fps : measuredRate;

  const remainingFrames = frameTarget !== null ? Math.max(0, frameTarget - ok) : null;
  const frameEta = remainingFrames !== null && rate !== null && rate > 0 ? remainingFrames / rate : null;
  const timeRemaining = timeTarget !== null && elapsed !== null ? Math.max(0, timeTarget - elapsed) : null;
  const candidates = [frameEta, timeRemaining].filter(v => v !== null);
  const remaining = terminal ? 0 : (candidates.length ? Math.min(...candidates) : null);
  const progressByFrame = frameTarget !== null ? Math.max(0, Math.min(100, 100 * ok / frameTarget)) : null;
  const progressByTime = timeTarget !== null && elapsed !== null ? Math.max(0, Math.min(100, 100 * elapsed / timeTarget)) : null;
  const progressCandidates = [progressByFrame, progressByTime].filter(v => v !== null);
  const progress = terminal ? 100 : (progressCandidates.length ? Math.max(...progressCandidates) : null);

  return {
    limits,
    terminal,
    elapsed,
    remaining,
    frameTarget,
    timeTarget,
    remainingFrames,
    rate,
    progress,
  };
}

function renderForecast(data, ok, phase) {
  const fc = computeForecast(data, ok, phase);
  const target = targetText(fc.limits);
  const etaValue = fc.remaining === null ? (fc.terminal ? '0s' : 'open-ended') : fmtDuration(fc.remaining);
  const etaDetail = fc.remaining === null
    ? 'set --max-frames or --max-seconds for ETA'
    : (fc.terminal ? 'run finished' : 'estimated from current FPS/elapsed time');
  const rateValue = fc.rate !== null && fc.rate > 0 ? fc.rate.toFixed(2) + ' fps' : '-';
  const progressValue = fc.progress !== null ? fc.progress.toFixed(1) + '%' : '-';
  const remainingFrames = fc.remainingFrames === null ? '-' : String(fc.remainingFrames);
  $('forecastMeta').textContent = 'target ' + target;
  $('runtimeForecast').innerHTML = [
    forecastItem('Elapsed', fmtDuration(fc.elapsed), fc.terminal ? 'final runtime' : 'since session start'),
    forecastItem('Remaining ETA', etaValue, etaDetail),
    forecastItem('Target Progress', progressValue, `target: ${target}`),
    forecastItem('Throughput', rateValue, `remaining frames: ${remainingFrames}`),
  ].join('');
}

function overlayCell(r) {
  const files = (r.overlay_files || []).filter(f => f && f.path);
  const status = r.overlay_status || (files.length ? 'ok' : '-');
  if (!files.length) return `<span class="mono">${esc(status)}</span>`;
  const links = files.slice(0, 4).map((f, i) => {
    const serial = (f.name || ('video_' + (i + 1))).replace(/^overlay_/, '').replace(/\.mp4$/i, '');
    const size = fmtBytes(f.size_bytes);
    const label = serial + (size ? ' ' + size : '');
    return `<button type="button" class="overlay-button mono" data-overlay-path="${esc(f.path)}" data-overlay-label="${esc(serial)}" data-overlay-size="${esc(size)}" title="${esc(f.path)}">${esc(label)}</button>`;
  }).join('');
  const more = files.length > 4 ? `<span class="muted">+${files.length - 4}</span>` : '';
  return `<div class="overlay-links"><span class="mono">${esc(status)}</span>${links}${more}</div>`;
}

function overlayUrl(path) {
  return '/overlay?path=' + encodeURIComponent(path);
}

function openOverlayVideo(path, label, size) {
  const title = label ? 'Overlay ' + label : 'Overlay Playback';
  $('overlayViewerTitle').textContent = title;
  $('overlayViewerSub').textContent = size ? size : '-';
  $('overlayMeta').textContent = path || '-';
  const video = $('overlayVideo');
  video.pause();
  video.src = overlayUrl(path);
  video.load();
  $('overlayViewer').classList.add('open');
  $('overlayViewer').setAttribute('aria-hidden', 'false');
  const play = video.play();
  if (play && play.catch) play.catch(() => {});
}

function closeOverlayVideo() {
  const video = $('overlayVideo');
  video.pause();
  video.removeAttribute('src');
  video.load();
  $('overlayViewer').classList.remove('open');
  $('overlayViewer').setAttribute('aria-hidden', 'true');
}

function render(data) {
  const st = data.state || {};
  const summary = data.summary || {};
  const manifest = data.manifest || {};
  const pcs = ((data.pc_status || {}).pcs) || {};
  const phase = st.phase || summary.status || 'unknown';
  const obj = st.obj || manifest.obj || '-';
  const trialDir = st.trial_dir || manifest.trial_dir || '';
  const trial = trialDir ? trialDir.split('/').filter(Boolean).pop() : '-';
  const ok = Number(st.frames_success ?? summary.frames_success ?? 0);
  const received = Number(st.frames_received ?? ok);
  const failed = Number(st.frames_failed ?? Math.max(0, received - ok));
  const ratio = received > 0 ? (100 * ok / received) : 0;
  const overlayStatus = ((st.overlay_check || {}).status) || ((summary.overlay_check || {}).status) || st.overlay_status || summary.overlay_status || '';

  $('clock').textContent = new Date().toLocaleTimeString();
  $('subtitle').textContent = data.out_dir || '-';
  $('snapshotMeta').textContent = fmtTime(st.updated_at || summary.finished_at || data.now);
  $('phase').textContent = phase;
  $('objtrial').textContent = obj + ' / ' + trial;
  $('frames').textContent = ok + ' ok';
  $('received').textContent = received + ' received / ' + failed + ' failed';
  $('fps').textContent = Number(st.fps || 0).toFixed(2);
  $('lastframe').textContent = 'last frame ' + safe(st.last_frame_id ?? summary.last_frame_index);
  $('ratio').textContent = received > 0 ? ratio.toFixed(1) + '%' : '-';
  $('ratioBar').style.width = Math.max(0, Math.min(100, ratio)) + '%';
  renderForecast(data, ok, phase);
  renderLifecycle(phase, overlayStatus);

  const pcUpdated = (data.pc_status || {}).updated_at;
  $('pcUpdated').textContent = 'updated ' + fmtTime(pcUpdated);
  const names = Object.keys(pcs).sort();
  const maxPcFrames = Math.max(...names.map(pc => Number((pcs[pc] || {}).frames_received || 0)), 0);
  if (!names.length) {
    $('pcRows').innerHTML = '<tr><td colspan="9" class="muted">no capture_pc_status.json yet</td></tr>';
  } else {
    $('pcRows').innerHTML = names.map(pc => {
      const p = pcs[pc] || {};
      const status = p.status || p.phase || '-';
      const cmd = p.last_command_result ? (p.last_command_result.ok ? 'ok' : 'error') : '-';
      const pcFrames = Number(p.frames_received || 0);
      const share = maxPcFrames > 0 ? Math.max(0, Math.min(100, 100 * pcFrames / maxPcFrames)) : 0;
      const shareClass = share < 75 ? 'bad' : (share < 95 ? 'warn' : '');
      return `<tr>
        <td class="mono">${esc(pc)}</td>
        <td><span class="status ${statusClass(status)}"><span class="dot"></span>${esc(status)}</span></td>
        <td class="mono">${esc(p.ip)}</td>
        <td class="mono">${esc(p.last_frame_id)}</td>
        <td class="mono">${esc(fmtAge(p.last_obs_age_s))}</td>
        <td class="mono">${esc(pcFrames)}</td>
        <td class="mono"><span class="bar slim ${shareClass}"><div style="width:${share}%"></div></span>${share.toFixed(0)}%</td>
        <td class="mono">${esc(p.daemon_count)}</td>
        <td class="${cmd === 'error' ? 'bad' : ''}">${esc(cmd)}</td>
      </tr>`;
    }).join('');
  }

  const fail = st.fail_by_reason || {};
  const failKeys = Object.keys(fail).sort();
  const f = data.files || {};
  $('files').textContent = f.poses_final_exists ? f.poses_final : f.poses_live;
  const failureText = failKeys.length ? failKeys.map(k => k + '=' + fail[k]).join('  ') : 'none';
  const lastFit = st.last_fit_ok;
  const lastFitValue = lastFit === undefined || lastFit === null ? '-' : (lastFit ? 'ok' : 'failed');
  const residual = st.mean_residual_mm ?? summary.mean_residual_mm;
  const residualValue = residual === undefined || residual === null ? '-' : Number(residual).toFixed(2) + ' mm';
  $('qualitySignals').innerHTML = [
    qualityItem('Last Fit', lastFitValue, st.fail_reason || 'latest tracker solve'),
    qualityItem('Mean Residual', residualValue, 'anchor fit residual'),
    qualityItem('Failure Reasons', failureText, failKeys.length ? 'aggregated frame failures' : 'no active frame failure'),
    qualityItem('Pose Output', f.poses_final_exists ? 'finalized' : (f.poses_live_exists ? 'live' : 'pending'), f.poses_final_exists ? f.poses_final : f.poses_live),
  ].join('');

  const runIndex = data.run_index || {};
  const runsObj = runIndex.runs || {};
  const runs = Object.values(runsObj).sort((a, b) => Number(b.updated_at || 0) - Number(a.updated_at || 0)).slice(0, 20);
  $('historyMeta').textContent = (runIndex.index_dir || 'runs_latest.json') + ' · ' + runs.length + ' shown';
  if (!runs.length) {
    $('runRows').innerHTML = '<tr><td colspan="8" class="muted">no accumulated runs yet</td></tr>';
  } else {
    $('runRows').innerHTML = runs.map(r => {
      const status = r.status || r.phase || '-';
      const pcCounts = r.pc_status_counts || {};
      const pcs = Object.keys(pcCounts).sort().map(k => k + '=' + pcCounts[k]).join(' ');
      const trial = r.trial_name || (r.trial_dir ? r.trial_dir.split('/').filter(Boolean).pop() : '-');
      const frames = r.frames_success ?? '-';
      return `<tr>
        <td><span class="status ${statusClass(status)}"><span class="dot"></span>${esc(status)}</span></td>
        <td>${esc(r.obj)}</td>
        <td class="mono">${esc(trial)}</td>
        <td class="mono">${esc(frames)}</td>
        <td>${overlayCell(r)}</td>
        <td class="mono">${esc(pcs || '-')}</td>
        <td class="mono">${esc(fmtTime(r.updated_at))}</td>
        <td class="mono">${esc(r.output_dir)}</td>
      </tr>`;
    }).join('');
  }

  const events = data.events || [];
  $('events').innerHTML = events.length ? events.slice().reverse().map(e => {
    const detail = {...e};
    delete detail.ts;
    delete detail.event;
    return `<div class="event">
      <div class="muted">${esc(fmtTime(e.ts))}</div>
      <div>${esc(e.event)}</div>
      <div class="detail muted">${esc(JSON.stringify(detail))}</div>
    </div>`;
  }).join('') : '<div class="event"><div class="muted">-</div><div>no events</div><div></div></div>';
}

async function tick() {
  try {
    const res = await fetch('/api/status', {cache: 'no-store'});
    render(await res.json());
  } catch (err) {
    $('subtitle').textContent = 'fetch error: ' + err;
  }
}

document.addEventListener('click', ev => {
  const target = ev.target instanceof Element ? ev.target : null;
  const btn = target ? target.closest('[data-overlay-path]') : null;
  if (btn) {
    ev.preventDefault();
    openOverlayVideo(btn.dataset.overlayPath, btn.dataset.overlayLabel, btn.dataset.overlaySize);
    return;
  }
  if (target === $('overlayViewer') || target === $('overlayClose')) {
    closeOverlayVideo();
  }
});

document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape' && $('overlayViewer').classList.contains('open')) {
    closeOverlayVideo();
  }
});

tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


def make_handler(out_dir: Path, run_index_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                body = json.dumps(collect(out_dir, run_index_dir), sort_keys=True, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/overlay":
                self._serve_overlay(parsed)
                return
            if parsed.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_overlay(self, parsed) -> None:
            target = parse_qs(parsed.query).get("path", [""])[0]
            if not target:
                self.send_error(400, "missing path")
                return
            try:
                requested = Path(target).expanduser().resolve()
            except OSError:
                self.send_error(400, "invalid path")
                return

            allowed = allowed_overlay_paths(out_dir, run_index_dir)
            video = allowed.get(str(requested))
            if video is None:
                self.send_error(403, "overlay file is not in dashboard run history")
                return
            if not video.is_file():
                self.send_error(404, "overlay video not found")
                return

            try:
                size = video.stat().st_size
            except OSError:
                self.send_error(404, "overlay video not readable")
                return

            start = 0
            end = size - 1
            status_code = 200
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    unit, spec = range_header.split("=", 1)
                    if unit.strip().lower() != "bytes":
                        raise ValueError("unsupported range unit")
                    spec = spec.split(",", 1)[0].strip()
                    if spec.startswith("-"):
                        length = int(spec[1:])
                        start = max(0, size - length)
                    else:
                        left, _, right = spec.partition("-")
                        start = int(left)
                        if right:
                            end = int(right)
                    end = min(end, size - 1)
                    if start < 0 or start > end:
                        raise ValueError("invalid range")
                    status_code = 206
                except Exception:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return

            content_len = end - start + 1
            content_type = mimetypes.guess_type(video.name)[0] or "video/mp4"
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_len))
            if status_code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            with video.open("rb") as f:
                f.seek(start)
                remaining = content_len
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, *_):
            pass

    return Handler


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trial-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--run-index-dir", default=None,
                   help="Default: ~/shared_data/AutoDex/object_tracking/gotrack_runs")
    p.add_argument("--open-browser", action="store_true")
    args = p.parse_args()

    out_dir = _out_dir(args)
    run_index_dir = _run_index_dir(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(out_dir, run_index_dir))
    url = f"http://{args.host}:{args.port}/"
    print(url, flush=True)
    print(f"reading {out_dir}", flush=True)
    print(f"run index {run_index_dir}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
