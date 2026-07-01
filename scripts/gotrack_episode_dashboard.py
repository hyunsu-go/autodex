#!/usr/bin/env python3
"""Browser dashboard for episode-level offline GoTrack scheduling."""
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

from autodex.tracking.episode_queue import EpisodeScheduleStore, summarize_schedule  # noqa: E402


def _load_events(path: Path, limit: int = 40) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"event": "corrupt_event", "raw": line})
    return out


def collect(schedule_dir: Path) -> Dict[str, Any]:
    store = EpisodeScheduleStore.open(schedule_dir)
    data = summarize_schedule(store)
    data["events"] = _load_events(store.events_path)
    return data


def allowed_overlay_paths(schedule_dir: Path) -> Dict[str, Path]:
    data = collect(schedule_dir)
    allowed: Dict[str, Path] = {}
    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue
        for item in task.get("overlay_files", []):
            try:
                path = Path(str(item)).expanduser().resolve()
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
<title>GoTrack Episode Queue</title>
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
h2 { font-size: 14px; margin: 0; color: #c9d1d9; font-weight: 650; }
.sub { color: var(--muted); font-size: 12px; margin-top: 2px; overflow-wrap: anywhere; }
main { padding: 16px 18px 22px; max-width: 1400px; margin: 0 auto; }
.section-title { display: flex; align-items: center; justify-content: space-between; margin: 18px 0 8px; }
.grid { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 10px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.metric-label { color: var(--muted); font-size: 12px; }
.metric-value { margin-top: 4px; font-size: 22px; font-weight: 700; letter-spacing: 0; }
.metric-small { color: var(--muted); margin-top: 4px; font-size: 12px; overflow-wrap: anywhere; }
.bar { width: 100%; height: 8px; background: #30363d; border-radius: 99px; overflow: hidden; }
.bar > div { height: 100%; width: 0%; background: var(--green); transition: width .2s ease; }
.bar.slim { width: 86px; height: 6px; display: inline-block; vertical-align: middle; margin-right: 8px; }
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
.done .dot, .skipped_done .dot, .dry_run_done .dot, .complete .dot { background: var(--green); }
.running .dot, .pending .dot, .claimed .dot, .gotrack .dot, .overlay .dot { background: var(--yellow); }
.failed .dot { background: var(--red); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.muted { color: var(--muted); }
.bad { color: var(--red); }
.ok { color: var(--green); }
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
.events {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 4px 0;
  max-height: 260px;
  overflow: auto;
}
.event { display: grid; grid-template-columns: 82px 170px 1fr; gap: 10px; padding: 6px 10px; border-bottom: 1px solid var(--line); }
.event:last-child { border-bottom: 0; }
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
.viewer video { width: 100%; max-height: 74vh; display: block; background: #000; }
.viewer-meta { padding: 8px 12px 10px; color: var(--muted); border-top: 1px solid var(--line); overflow-wrap: anywhere; }
@media (max-width: 1000px) {
  .grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
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
    <h1>GoTrack Episode Queue</h1>
    <div class="sub" id="subtitle">loading</div>
  </div>
  <div class="sub mono" id="clock">-</div>
</header>
<main>
  <div class="section-title">
    <h2>Queue Summary</h2>
    <span class="sub" id="summaryMeta">schedule</span>
  </div>
  <div class="grid">
    <div class="panel"><div class="metric-label">Episodes</div><div class="metric-value" id="episodes">-</div><div class="metric-small" id="episodeCounts">-</div></div>
    <div class="panel"><div class="metric-label">Progress</div><div class="metric-value" id="progress">-</div><div class="bar" style="margin-top:8px"><div id="progressBar"></div></div></div>
    <div class="panel"><div class="metric-label">ETA</div><div class="metric-value" id="eta">-</div><div class="metric-small" id="etaDetail">-</div></div>
    <div class="panel"><div class="metric-label">Throughput</div><div class="metric-value" id="throughput">-</div><div class="metric-small">episodes/hour</div></div>
    <div class="panel"><div class="metric-label">Workers</div><div class="metric-value" id="workers">-</div><div class="metric-small" id="workerDetail">-</div></div>
  </div>

  <div class="section-title">
    <h2>Workers</h2>
    <span class="sub">capture PC work stealing status</span>
  </div>
  <table>
    <thead><tr><th>Worker</th><th>Status</th><th>Object</th><th>Episode</th><th>Task</th><th>Last Result</th><th>Updated</th></tr></thead>
    <tbody id="workerRows"><tr><td colspan="7" class="muted">loading</td></tr></tbody>
  </table>

  <div class="section-title">
    <h2>Episode Queue</h2>
    <span class="sub" id="queueMeta">tasks</span>
  </div>
  <table>
    <thead><tr><th>Status</th><th>Object</th><th>Episode</th><th>Stage</th><th>Frame Progress</th><th>Worker</th><th>Overlay Playback</th><th>Runtime</th><th>Output</th></tr></thead>
    <tbody id="taskRows"><tr><td colspan="9" class="muted">loading</td></tr></tbody>
  </table>

  <div class="section-title">
    <h2>Scheduler Events</h2>
    <span class="sub">events.jsonl tail</span>
  </div>
  <div class="events mono" id="events"></div>
</main>
<div class="viewer-backdrop" id="overlayViewer" aria-hidden="true">
  <div class="viewer" role="dialog" aria-modal="true" aria-labelledby="overlayViewerTitle">
    <div class="viewer-head">
      <div>
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
const esc = s => safe(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statusClass = s => safe(s).replace(/[^a-zA-Z0-9_]/g, '_');
const fmtTime = ts => ts ? new Date(Number(ts) * 1000).toLocaleTimeString() : '-';
const fmtDuration = seconds => {
  const n = Number(seconds);
  if (!Number.isFinite(n)) return '-';
  const s = Math.max(0, Math.round(n));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(r).padStart(2, '0')}s`;
  return `${r}s`;
};

function overlayCell(task) {
  const files = (task.overlay_files || []).filter(Boolean);
  if (!files.length) return '<span class="muted">-</span>';
  const links = files.slice(0, 4).map(path => {
    const name = path.split('/').pop() || 'overlay.mp4';
    const serial = name.replace(/^overlay_/, '').replace(/\.mp4$/i, '');
    return `<button type="button" class="overlay-button mono" data-overlay-path="${esc(path)}" data-overlay-label="${esc(serial)}" title="${esc(path)}">${esc(serial)}</button>`;
  }).join('');
  const more = files.length > 4 ? `<span class="muted">+${files.length - 4}</span>` : '';
  return `<div class="overlay-links">${links}${more}</div>`;
}

function frameProgress(task) {
  const done = Number(task.frame_done || 0);
  const total = Number(task.frame_total || 0);
  if (!total) return '-';
  const pct = Math.max(0, Math.min(100, 100 * done / total));
  return `<span class="bar slim"><div style="width:${pct}%"></div></span>${done}/${total}`;
}

function render(data) {
  const manifest = data.manifest || {};
  const counts = data.counts || {};
  const tasks = (data.tasks || []).slice().sort((a, b) => {
    const order = {running: 0, pending: 1, failed: 2, done: 3, skipped_done: 4, dry_run_done: 5};
    return (order[a.status] ?? 9) - (order[b.status] ?? 9) || safe(a.obj).localeCompare(safe(b.obj)) || safe(a.episode).localeCompare(safe(b.episode));
  });
  const workers = data.workers || [];
  const done = Number(data.done_like || 0);
  const total = Number(data.n_tasks || tasks.length || 0);
  const pct = total ? Math.max(0, Math.min(100, 100 * done / total)) : 0;
  const running = counts.running || 0;
  const pending = counts.pending || 0;
  const failed = counts.failed || 0;
  $('clock').textContent = new Date().toLocaleTimeString();
  $('subtitle').textContent = data.schedule_dir || '-';
  $('summaryMeta').textContent = `${manifest.hand || '-'} · ${manifest.stages || 'both'}`;
  $('episodes').textContent = `${done}/${total}`;
  $('episodeCounts').textContent = `running=${running} pending=${pending} failed=${failed}`;
  $('progress').textContent = pct.toFixed(1) + '%';
  $('progressBar').style.width = pct + '%';
  $('eta').textContent = data.eta_sec === null || data.eta_sec === undefined ? '-' : fmtDuration(data.eta_sec);
  $('etaDetail').textContent = `elapsed ${fmtDuration(data.elapsed_sec)} · remaining ${data.remaining ?? '-'}`;
  $('throughput').textContent = Number(data.throughput_eps_per_hour || 0).toFixed(2);
  $('workers').textContent = String(workers.length);
  $('workerDetail').textContent = workers.map(w => `${w.worker_id}:${w.status}`).join('  ') || '-';

  $('workerRows').innerHTML = workers.length ? workers.map(w => {
    const status = w.status || '-';
    return `<tr>
      <td class="mono">${esc(w.worker_id)}</td>
      <td><span class="status ${statusClass(status)}"><span class="dot"></span>${esc(status)}</span></td>
      <td>${esc(w.obj)}</td>
      <td class="mono">${esc(w.episode)}</td>
      <td class="mono">${esc(w.task_id)}</td>
      <td class="mono">${esc(w.last_status)}</td>
      <td class="mono">${esc(fmtTime(w.updated_at))}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="7" class="muted">no workers yet</td></tr>';

  $('queueMeta').textContent = `${tasks.length} shown`;
  $('taskRows').innerHTML = tasks.length ? tasks.slice(0, 100).map(t => {
    const status = t.status || '-';
    const output = t.overlay_output_dir || (t.episode_dir ? t.episode_dir + '/object_tracking/gotrack_output' : '-');
    return `<tr>
      <td><span class="status ${statusClass(status)}"><span class="dot"></span>${esc(status)}</span></td>
      <td>${esc(t.obj)}</td>
      <td class="mono">${esc(t.episode)}</td>
      <td class="mono">${esc(t.phase)}</td>
      <td class="mono">${frameProgress(t)}</td>
      <td class="mono">${esc(t.worker_id)}</td>
      <td>${overlayCell(t)}</td>
      <td class="mono">${esc(t.runtime_sec ? fmtDuration(t.runtime_sec) : '-')}</td>
      <td class="mono">${esc(output)}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="9" class="muted">no tasks</td></tr>';

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

function openOverlayVideo(path, label) {
  $('overlayViewerTitle').textContent = label ? 'Overlay ' + label : 'Overlay Playback';
  $('overlayViewerSub').textContent = path || '-';
  $('overlayMeta').textContent = path || '-';
  const video = $('overlayVideo');
  video.pause();
  video.src = '/overlay?path=' + encodeURIComponent(path);
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
    openOverlayVideo(btn.dataset.overlayPath, btn.dataset.overlayLabel);
    return;
  }
  if (target === $('overlayViewer') || target === $('overlayClose')) closeOverlayVideo();
});

document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape' && $('overlayViewer').classList.contains('open')) closeOverlayVideo();
});

tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


def make_handler(schedule_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                body = json.dumps(collect(schedule_dir), sort_keys=True, default=str).encode("utf-8")
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
            video = allowed_overlay_paths(schedule_dir).get(str(requested))
            if video is None:
                self.send_error(403, "overlay file is not in this schedule")
                return
            if not video.is_file():
                self.send_error(404, "overlay video not found")
                return
            size = video.stat().st_size
            start, end, status_code = 0, size - 1, 200
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    unit, spec = range_header.split("=", 1)
                    if unit.strip().lower() != "bytes":
                        raise ValueError("unsupported unit")
                    spec = spec.split(",", 1)[0].strip()
                    if spec.startswith("-"):
                        start = max(0, size - int(spec[1:]))
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
            self.send_response(status_code)
            self.send_header("Content-Type", mimetypes.guess_type(video.name)[0] or "video/mp4")
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
    p.add_argument("--schedule-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--open-browser", action="store_true")
    args = p.parse_args()

    schedule_dir = Path(args.schedule_dir).expanduser()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(schedule_dir))
    url = f"http://{args.host}:{args.port}/"
    print(url, flush=True)
    print(f"reading {schedule_dir}", flush=True)
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
