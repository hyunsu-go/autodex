"""Tiny status page for batch_object_overlay.py progress.

Run:
    python src/process/overlay_status_server.py            # http://localhost:8765
    python src/process/overlay_status_server.py --port 9000
"""
import argparse
import html
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

EXP_BASE = Path.home() / "shared_data/AutoDex/experiment/selected_100"
OUT_BASE = Path.home() / "shared_data/AutoDex/object_overlay_video"
GOTRACK_REL = "object_tracking/gotrack_output/world_pose_records.json"
HANDS = ["allegro", "inspire"]


def episode_status(ep_dir: Path, out_ep_dir: Path):
    serials = sorted(p.stem for p in (ep_dir / "videos").glob("*.avi")) if (ep_dir / "videos").is_dir() else []
    if not serials or not (ep_dir / "pose_world.npy").exists():
        return None

    gt_rec = ep_dir / GOTRACK_REL
    gt_ok = False
    if gt_rec.exists():
        try:
            recs = json.load(open(gt_rec))
            gt_ok = any(r.get("pose_world") is not None for r in recs)
        except Exception:
            gt_ok = False

    n_overlay = sum(1 for s in serials if (out_ep_dir / f"overlay_{s}.mp4").exists())
    return {
        "n_serials": len(serials),
        "gotrack_done": gt_ok,
        "overlay_done": n_overlay,
        "overlay_total": len(serials),
        "fully_done": gt_ok and n_overlay == len(serials),
    }


def collect(hand: str):
    hand_dir = EXP_BASE / hand
    if not hand_dir.is_dir():
        return []
    out = []
    for obj_dir in sorted(hand_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        eps = []
        for ep_dir in sorted(obj_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            st = episode_status(ep_dir, OUT_BASE / hand / obj_dir.name / ep_dir.name)
            if st is None:
                continue
            eps.append((ep_dir.name, st))
        if eps:
            out.append((obj_dir.name, eps))
    return out


def render_page() -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta http-equiv='refresh' content='30'>",
        "<title>overlay status</title>",
        "<style>",
        "body{font-family:ui-monospace,Menlo,monospace;background:#0e0e10;color:#ddd;margin:24px;}",
        "h1{margin:0 0 8px;font-size:18px;}h2{margin:24px 0 6px;font-size:14px;color:#aaa;}",
        "table{border-collapse:collapse;width:100%;font-size:12px;}",
        "th,td{padding:4px 10px;text-align:left;border-bottom:1px solid #222;}",
        "th{color:#888;font-weight:normal;}",
        ".bar{display:inline-block;height:8px;background:#222;width:120px;vertical-align:middle;border-radius:2px;overflow:hidden;}",
        ".fill{height:100%;background:#3fb950;}",
        ".done{color:#3fb950;}.partial{color:#d29922;}.todo{color:#666;}",
        "small{color:#666;}",
        "</style></head><body>",
        f"<h1>batch_object_overlay status</h1>",
        f"<small>refreshes every 30s · {time.strftime('%Y-%m-%d %H:%M:%S')}</small>",
    ]

    for hand in HANDS:
        data = collect(hand)
        if not data:
            continue
        total_eps = sum(len(eps) for _, eps in data)
        done_eps = sum(1 for _, eps in data for _, st in eps if st["fully_done"])
        gt_eps = sum(1 for _, eps in data for _, st in eps if st["gotrack_done"])
        parts.append(f"<h2>{hand} — {done_eps}/{total_eps} eps complete · gotrack {gt_eps}/{total_eps}</h2>")
        parts.append("<table><tr><th>object</th><th>eps</th><th>gotrack</th><th>overlay</th><th></th></tr>")
        for obj, eps in data:
            n = len(eps)
            n_gt = sum(1 for _, st in eps if st["gotrack_done"])
            n_ov = sum(1 for _, st in eps if st["fully_done"])
            n_partial = sum(1 for _, st in eps if 0 < st["overlay_done"] < st["overlay_total"])
            pct = int(100 * n_ov / n)
            cls = "done" if n_ov == n else ("partial" if n_ov > 0 or n_partial > 0 else "todo")
            parts.append(
                f"<tr><td>{html.escape(obj)}</td>"
                f"<td>{n}</td>"
                f"<td>{n_gt}/{n}</td>"
                f"<td class='{cls}'>{n_ov}/{n}{(' (+'+str(n_partial)+' partial)') if n_partial else ''}</td>"
                f"<td><span class='bar'><span class='fill' style='width:{pct}%'></span></span> {pct}%</td>"
                f"</tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    return "".join(parts)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = render_page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"http://{args.host}:{args.port}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
