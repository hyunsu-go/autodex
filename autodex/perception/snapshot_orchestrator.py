#!/usr/bin/env python3
"""Robot-PC orchestrator for snapshot_daemon.

Sends "snap" to all capture PC snapshot_daemons (port 6894), subscribes to
their JPEG PUB stream (port 5009), collects one frame per camera, optionally
decodes to BGR ndarrays and/or writes raw JPGs to a local directory.

Lightweight cousin of InitOrchestrator — no inference, no models, no IoU/sil.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)


def _to_home_relative(p) -> str:
    p = str(p)
    home = str(Path.home())
    if p.startswith(home + "/"):
        return "~/" + p[len(home) + 1:]
    return p


def _parse_multipart(parts: List[bytes]):
    if len(parts) < 2 or parts[0] != b"data":
        return None, []
    try:
        meta = json.loads(parts[1].decode("utf-8"))
    except Exception:
        return None, []
    return meta, list(parts[2:])


class _SnapBuffer:
    def __init__(self):
        self._d: Dict[int, Dict[str, bytes]] = defaultdict(dict)
        self._lock = threading.Lock()

    def put(self, req_id: int, serial: str, blob: bytes):
        with self._lock:
            self._d[req_id][serial] = blob

    def get(self, req_id: int) -> Dict[str, bytes]:
        with self._lock:
            return dict(self._d.get(req_id, {}))

    def drop(self, req_id: int):
        with self._lock:
            self._d.pop(req_id, None)


class _SubThread(threading.Thread):
    def __init__(self, capture_ips: List[str], port: int,
                 buffer: _SnapBuffer):
        super().__init__(daemon=True, name="snapshot_sub")
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        for ip in capture_ips:
            self.sock.connect(f"tcp://{ip}:{port}")
        self.buffer = buffer
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                if self.sock.poll(timeout=100):
                    parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                    meta, blobs = _parse_multipart(parts)
                    if meta is None:
                        continue
                    for item, blob in zip(meta.get("items", []), blobs):
                        self.buffer.put(int(item["req_id"]),
                                        str(item["serial"]), blob)
            except zmq.Again:
                pass
            except Exception as exc:
                logger.warning(f"[snapshot_sub] {exc}")


class SnapshotOrchestrator:
    """Coordinate one-shot JPEG snapshots across capture PCs.

    Parameters
    ----------
    pc_list : capture PC names (must match daemons launched via
              scripts/snapshot_daemons.sh).
    capture_ips : IPs aligned with pc_list.
    port_snap : daemon PUB port (5009 default).
    port_cmd  : daemon REQ/REP port (6894 default).
    """

    def __init__(
        self,
        pc_list: List[str],
        capture_ips: List[str],
        port_snap: int = 5009,
        port_cmd: int = 6894,
    ):
        from paradex.io.capture_pc.command_sender import CommandSender

        assert len(pc_list) == len(capture_ips)
        self.pc_list = pc_list
        self.capture_ips = capture_ips
        self.cmd = CommandSender(pc_list=pc_list, port=port_cmd)
        self.buf = _SnapBuffer()
        self._sub_thread = _SubThread(capture_ips, port_snap, self.buf)
        self._sub_thread.start()
        time.sleep(0.3)  # SUB connect handshake

    def snap(
        self,
        n_expected: int,
        timeout_s: float = 3.0,
        save_dir_local: Optional[str] = None,
        save_dir_remote: Optional[str] = None,
        request_id: Optional[int] = None,
        decode: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Trigger one snapshot pass and collect ``n_expected`` JPEGs.

        save_dir_local : if set, write each camera's JPEG to
            ``<save_dir_local>/<serial>.jpg`` on the robot PC (the path the
            auto_label_charuco caller will then read from).
        save_dir_remote : if set, also instruct each capture PC daemon to
            dump a copy locally (rarely needed; mainly useful for debugging).
        decode : if True, also return per-cam BGR ndarrays.

        Returns
        -------
        result : {serial: {"jpeg": bytes, "image": np.ndarray or None}}
        timing : {dispatch_to_collected_s, n_received, n_expected, ...}
        """
        if request_id is None:
            request_id = int(time.time() * 1000) & 0x7fffffff
        # Drop any stale buffers from older trials.
        with self.buf._lock:
            self.buf._d.clear()

        cmd_info = {"request_id": int(request_id)}
        if save_dir_remote is not None:
            cmd_info["save_dir"] = _to_home_relative(save_dir_remote)

        t_dispatch = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            self.cmd.send_command("snap", wait=False, cmd_info=cmd_info)

        deadline = time.perf_counter() + timeout_s
        last_n = -1
        while time.perf_counter() < deadline:
            got = self.buf.get(request_id)
            if len(got) != last_n:
                last_n = len(got)
                logger.info(f"[snap] req={request_id} {len(got)}/{n_expected} "
                            f"({time.perf_counter()-t_dispatch:.2f}s)")
            if len(got) >= n_expected:
                break
            time.sleep(0.01)
        blobs = self.buf.get(request_id)
        t_collected = time.perf_counter()

        result: Dict[str, Dict[str, Any]] = {}
        if save_dir_local:
            Path(save_dir_local).mkdir(parents=True, exist_ok=True)
        for s, blob in blobs.items():
            entry: Dict[str, Any] = {"jpeg": blob, "image": None}
            if decode:
                arr = np.frombuffer(blob, dtype=np.uint8)
                entry["image"] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if save_dir_local:
                (Path(save_dir_local) / f"{s}.jpg").write_bytes(blob)
            result[s] = entry
        self.buf.drop(request_id)

        timing = {
            "request_id": int(request_id),
            "dispatch_to_collected_s": t_collected - t_dispatch,
            "n_received": len(blobs),
            "n_expected": int(n_expected),
            "save_dir_local": save_dir_local,
            "save_dir_remote": save_dir_remote,
        }
        return result, timing

    def close(self):
        self._sub_thread.stop()
        try:
            for s in self.cmd.sockets.values():
                try:
                    s.setsockopt(zmq.LINGER, 0)
                except Exception:
                    pass
                s.close()
            self.cmd.context.term()
        except Exception:
            pass
