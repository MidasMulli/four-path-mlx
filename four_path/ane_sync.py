"""
Synchronized ANE Draft Source
==============================

Per-round ANE drafting with warm KV cache. Instead of pre-generating a
lookahead buffer (which diverges), we maintain the ANE's KV state in sync
with the GPU generate loop and produce one draft token per round.

Prefill: ~330ms (one-time per conversation)
Decode:  ~17ms per token (56 tok/s on ANE)

The ANE draft fits inside the GPU's verification window (~50ms on 9B).
"""

import json
import socket
import struct
import threading
import time
from typing import Optional


ANE_SOCKET = "/tmp/orion-ane-server.sock"


def _ane_rpc(cmd_dict: dict, timeout: float = 10.0) -> dict:
    """Send a command to the ANE server via Unix socket."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(ANE_SOCKET)
        req = json.dumps(cmd_dict).encode()
        sock.sendall(struct.pack("!I", len(req)) + req)
        raw_len = sock.recv(4)
        if len(raw_len) < 4:
            return {"status": "error", "error": "connection closed"}
        resp_len = struct.unpack("!I", raw_len)[0]
        raw = b""
        while len(raw) < resp_len:
            chunk = sock.recv(resp_len - len(raw))
            if not chunk:
                break
            raw += chunk
        sock.close()
        return json.loads(raw)
    except Exception as e:
        return {"status": "error", "error": str(e)}


class ANESyncDrafter:
    """Synchronized ANE draft source with warm KV cache.

    Usage:
        drafter = ANESyncDrafter()
        drafter.prefill("user's message text")  # One-time, ~330ms

        # Per token round:
        draft = drafter.draft_one()  # ~17ms
        if draft is not None:
            # Verify draft against GPU
            ...
    """

    def __init__(self):
        self.active = False
        self.prefill_ms = 0
        self.tokens_drafted = 0
        self.decode_times = []

    def prefill(self, prompt: str) -> bool:
        """Prefill the ANE KV cache. Call once per conversation/prompt."""
        r = _ane_rpc({"cmd": "prefill", "prompt": prompt}, timeout=30.0)
        if r.get("status") == "ok":
            self.active = True
            self.prefill_ms = r.get("prefill_ms", 0)
            self.tokens_drafted = 0
            self.decode_times = []
            return True
        self.active = False
        return False

    def draft_one(self) -> Optional[int]:
        """Generate one draft token from the ANE with warm KV cache.
        Returns token_id or None if unavailable."""
        if not self.active:
            return None

        t0 = time.perf_counter()
        r = _ane_rpc({"cmd": "decode_one"}, timeout=5.0)
        elapsed = (time.perf_counter() - t0) * 1000
        self.decode_times.append(elapsed)

        if r.get("status") != "ok" or r.get("is_stop"):
            self.active = False
            return None

        self.tokens_drafted += 1
        return r.get("token_id")

    def draft_one_async(self) -> "ANEDraftFuture":
        """Launch ANE decode in a background thread. Returns a future."""
        future = ANEDraftFuture()
        if not self.active:
            future._set(None, 0)
            return future

        def _worker():
            t0 = time.perf_counter()
            r = _ane_rpc({"cmd": "decode_one"}, timeout=5.0)
            elapsed = (time.perf_counter() - t0) * 1000
            self.decode_times.append(elapsed)
            if r.get("status") != "ok" or r.get("is_stop"):
                self.active = False
                future._set(None, elapsed)
            else:
                self.tokens_drafted += 1
                future._set(r.get("token_id"), elapsed)

        threading.Thread(target=_worker, daemon=True).start()
        return future

    def stats(self) -> dict:
        times = self.decode_times
        return {
            "prefill_ms": self.prefill_ms,
            "tokens_drafted": self.tokens_drafted,
            "active": self.active,
            "decode_median_ms": sorted(times)[len(times) // 2] if times else 0,
            "decode_mean_ms": sum(times) / len(times) if times else 0,
        }


class ANEDraftFuture:
    """Future for async ANE draft token."""

    def __init__(self):
        self._event = threading.Event()
        self._token_id = None
        self._elapsed_ms = 0

    def _set(self, token_id, elapsed_ms):
        self._token_id = token_id
        self._elapsed_ms = elapsed_ms
        self._event.set()

    def wait(self, timeout: float = 1.0) -> Optional[int]:
        """Wait for the draft token. Returns token_id or None."""
        self._event.wait(timeout)
        return self._token_id

    def is_ready(self) -> bool:
        return self._event.is_set()

    @property
    def token_id(self):
        return self._token_id

    @property
    def elapsed_ms(self):
        return self._elapsed_ms
