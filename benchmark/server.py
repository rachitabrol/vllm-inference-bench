"""Start/stop `vllm serve` as a subprocess, wait for /health, capture logs. Stdlib only."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Self


class ServerError(RuntimeError):
    pass


class VllmServer:
    def __init__(self, cmd: list[str], log_path: str | Path, port: int = 8000):
        self.cmd = cmd
        self.log_path = Path(log_path)
        self.port = port
        self.proc: subprocess.Popen | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")  # noqa: SIM115 - closed in stop()
        # new session so we can kill the whole process group (vLLM spawns workers)
        self.proc = subprocess.Popen(
            self.cmd, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True
        )

    def wait_healthy(self, timeout_s: float = 900.0, poll_s: float = 2.0) -> float:
        """Block until GET /health returns 200. Returns seconds waited. Raises with a log tail on failure."""
        assert self.proc is not None, "start() first"
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rc = self.proc.poll()
            if rc is not None:
                raise ServerError(f"vllm serve exited early (code {rc}). Log tail:\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=3) as r:
                    if r.status == 200:
                        return time.time() - t0
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(poll_s)
        raise ServerError(f"vllm serve not healthy after {timeout_s:.0f}s. Log tail:\n{self.log_tail()}")

    def log_tail(self, n_lines: int = 60) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return "<no log>"
        return "\n".join(lines[-n_lines:])

    def stop(self, grace_s: float = 30.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=grace_s)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
