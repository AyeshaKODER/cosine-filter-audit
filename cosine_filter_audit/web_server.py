"""Loopback-only local UI and guarded Flower run launcher."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
RUNS_ROOT = PROJECT_ROOT / "runs"
RESULTS_ROOT = PROJECT_ROOT / "results"
HOST = "127.0.0.1"
PORT = 8765
MAX_ROUNDS = 10
SELECTION_MODES = {
    "none",
    "detector",
    "oracle",
    "random_same_size",
    "benign_only",
}
_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z_[a-f0-9]{12}$")
_JOB_LOCK = threading.Lock()
_ACTIVE_JOB: dict[str, Any] | None = None


def validate_run_config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    expected = {
        "representation_mode",
        "reference_mode",
        "selection_mode",
        "threshold",
        "attack_enabled",
        "attack_scale",
        "num_rounds",
        "seed",
    }
    if set(payload) != expected:
        raise ValueError("Request must contain exactly the supported run settings")

    representation = payload["representation_mode"]
    if not isinstance(representation, str) or representation not in {
        "delta",
        "legacy_full_model",
    }:
        raise ValueError("Unsupported representation mode")
    reference = payload["reference_mode"]
    references = (
        {"current_mean", "previous_accepted_delta"}
        if representation == "delta"
        else {"current_mean", "previous_global_model"}
    )
    if not isinstance(reference, str) or reference not in references:
        raise ValueError("Reference mode is incompatible with the representation")
    selection = payload["selection_mode"]
    if not isinstance(selection, str) or selection not in SELECTION_MODES:
        raise ValueError("Unsupported selection mode")
    if selection in {"oracle", "benign_only"} and representation != "delta":
        raise ValueError("Oracle selection requires delta representation")

    threshold = _finite_number(payload["threshold"], "threshold")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between -1 and 1")
    attack_scale = _finite_number(payload["attack_scale"], "attack_scale")
    rounds = _integer(payload["num_rounds"], "num_rounds")
    if not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"Number of rounds must be between 1 and {MAX_ROUNDS}")
    seed = _integer(payload["seed"], "seed")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be between 0 and 2^32-1")
    if not isinstance(payload["attack_enabled"], bool):
        raise ValueError("attack_enabled must be a boolean")

    return {
        "representation_mode": representation,
        "reference_mode": reference,
        "selection_mode": selection,
        "threshold": threshold,
        "attack_enabled": payload["attack_enabled"],
        "attack_scale": attack_scale,
        "num_rounds": rounds,
        "seed": seed,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def flower_run_config(config: dict[str, Any]) -> str:
    attack_start = 1 if config["attack_enabled"] else 999
    return " ".join(
        (
            f"num-server-rounds={config['num_rounds']}",
            "fraction-evaluate=0.5",
            "learning-rate=0.01",
            "batch-size=32",
            "local-epochs=1",
            "save-model=false",
            f"representation-mode='{config['representation_mode']}'",
            f"reference-mode='{config['reference_mode']}'",
            f"selection-mode='{config['selection_mode']}'",
            f"cosine-threshold={config['threshold']!r}",
            f"attack-start-round={attack_start}",
            f"attack-scale={config['attack_scale']!r}",
            "attacker-node-id=-1",
            f"seed={config['seed']}",
            "deterministic=true",
        )
    )


def _run_summaries() -> tuple[list[dict[str, Any]], list[str]]:
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    artifacts = (
        list(RUNS_ROOT.rglob("run.json")) if RUNS_ROOT.exists() else []
    ) + (list(RESULTS_ROOT.rglob("run.json")) if RESULTS_ROOT.exists() else [])
    for artifact in artifacts:
        if not artifact.is_file():
            continue
        try:
            record = json.loads(artifact.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "run_id": record["run_id"],
                    "created_at": record["created_at"],
                    "config": record["config"],
                    "provenance": record["provenance"],
                    "server_evaluation_metrics": record[
                        "server_evaluation_metrics"
                    ],
                    "rounds": {
                        number: {
                            "round_status": item.get("round_status"),
                            "selected_count": len(
                                item.get("selected_node_ids", [])
                            ),
                            "flagged_count": sum(
                                client.get("flagged") is True
                                for client in item.get("clients", {}).values()
                            ),
                        }
                        for number, item in record["rounds"].items()
                    },
                }
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"{artifact.relative_to(PROJECT_ROOT)}: {exc}")
    summaries.sort(key=lambda item: item["created_at"], reverse=True)
    return summaries, errors


def _tail_log(path: Path, limit: int = 12000) -> str:
    try:
        with path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - limit), os.SEEK_SET)
            return log.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _start_run(config: dict[str, Any]) -> dict[str, Any]:
    global _ACTIVE_JOB
    executable = shutil.which("flwr")
    if executable is None:
        raise RuntimeError(
            "Flower CLI was not found. Activate the environment installed from "
            "this project's pyproject.toml before starting the UI."
        )

    with _JOB_LOCK:
        if _ACTIVE_JOB is not None and _ACTIVE_JOB["process"].poll() is None:
            raise RuntimeError("A Flower run is already active")
        job_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:12]
        )
        log_dir = RUNS_ROOT / "ui_jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"
        command = [
            executable,
            "run",
            ".",
            "--stream",
            "--run-config",
            flower_run_config(config),
        ]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        with log_path.open("x", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                shell=False,
            )
        _ACTIVE_JOB = {
            "job_id": job_id,
            "process": process,
            "config": config,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "log_path": log_path,
        }
        return {
            "job_id": job_id,
            "started_at": _ACTIVE_JOB["started_at"],
            "config": config,
        }


def _job_status() -> dict[str, Any]:
    with _JOB_LOCK:
        if _ACTIVE_JOB is None:
            return {"running": False, "job": None}
        process = _ACTIVE_JOB["process"]
        return_code = process.poll()
        return {
            "running": return_code is None,
            "job": {
                "job_id": _ACTIVE_JOB["job_id"],
                "started_at": _ACTIVE_JOB["started_at"],
                "return_code": return_code,
                "config": _ACTIVE_JOB["config"],
                "log_tail": _tail_log(_ACTIVE_JOB["log_path"]),
            },
        }


class LocalUIHandler(BaseHTTPRequestHandler):
    server_version = "CosineFilterAuditUI/1.0"

    def do_GET(self) -> None:
        if not self._valid_host():
            self._send_json(403, {"error": "Only loopback hostnames are accepted"})
            return
        path = unquote(self.path.split("?", 1)[0])
        if path in {"/", "/index.html"}:
            self._send_file("index.html", "text/html; charset=utf-8")
        elif path == "/styles.css":
            self._send_file("styles.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._send_file("app.js", "text/javascript; charset=utf-8")
        elif path == "/api/health":
            self._send_json(200, {"ok": True, "host": HOST})
        elif path == "/api/runs":
            runs, errors = _run_summaries()
            self._send_json(200, {"runs": runs, "artifact_errors": errors})
        elif path == "/api/status":
            self._send_json(200, _job_status())
        elif path.startswith("/api/runs/"):
            run_id = path.removeprefix("/api/runs/")
            if not _RUN_ID_PATTERN.fullmatch(run_id):
                self._send_json(400, {"error": "Invalid run ID"})
                return
            artifact_candidates = (
                RUNS_ROOT / run_id / "run.json",
                RESULTS_ROOT / "experiment_artifacts" / "smoke" / run_id / "run.json",
                RESULTS_ROOT
                / "experiment_artifacts"
                / "exploratory_delta_pilot"
                / run_id
                / "run.json",
            )
            artifact = next(
                (candidate for candidate in artifact_candidates if candidate.is_file()),
                None,
            )
            if artifact is None:
                self._send_json(404, {"error": "Run artifact not found"})
                return
            try:
                record = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._send_json(500, {"error": f"Could not read run artifact: {exc}"})
                return
            self._send_json(200, record)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._valid_host() or not self._valid_origin():
            self._send_json(403, {"error": "Request must originate from this local UI"})
            return
        if self.path != "/api/start":
            self._send_json(404, {"error": "Not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Expected application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if not 0 < length <= 16_384:
            self._send_json(413, {"error": "Request body size is invalid"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            config = validate_run_config(payload)
            job = _start_run(config)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except (OSError, RuntimeError) as exc:
            self._send_json(503, {"error": str(exc)})
            return
        self._send_json(202, {"job": job})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[Local UI] {self.address_string()} {format % args}")

    def _valid_host(self) -> bool:
        host = self.headers.get("Host", "").lower()
        return host in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def _valid_origin(self) -> bool:
        origin = self.headers.get("Origin", "").lower()
        return origin in {
            f"http://127.0.0.1:{self.server.server_port}",
            f"http://localhost:{self.server.server_port}",
        }

    def _send_file(self, name: str, content_type: str) -> None:
        path = FRONTEND_ROOT / name
        try:
            content = path.read_bytes()
        except OSError as exc:
            self._send_json(500, {"error": f"Could not read UI asset: {exc}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), LocalUIHandler)
    print(f"Cosine Filter Audit UI listening at http://{HOST}:{PORT}")
    print("Loopback-only prototype; do not expose it to a network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
