from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from ...common.schemas import TrainingSpec


class TrainingJobManager:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.jobs: dict[str, dict[str, Any]] = {}

    def handle_train(self, request: dict[str, Any], dry_run: bool) -> dict:
        if dry_run:
            return {
                **public_request(request),
                "status": "dry_run",
                "message": "Training command prepared but not launched. Set dry_run=false to start it.",
            }

        process_env = os.environ.copy()
        process_env.update(request.get("env", {}))
        log_handle = None
        log_path = request.get("log_path")
        if log_path:
            log_file = Path(log_path).expanduser()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_file.open("ab")
            request["log_path"] = str(log_file)

        try:
            process = subprocess.Popen(
                request["command"],
                cwd=request["cwd"],
                env=process_env,
                stdout=log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            if log_handle is not None:
                log_handle.close()

        job_id = str(request["job_id"])
        self.jobs[job_id] = {
            "process": process,
            "request": public_request(request),
            "started_at": time.time(),
        }
        return {
            **self.status(job_id),
            "message": f"{self.model_name} training launched.",
        }

    def status(self, job_id: str | None = None) -> dict:
        if job_id is None:
            jobs = [self._job_status(job_id, job) for job_id, job in self.jobs.items()]
            return {"status": "ok", "jobs": jobs}
        job = self.jobs.get(job_id)
        if job is None:
            return {"status": "not_found", "job_id": job_id}
        return self._job_status(job_id, job)

    def _job_status(self, job_id: str, job: dict[str, Any]) -> dict:
        process = job["process"]
        returncode = process.poll()
        if returncode is None:
            status = "running"
        elif returncode == 0:
            status = "succeeded"
        else:
            status = "failed"
        return {
            **job["request"],
            "status": status,
            "job_id": job_id,
            "pid": process.pid,
            "returncode": returncode,
            "started_at": job["started_at"],
        }


def external_training_request(
    model_name: str,
    spec: TrainingSpec,
    *,
    default_cwd: str | Path,
    default_log_name: str,
) -> dict[str, Any]:
    metadata = spec.metadata
    command_value = (
        metadata.get("training_command")
        or metadata.get("train_command")
        or metadata.get("command")
    )
    if not command_value:
        raise NotImplementedError(
            f"{model_name} has no native training launcher in this checkout; "
            "provide metadata.training_command to use the generic launcher"
        )

    command = command_from_value(command_value)
    command.extend(string_list(metadata.get("extra_args")))
    cwd = Path(metadata.get("cwd") or metadata.get("repo_path") or default_cwd).expanduser().resolve()
    log_path = metadata.get("log_path")
    if log_path is None and spec.output_dir:
        log_path = str(Path(spec.output_dir).expanduser() / default_log_name)

    return {
        "model_name": model_name,
        "mode": "external-training-launcher",
        "job_id": str(metadata.get("job_id") or f"{model_name}-{time.time_ns()}"),
        "command": command,
        "command_text": shlex.join(command),
        "cwd": str(cwd),
        "env": string_dict(metadata.get("env", {})),
        "log_path": str(log_path) if log_path else None,
        "notes": [
            "This adapter uses metadata.training_command because no native training entrypoint is registered for this model."
        ],
    }


def public_request(request: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in request.items() if key not in {"env", "env_preview"}}
    public["env"] = request.get("env_preview") or mask_env(request.get("env", {}))
    return public


def command_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError("training command must be a string or list of strings")


def bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def mask_env(env: dict[str, str]) -> dict[str, str]:
    masked = {}
    for key, value in env.items():
        if "KEY" in key or "TOKEN" in key or "SECRET" in key or "PASSWORD" in key:
            masked[key] = "***"
        else:
            masked[key] = value
    return masked
