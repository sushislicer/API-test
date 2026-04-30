from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .serialization import dumps_json, loads_json, normalize_endpoint


class JsonHandler(BaseHTTPRequestHandler):
    server_version = "EvalSystem/0.1"

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return loads_json(self.rfile.read(length))

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        raw = dumps_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args) -> None:
        return


def serve(handler_cls: type[BaseHTTPRequestHandler], host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), handler_cls)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def post_json(endpoint: str, path: str, payload: dict | None = None) -> dict:
    base = normalize_endpoint(endpoint).rstrip("/")
    request = Request(
        f"{base}{path}",
        data=dumps_json(payload or {}),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(os.environ.get("EVAL_SYSTEM_HTTP_TIMEOUT_S", "120"))
    try:
        with urlopen(request, timeout=timeout) as response:
            response_payload = loads_json(response.read())
    except HTTPError as exc:
        raw = exc.read()
        detail = loads_json(raw).get("error", raw.decode("utf-8", errors="replace")) if raw else str(exc)
        raise RuntimeError(f"{endpoint}{path} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {endpoint}{path}: {exc.reason}") from exc

    if isinstance(response_payload, dict) and response_payload.get("error"):
        raise RuntimeError(f"{endpoint}{path} failed: {response_payload['error']}")
    return response_payload
