from __future__ import annotations

import argparse
from http import HTTPStatus

from ..common.http import JsonHandler, serve
from ..common.registry import SIM_ADAPTERS
from ..common.schemas import Action, TaskSpec
from . import adapters  # noqa: F401


def build_handler(adapter_name: str) -> type[JsonHandler]:
    adapter_cls = SIM_ADAPTERS[adapter_name]
    adapter = adapter_cls()

    class SimHandler(JsonHandler):
        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/reset":
                    response = adapter.reset(TaskSpec.from_dict(payload))
                    self._write_json(response.to_dict())
                    return
                if self.path == "/step":
                    response = adapter.step(Action.from_dict(payload))
                    self._write_json(response.to_dict())
                    return
                if self.path == "/get_state":
                    self._write_json(adapter.get_state())
                    return
                if self.path == "/close":
                    self._write_json(adapter.close())
                    return
                if self.path == "/health":
                    self._write_json({"status": "ok", "adapter": adapter_name})
                    return
                self._write_json({"error": f"unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._write_json(
                    {"error": f"{type(exc).__name__}: {exc}", "adapter": adapter_name, "path": self.path},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return SimHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulator HTTP service")
    parser.add_argument("--adapter", choices=sorted(SIM_ADAPTERS.keys()), default="dummy-simulator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    serve(build_handler(args.adapter), args.host, args.port)


if __name__ == "__main__":
    main()
