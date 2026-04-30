from __future__ import annotations

import argparse
from http import HTTPStatus

from ..common.http import JsonHandler, serve
from ..common.registry import MODEL_ADAPTERS
from ..common.schemas import Action, Observation, TaskSpec, TrainingSpec, TransitionQuery
from . import adapters  # noqa: F401


def build_handler(adapter_name: str) -> type[JsonHandler]:
    adapter_cls = MODEL_ADAPTERS[adapter_name]
    adapter = adapter_cls()

    class ModelHandler(JsonHandler):
        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/reset":
                    response = adapter.reset(TaskSpec.from_dict(payload))
                    self._write_json(response)
                    return
                if self.path == "/act":
                    response = adapter.act(Observation.from_dict(payload))
                    self._write_json(response.to_dict())
                    return
                if self.path == "/predict_next":
                    query = TransitionQuery.from_dict(payload)
                    response = adapter.predict_next(query.observation, query.action)
                    self._write_json(response.to_dict())
                    return
                if self.path == "/train":
                    response = adapter.train(TrainingSpec.from_dict(payload))
                    self._write_json(response)
                    return
                if self.path == "/training_status":
                    response = adapter.training_status(payload.get("job_id"))
                    self._write_json(response)
                    return
                if self.path == "/health":
                    self._write_json({"status": "ok", "adapter": adapter_name})
                    return
                self._write_json({"error": f"unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)
            except NotImplementedError as exc:
                self._write_json({"error": str(exc), "adapter": adapter_name}, status=HTTPStatus.NOT_IMPLEMENTED)
            except Exception as exc:
                self._write_json(
                    {"error": f"{type(exc).__name__}: {exc}", "adapter": adapter_name, "path": self.path},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return ModelHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="World model HTTP service")
    parser.add_argument("--adapter", choices=sorted(MODEL_ADAPTERS.keys()), default="dummy-model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    serve(build_handler(args.adapter), args.host, args.port)


if __name__ == "__main__":
    main()
