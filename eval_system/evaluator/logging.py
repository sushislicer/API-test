from __future__ import annotations

import csv
import json
from pathlib import Path

from ..common.serialization import ensure_dir
from ..common.schemas import EpisodeLog, EvalReport


class StructuredLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = ensure_dir(output_dir)

    def write_episode(self, episode: EpisodeLog) -> Path:
        path = self.output_dir / f"episode_{episode.episode_index:04d}.json"
        path.write_text(json.dumps(episode.to_dict(), indent=2), encoding="utf-8")
        return path

    def write_report(self, report: EvalReport) -> Path:
        path = self.output_dir / "report.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return path

    def write_summary_csv(self, report: EvalReport) -> Path:
        path = self.output_dir / "summary.csv"
        fieldnames = ["episode_index", "task", "steps", "episode_return", "success", "avg_prediction_error"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in report.episode_summaries:
                writer.writerow({name: row.get(name) for name in fieldnames})
        return path
