from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import STATUS_DIR
from app.models import PipelineStage, PipelineStatus, StageInfo, StageStatus

logger = logging.getLogger(__name__)

DEFAULT_STAGES = [
    PipelineStage.UPLOADED,
    PipelineStage.PARSING,
    PipelineStage.INDEXING,
    PipelineStage.SUMMARIZING,
    PipelineStage.PLANNING,
    PipelineStage.AWAITING_APPROVAL,
    PipelineStage.SCRIPTING,
    PipelineStage.VOICE,
    PipelineStage.COMPLETED,
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StatusStore:
    def __init__(self, base_dir: Path = STATUS_DIR) -> None:
        self.base_dir = base_dir

    def _path(self, paper_id: str) -> Path:
        return self.base_dir / f"{paper_id}.json"

    def create(self, paper_id: str, filename: str) -> PipelineStatus:
        now = _now()
        stages = {
            stage.value: StageInfo(status=StageStatus.PENDING, message="Waiting")
            for stage in DEFAULT_STAGES
        }
        stages[PipelineStage.UPLOADED.value] = StageInfo(
            status=StageStatus.COMPLETED,
            message="PDF uploaded",
            updated_at=now,
        )
        status = PipelineStatus(
            paper_id=paper_id,
            filename=filename,
            current_stage=PipelineStage.UPLOADED,
            stages=stages,
            created_at=now,
            updated_at=now,
        )
        self.save(status)
        return status

    def load(self, paper_id: str) -> PipelineStatus | None:
        path = self._path(paper_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return PipelineStatus.model_validate(data)

    def save(self, status: PipelineStatus) -> None:
        status.updated_at = _now()
        self._path(status.paper_id).write_text(
            status.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def set_stage(
        self,
        paper_id: str,
        stage: PipelineStage,
        status: StageStatus,
        message: str = "",
    ) -> PipelineStatus:
        current = self.load(paper_id)
        if current is None:
            raise FileNotFoundError(f"No status for {paper_id}")

        now = _now()
        current.stages[stage.value] = StageInfo(
            status=status,
            message=message,
            updated_at=now,
        )
        current.current_stage = stage
        if status == StageStatus.FAILED:
            current.error = message
        self.save(current)
        return current

    def mark_failed(self, paper_id: str, message: str) -> PipelineStatus:
        current = self.load(paper_id)
        if current is None:
            raise FileNotFoundError(f"No status for {paper_id}")
        current.current_stage = PipelineStage.FAILED
        current.error = message
        current.stages[PipelineStage.FAILED.value] = StageInfo(
            status=StageStatus.FAILED,
            message=message,
            updated_at=_now(),
        )
        self.save(current)
        return current
