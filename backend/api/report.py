"""Report generation endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.schemas import ReportRequest
from backend.utils.logger import get_logger
from backend.utils.pdf_export import generate_report

router = APIRouter(prefix="/report", tags=["report"])
logger = get_logger(__name__)


@router.post("")
async def create_report(payload: ReportRequest) -> dict[str, Any]:
    """Generate a PDF report and return path."""
    try:
        report_path = generate_report(Path("assets/report.pdf"), payload.summary, payload.scores)
    except Exception as exc:  # noqa: BLE001 - want to bubble as HTTP 500
        raise HTTPException(status_code=500, detail="Failed to generate report") from exc
    return {"report_path": str(report_path)}
