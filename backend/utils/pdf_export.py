"""PDF export helper using fpdf2."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fpdf import FPDF

from .logger import get_logger

logger = get_logger(__name__)


class AnalysisReport(FPDF):
    """PDF report for audio analysis."""

    def header(self) -> None:  # type: ignore[override]
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Real-Time Audio Deepfake Detection Report", ln=True, align="C")
        self.ln(5)

    def footer(self) -> None:  # type: ignore[override]
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_report(output_path: Path, summary: Dict[str, str], scores: Dict[str, float]) -> Path:
    """Generate a concise PDF report summarizing scores.

    Args:
        output_path: Where to save the PDF.
        summary: Key textual summary.
        scores: Score mapping.
    Returns:
        Path to the saved PDF.
    """
    pdf = AnalysisReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    pdf.cell(0, 10, "Summary", ln=True)
    for key, text in summary.items():
        pdf.multi_cell(0, 8, f"- {key}: {text}")

    pdf.ln(5)
    pdf.cell(0, 10, "Scores", ln=True)
    for key, value in scores.items():
        pdf.cell(0, 8, f"- {key}: {value:.2f}", ln=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    logger.info("Generated PDF report at %s", output_path)
    return output_path
