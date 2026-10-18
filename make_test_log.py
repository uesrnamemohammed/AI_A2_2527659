"""Run pytest -q and save the output as evidence/AI_A2_<REGDIGITS>_TEST_LOG.pdf (needs reportlab)."""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Preformatted, SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg-digits", required=True, help="Registration number without the slash, e.g. 2527659")
    args = parser.parse_args()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "not a git repository"
    styles = getSampleStyleSheet()
    mono = ParagraphStyle("mono", fontName="Courier", fontSize=8, leading=10)
    out = ROOT / "evidence" / f"AI_A2_{args.reg_digits}_TEST_LOG.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Test log")
    story = [
        Paragraph("Test log: pytest -q", styles["Title"]),
        Paragraph(escape(f"Date: {datetime.now().astimezone().isoformat(timespec='seconds')} | commit: {commit} | exit code: {result.returncode}"), styles["BodyText"]),
        Spacer(1, 10),
        Preformatted(result.stdout + result.stderr, mono),
    ]
    doc.build(story)
    print(f"Wrote {out} (pytest exit code {result.returncode})")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
