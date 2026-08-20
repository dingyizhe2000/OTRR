"""User-facing entry point for rebuilding empirical paper outputs."""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from otrank.results_analysis.reproduce_paper_outputs import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
