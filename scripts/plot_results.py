"""Draw the figures of an evaluated study.

    python scripts/plot_results.py --study results/local --out figures
    python scripts/plot_results.py --study runs/smoke --out runs/smoke/figures
"""

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from mgn_vortex.plotting import plot_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True, type=Path, help="study directory with summary.json")
    parser.add_argument("--out", required=True, type=Path, help="directory for the PNG files")
    args = parser.parse_args()
    for path in plot_study(args.study, args.out):
        print(path)


if __name__ == "__main__":
    main()
