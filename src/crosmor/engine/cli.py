"""Command-line interface (CLI) for running CROSMOR simulations."""

import argparse
import sys
from pathlib import Path

from crosmor.io.inp_reader import read_inp
from crosmor.io.exporters import export_results_to_excel, export_profile_to_csv
from crosmor.engine.runner import SimulationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="CROSMOR 2026 - Coastal Morphodynamic Model")
    parser.add_argument("input_file", help="Path to CROSMOR .inp input file")
    parser.add_argument("--out-excel", help="Path to export results as Excel workbook (.xlsx)")
    parser.add_argument("--out-csv", help="Path to export final profile as CSV")

    args = parser.parse_args()
    inp_path = Path(args.input_file)
    if not inp_path.is_file():
        print(f"Error: Input file '{inp_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading configuration from {inp_path.name}...")
    cfg = read_inp(inp_path)
    print(f"  Title: {cfg.title}")
    print(f"  Tend: {cfg.morpho.tend} s, Bathymetry points: {len(cfg.bathymetry)}")

    runner = SimulationRunner(cfg)

    def print_progress(step_res, prog):
        print(
            f"\rStep {step_res.step_number:3d} | Time: {step_res.time:8.1f} s ({prog*100:5.1f}%) | "
            f"Balance: {step_res.volume.area_balance_m2:9.3f} m2",
            end="",
            flush=True,
        )

    print("\nStarting simulation...")
    results = runner.run(on_step_callback=print_progress)
    print("\nSimulation completed successfully!")

    if args.out_excel:
        print(f"Exporting to Excel: {args.out_excel}...")
        export_results_to_excel(results, args.out_excel)
    if args.out_csv and results.time_steps:
        print(f"Exporting final profile to CSV: {args.out_csv}...")
        export_profile_to_csv(results.time_steps[-1], args.out_csv)


if __name__ == "__main__":
    main()
