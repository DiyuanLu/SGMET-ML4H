from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import requests

from src.nhanes.constants import BASE_URL, CYCLE_INFO, DEFAULT_MODULES, module_file_prefix


def iter_targets(cycles: Iterable[str], modules: Iterable[str]) -> Iterable[tuple[str, str, str]]:
    for cycle in cycles:
        suffix = CYCLE_INFO[cycle]["suffix"]
        folder = CYCLE_INFO[cycle]["folder"]
        for module in modules:

            actual_module = module_file_prefix(module, cycle)
                
            # Handle the special 2019-2020 Pre-Pandemic dataset naming convention
            if cycle == "2019-2020" or suffix == "P":
                filename = f"P_{actual_module}.XPT"
                url = f"{BASE_URL}/{folder}/DataFiles/P_{actual_module}.xpt"
            else:
                # Standard format
                filename = f"{actual_module}_{suffix}.XPT"
                url = f"{BASE_URL}/{folder}/DataFiles/{actual_module}_{suffix}.xpt"

            # Yield the actual_module so it saves correctly locally
            yield cycle, actual_module, url
def download_file(url: str, out_path: Path, timeout_s: int = 60) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout_s)
    if r.status_code != 200:
        return False
    # Some CDC URLs return branded HTML "page not found" with HTTP 200.
    # XPT files start with "HEADER RECORD", so we use this guard.
    if not r.content.startswith(b"HEADER RECORD"):
        return False
    out_path.write_bytes(r.content)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NHANES XPT files for selected cycles/modules.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"), help="Output directory.")
    parser.add_argument(
        "--cycles",
        nargs="+",
        default=["2011-2012", "2013-2014", "2015-2016", "2017-2018", "2019-2020", "2021-2023"],
        choices=sorted(CYCLE_INFO.keys()),
        help="NHANES cycles to download.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=DEFAULT_MODULES,
        help="NHANES module prefixes (e.g., DEMO BMX BPX DIQ).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    success, failed = 0, 0
    print("Starting NHANES download...")
    for cycle, module, url in iter_targets(args.cycles, args.modules):
        suffix = CYCLE_INFO[cycle]["suffix"]
        if cycle == "2019-2020" or suffix == "P":
            filename = f"P_{module}.XPT"
        else:
            filename = f"{module}_{suffix}.XPT"
        out_path = args.out_dir / cycle / filename
        ok = download_file(url, out_path)
        if ok:
            success += 1
            print(f"[OK]   {cycle} {module} -> {out_path}")
        else:
            failed += 1
            print(f"[FAIL] {cycle} {module} (URL may not exist): {url}")
    print(f"Finished. Success={success}, Failed={failed}")


if __name__ == "__main__":
    main()
