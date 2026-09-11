from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import requests

from src.nhanes.constants import BASE_URL, CYCLE_INFO

# Imaging-related NHANES tabular modules.
# These XPT files usually contain imaging-derived measurements/metadata.
DEFAULT_DXA_MODULES = ["DXXFEM", "DXXSPN", "DXXLAA", "DXXLAB", "DXXHEA", "DXXAG"]
DEFAULT_FUNDUS_MODULES = ["OPXRET", "OPXFMR", "OPD"]


def download_binary(url: str, out_path: Path, timeout_s: int = 60) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=timeout_s)
    except requests.RequestException:
        return False
    if r.status_code != 200:
        return False
    out_path.write_bytes(r.content)
    return True


def download_xpt(url: str, out_path: Path, timeout_s: int = 60) -> bool:
    if not download_binary(url, out_path, timeout_s=timeout_s):
        return False
    # Some endpoints can return HTML pages with status 200.
    if not out_path.read_bytes().startswith(b"HEADER RECORD"):
        out_path.unlink(missing_ok=True)
        return False
    return True


def iter_xpt_targets(cycles: Iterable[str], modules: Iterable[str]) -> Iterable[tuple[str, str, str, str]]:
    for cycle in cycles:
        suffix = CYCLE_INFO[cycle]["suffix"]
        folder = CYCLE_INFO[cycle]["folder"]
        for module in modules:
            file_upper = f"{module}_{suffix}.XPT"
            file_lower = f"{module}_{suffix}.xpt"
            url = f"{BASE_URL}/{folder}/DataFiles/{file_lower}"
            yield cycle, module, url, file_upper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NHANES imaging modalities (DXA/Fundus) as available XPT files."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw_imaging"))
    parser.add_argument(
        "--cycles",
        nargs="+",
        default=["2011-2012", "2013-2014", "2015-2016", "2017-2018", "2019-2020"],
        choices=sorted(CYCLE_INFO.keys()),
        help="NHANES cycles to query for imaging modules.",
    )
    parser.add_argument(
        "--dxa-modules",
        nargs="+",
        default=DEFAULT_DXA_MODULES,
        help="DXA-related NHANES module prefixes.",
    )
    parser.add_argument(
        "--fundus-modules",
        nargs="+",
        default=DEFAULT_FUNDUS_MODULES,
        help="Fundus/ophthalmology-related NHANES module prefixes.",
    )
    parser.add_argument(
        "--archive-url",
        action="append",
        default=[],
        help="Optional direct URL(s) to image archives (zip/tar). Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = list(dict.fromkeys(args.dxa_modules + args.fundus_modules))

    success, failed = 0, 0
    print("Starting NHANES imaging module download (XPT)...")
    for cycle, module, url, filename in iter_xpt_targets(args.cycles, modules):
        out_path = args.out_dir / cycle / filename
        ok = download_xpt(url, out_path)
        if ok:
            success += 1
            print(f"[OK]   {cycle} {module} -> {out_path}")
        else:
            failed += 1
            print(f"[MISS] {cycle} {module} not found at: {url}")

    archive_ok, archive_fail = 0, 0
    if args.archive_url:
        print("Trying direct imaging archive URLs...")
    for raw_url in args.archive_url:
        url = raw_url.strip()
        if not url:
            continue
        filename = Path(url.split("?")[0]).name or "image_archive.bin"
        out_path = args.out_dir / "archives" / filename
        ok = download_binary(url, out_path)
        if ok:
            archive_ok += 1
            print(f"[OK]   archive -> {out_path}")
        else:
            archive_fail += 1
            print(f"[FAIL] archive URL unreachable: {url}")

    print(
        f"Finished. XPT success={success}, XPT missing={failed}, "
        f"archive_success={archive_ok}, archive_failed={archive_fail}"
    )
    print(
        "Note: many NHANES imaging data assets are metadata/derived tables; "
        "raw image files are not always publicly hosted in cycle DataFiles endpoints."
    )


if __name__ == "__main__":
    main()
