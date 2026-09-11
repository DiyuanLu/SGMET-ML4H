#!/usr/bin/env python3
"""Sequential overnight queue for the five-fold FT missingness experiment."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/ft_transformer_v5_matched_missingness_uniform_groups_seed42"
TRAIN = EXPERIMENTS / "ft_transformer_missingness_augmented_v5.py"
VALIDATE = EXPERIMENTS / "ft_transformer_missingness_validation_v5.py"
FINALIZE = EXPERIMENTS / "ft_transformer_missingness_finalize_v5.py"


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def wait_for_fold0_gate() -> dict:
    path = OUTPUT / "fold0/validation_robustness/DONE"
    while not path.exists():
        print("Waiting for the active fold-0 paired validation gate", flush=True)
        time.sleep(15)
    return json.loads(path.read_text())


def queue() -> None:
    gate = wait_for_fold0_gate()
    if not gate["fivefold_expansion_recommended"]:
        marker = OUTPUT / "QUEUE_STOPPED_FOLD0_GATE_FAILED"
        marker.write_text(json.dumps(gate, indent=2))
        print("Fold-0 gate failed. Queue stopped before additional training.", flush=True)
        return

    for fold in range(1, 5):
        if not (OUTPUT / f"fold{fold}/DONE").exists():
            run(str(TRAIN), "--fold", str(fold))
        if not (OUTPUT / f"fold{fold}/validation_robustness/DONE").exists():
            run(str(VALIDATE), "--fold", str(fold))

    run(str(FINALIZE), "--summarize-validation")
    (OUTPUT / "VALIDATION_COMPLETE").write_text(
        json.dumps({"folds": 5, "test_scored": False}, indent=2)
    )
    for fold in range(5):
        run(str(FINALIZE), "--test-fold", str(fold))
    run(str(FINALIZE), "--summarize-test")
    (OUTPUT / "TEST_COMPLETE").write_text(
        json.dumps({"folds": 5, "selection_or_tuning_from_test": False}, indent=2)
    )
    run(str(FINALIZE), "--package")
    print("FT missingness experiment and handoff package complete", flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT / "QUEUE.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another FT missingness queue is active") from error
        lock.write(str(os.getpid()))
        lock.flush()
        queue()


if __name__ == "__main__":
    main()
