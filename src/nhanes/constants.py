from __future__ import annotations

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

# NHANES cycle metadata:
# - folder: URL path segment used by CDC
# - suffix: file suffix used in XPT/HTM files
CYCLE_INFO = {
    "2011-2012": {"folder": "2011", "suffix": "G"},
    "2013-2014": {"folder": "2013", "suffix": "H"},
    "2015-2016": {"folder": "2015", "suffix": "I"},
    "2017-2018": {"folder": "2017", "suffix": "J"},
    "2019-2020": {"folder": "2017", "suffix": "P"},
    "2021-2023": {"folder": "2021", "suffix": "L"},
}

# Extended default module list. GLU / TRIGLY and KIQ_U are label feeders for
# CKM_Stage and TyG_WHtR; GHB, SMQ, and ALQ are retained as predictor modules.
DEFAULT_MODULES = [
    # Demographics
    "DEMO",
    # Body measures
    "BMX",
    # Blood pressure examination + questionnaire
    "BPX", "BPQ",
    # Diabetes questionnaire
    "DIQ",
    # Medical conditions questionnaire
    "MCQ",
    # Glycated hemoglobin
    "GHB",
    # Plasma glucose
    "GLU",
    # Triglycerides and LDL cholesterol
    "TRIGLY",
    # Kidney conditions (KIQ022 is a CKM_Stage feeder). Use bare "KIQ";
    # download_codebooks.py maps it and remaps to the KIQ_U file per cycle.
    "KIQ",
    # Smoking
    "SMQ",
    # Alcohol use
    "ALQ",
]


def module_file_prefix(module: str, cycle: str) -> str:
    """Return the CDC data-file prefix for a logical module name."""
    module = module.upper()
    if module == "BPX" and cycle in {"2019-2020", "2021-2023"}:
        return "BPXO"
    if module in {"KIQ", "KIQ_U"}:
        return "KIQ_U"
    return module
