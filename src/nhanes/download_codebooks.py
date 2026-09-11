from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from src.nhanes.constants import BASE_URL, CYCLE_INFO, DEFAULT_MODULES, module_file_prefix

VARIABLE_LIST_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx"

MODULE_COMPONENT = {
    "DEMO": "Demographics",
    "BMX": "Examination",
    "BPX": "Examination",
    "BPQ": "Questionnaire",
    "DIQ": "Questionnaire",
    "SMQ": "Questionnaire",
    "ALQ": "Questionnaire",
    "MCQ": "Questionnaire",
    "KIQ": "Questionnaire",
    "KIQ_U": "Questionnaire",
    "GHB": "Laboratory",
    "TRIGLY": "Laboratory",
    "GLU": "Laboratory",
}

CYCLE_BEGIN_YEAR = {
    "2011-2012": 2011,
    "2013-2014": 2013,
    "2015-2016": 2015,
    "2017-2018": 2017,
    # Trigger the special 'Cycle' parameter instead of 'CycleBeginYear'
    "2019-2020": "2017-2020", 
    "2021-2023": "2021-2023",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NHANES variable metadata for LLM encoding from NHANES variable list pages.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/external/codebooks"))
    parser.add_argument(
        "--cycles",
        nargs="+",
        default=["2011-2012", "2013-2014", "2015-2016", "2017-2018", "2019-2020", "2021-2023"],
        choices=sorted(CYCLE_INFO.keys()),
    )
    parser.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    parser.add_argument(
        "--skip-detailed-codebooks",
        action="store_true",
        help="Only write the legacy variable-list metadata, without parsing full CDC codebook pages.",
    )
    parser.add_argument("--timeout-s", type=int, default=60, help="HTTP timeout for detailed codebook pages.")
    return parser.parse_args()

def load_variable_list(component: str, begin_year: int | str) -> pd.DataFrame | None:
    # Dynamically handle the CDC's URL parameter inconsistency
    if isinstance(begin_year, str) and "-" in begin_year:
        url = f"{VARIABLE_LIST_URL}?Component={component}&Cycle={begin_year}"
    else:
        url = f"{VARIABLE_LIST_URL}?Component={component}&CycleBeginYear={begin_year}"
        
    try:
        tables = pd.read_html(url)
    except ValueError:
        return None
    if not tables:
        return None
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Variable Name" not in df.columns or "Variable Description" not in df.columns or "Data File Name" not in df.columns:
        return None
    df["source_url"] = url
    return df

def codebook_url(cycle: str, file_name: str) -> str:
    folder = CYCLE_INFO[cycle]["folder"]
    return f"{BASE_URL}/{folder}/DataFiles/{file_name}.htm"

def load_codebook_page(cycle: str, file_name: str, timeout_s: int) -> BeautifulSoup | None:
    url = codebook_url(cycle, file_name)
    try:
        response = requests.get(url, timeout=timeout_s)
    except requests.RequestException as exc:
        print(f"[WARN] Failed to fetch detailed codebook {file_name}: {exc}")
        return None
    if response.status_code != 200:
        print(f"[WARN] Detailed codebook returned HTTP {response.status_code}: {url}")
        return None
    return BeautifulSoup(response.text, "lxml")

def parse_codebook_page(soup: BeautifulSoup) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    page_sections = parse_page_sections(soup)
    page_sections.update(parse_publication_metadata(soup))
    variables: dict[str, dict[str, object]] = {}
    codebook_order = 0

    for heading in soup.find_all(["h3", "h4"]):
        heading_text = clean_text(heading.get_text(" ", strip=True))
        variable_match = re.match(r"^([A-Za-z][A-Za-z0-9_]+)\s+-\s+(.+)$", heading_text)
        if not variable_match:
            continue

        codebook_order += 1
        variable_name = variable_match.group(1).upper()
        heading_label = variable_match.group(2).strip()
        block_elements = elements_until_next_variable(heading)
        block_lines = element_lines(block_elements)
        fields = parse_variable_fields(block_lines)
        value_labels, skip_rules = parse_frequency_rows(block_elements, block_lines)

        sas_label = fields.get("SAS Label") or heading_label
        english_text = fields.get("English Text")
        units = extract_unit(sas_label) or extract_unit(english_text) or extract_unit(heading_label)
        detailed_description = build_detailed_description(
            sas_label=sas_label,
            english_text=english_text,
            page_sections=page_sections,
        )

        variables[variable_name] = {
            "codebook_order": codebook_order,
            "codebook_heading": heading_label,
            "sas_label": sas_label,
            "english_text": english_text,
            "target": fields.get("Target"),
            "hard_edits": fields.get("Hard Edits"),
            "unit": units,
            "detailed_description": detailed_description,
            "value_labels": value_labels,
            "skip_rules": skip_rules,
        }

    return page_sections, variables

def parse_page_sections(soup: BeautifulSoup) -> dict[str, str]:
    wanted = {
        "Component Description",
        "Eligible Sample",
        "Protocol and Procedure",
        "Interview Setting and Mode of Administration",
        "Quality Assurance & Quality Control",
        "Data Processing and Editing",
        "Analytic Notes",
    }
    sections: dict[str, str] = {}
    for heading in soup.find_all("h2"):
        title = clean_text(heading.get_text(" ", strip=True))
        if title not in wanted:
            continue
        texts = []
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and sibling.name == "h2":
                break
            if isinstance(sibling, Tag):
                texts.append(sibling.get_text(" ", strip=True))
        text = clean_text(" ".join(texts))
        if text:
            sections[section_key(title)] = text
    return sections

def parse_publication_metadata(soup: BeautifulSoup) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for heading in soup.find_all(["h3", "h4", "h5"]):
        text = clean_text(heading.get_text(" ", strip=True))
        if text.startswith("First Published:"):
            metadata["first_published"] = text.removeprefix("First Published:").strip()
        elif text.startswith("Last Revised:"):
            metadata["last_revised"] = text.removeprefix("Last Revised:").strip()
        elif text.startswith("Data File:"):
            metadata["data_file"] = text.removeprefix("Data File:").strip()
    return metadata

def elements_until_next_variable(heading: Tag) -> list[Tag]:
    elements = []
    for sibling in heading.next_siblings:
        if not isinstance(sibling, Tag):
            continue
        if sibling.name in {"h2", "h3", "h4"}:
            break
        elements.append(sibling)
    return elements

def element_lines(elements: list[Tag]) -> list[str]:
    lines: list[str] = []
    for element in elements:
        lines.extend(clean_text(text) for text in element.stripped_strings)
    return [line for line in lines if line]

def parse_variable_fields(lines: list[str]) -> dict[str, str]:
    labels = {"Variable Name", "SAS Label", "English Text", "Target", "Hard Edits"}
    fields: dict[str, str] = {}
    current_label: str | None = None
    current_values: list[str] = []

    def flush() -> None:
        if current_label is not None and current_values:
            fields[current_label] = clean_text(" ".join(current_values))

    for line in lines:
        label = line[:-1] if line.endswith(":") else None
        if label in labels:
            flush()
            current_label = label
            current_values = []
            continue
        if line.startswith("Code or Value"):
            flush()
            current_label = None
            current_values = []
            continue
        if current_label is not None:
            current_values.append(line)
    flush()
    return fields

def parse_frequency_rows(elements: list[Tag], lines: list[str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    table_labels, table_skip_rules = parse_frequency_rows_from_tables(elements)
    if table_labels or table_skip_rules:
        return table_labels, table_skip_rules
    return parse_frequency_rows_from_lines(lines)

def parse_frequency_rows_from_tables(elements: list[Tag]) -> tuple[dict[str, str], list[dict[str, str]]]:
    value_labels: dict[str, str] = {}
    skip_rules: list[dict[str, str]] = []
    tables = []
    for element in elements:
        if element.name == "table":
            tables.append(element)
        tables.extend(element.find_all("table"))
    for table in tables:
        rows = []
        for tr in table.find_all("tr"):
            cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        header = [cell.lower() for cell in rows[0]]
        code_idx = find_header_index(header, ["code or value", "code", "value"])
        description_idx = find_header_index(header, ["value description", "description"])
        skip_idx = find_header_index(header, ["skip to item", "skip to"])
        if code_idx is None or description_idx is None:
            continue
        for row in rows[1:]:
            if len(row) <= max(code_idx, description_idx):
                continue
            code = row[code_idx]
            description = row[description_idx]
            if code and description:
                value_labels[code] = description
            if skip_idx is not None and len(row) > skip_idx:
                skip_to = normalize_skip_destination(row[skip_idx])
                if code and skip_to:
                    skip_rules.append(
                        {
                            "source_value": code,
                            "source_value_label": description,
                            "skip_to": skip_to,
                        }
                    )
    return value_labels, skip_rules

def parse_frequency_rows_from_lines(lines: list[str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    value_labels: dict[str, str] = {}
    skip_rules: list[dict[str, str]] = []
    in_values = False
    for line in lines:
        if line.startswith("Code or Value"):
            in_values = True
            continue
        if not in_values:
            continue
        if line.endswith(":") or line in {"Variable Name", "SAS Label", "English Text", "Target", "Hard Edits"}:
            break
        match = re.match(r"^(\S+)\s+(.+?)\s+[-+]?\d[\d,]*(?:\.\d+)?\s+[-+]?\d[\d,]*(?:\.\d+)?(?:\s+.*)?$", line)
        if match:
            code = match.group(1)
            value_labels[code] = clean_text(match.group(2))
            skip_to = extract_skip_destination_from_line(line)
            if skip_to:
                skip_rules.append(
                    {
                        "source_value": code,
                        "source_value_label": clean_text(match.group(2)),
                        "skip_to": skip_to,
                    }
                )
    return value_labels, skip_rules

def normalize_skip_destination(value: str | None) -> str | None:
    if not value:
        return None
    value = clean_text(value)
    if not value:
        return None
    match = re.search(r"\b([A-Z][A-Z0-9_]+)\b", value)
    return match.group(1) if match else None

def extract_skip_destination_from_line(line: str) -> str | None:
    candidates = re.findall(r"\b[A-Z][A-Z0-9_]+\b", line)
    return candidates[-1] if candidates else None

def find_header_index(header: list[str], candidates: list[str]) -> int | None:
    for candidate in candidates:
        for idx, value in enumerate(header):
            if candidate == value or candidate in value:
                return idx
    return None

def build_detailed_description(
    sas_label: str | None,
    english_text: str | None,
    page_sections: dict[str, str],
) -> str | None:
    parts = []
    if english_text and english_text != sas_label:
        parts.append(english_text)
    elif sas_label:
        parts.append(sas_label)
    for key in ["component_description", "data_processing_and_editing", "analytic_notes"]:
        value = page_sections.get(key)
        if value:
            parts.append(value)
    return clean_text(" ".join(parts)) if parts else None

def extract_unit(text: str | None) -> str | None:
    if not text:
        return None
    matches = re.findall(r"\(([^()]{1,40})\)", text)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if looks_like_unit(candidate):
            return candidate
    return None

def looks_like_unit(candidate: str) -> bool:
    lowered = candidate.lower()
    unit_terms = [
        "%",
        "/",
        "kg",
        "cm",
        "mm",
        "mg",
        "g",
        "ml",
        "dl",
        "l",
        "iu",
        "ng",
        "pg",
        "umol",
        "µmol",
        "mmol",
        "years",
        "months",
    ]
    return any(term in lowered for term in unit_terms) and not any(char in candidate for char in "?!")

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

def section_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

def enrich_with_detailed_codebooks(metadata: pd.DataFrame, timeout_s: int) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    enriched = metadata.copy()
    detail_columns = [
        "documentation_url",
        "codebook_order",
        "codebook_heading",
        "sas_label",
        "english_text",
        "target",
        "hard_edits",
        "unit",
        "detailed_description",
        "value_labels_json",
        "skip_rules_json",
        "data_file",
        "first_published",
        "last_revised",
        "component_description",
        "eligible_sample",
        "protocol_and_procedure",
        "interview_setting_and_mode_of_administration",
        "quality_assurance_quality_control",
        "data_processing_and_editing",
        "analytic_notes",
    ]
    for column in detail_columns:
        enriched[column] = None

    category_maps: dict[str, dict[str, str]] = {}
    for (cycle, file_name), group in enriched.groupby(["cycle", "file_name"], dropna=False):
        if pd.isna(cycle) or pd.isna(file_name):
            continue
        if str(cycle) not in CYCLE_INFO:
            print(f"[INFO] Skipping detailed codebook for non-CDC cycle/file: {cycle} {file_name}")
            continue
        url = codebook_url(str(cycle), str(file_name))
        soup = load_codebook_page(str(cycle), str(file_name), timeout_s=timeout_s)
        if soup is None:
            continue
        page_sections, variables = parse_codebook_page(soup)
        mask = (enriched["cycle"] == cycle) & (enriched["file_name"] == file_name)
        enriched.loc[mask, "documentation_url"] = url
        for key, value in page_sections.items():
            if key in enriched.columns:
                enriched.loc[mask, key] = value

        for idx in group.index:
            variable_name = str(enriched.at[idx, "variable_name"])
            variable = variables.get(variable_name)
            if variable is None:
                continue
            for column in [
                "codebook_order",
                "codebook_heading",
                "sas_label",
                "english_text",
                "target",
                "hard_edits",
                "unit",
                "detailed_description",
            ]:
                enriched.at[idx, column] = variable.get(column)
            value_labels = variable.get("value_labels") or {}
            if value_labels:
                map_key = f"{cycle}|{file_name}|{variable_name}"
                category_maps[map_key] = value_labels
                enriched.at[idx, "value_labels_json"] = json.dumps(value_labels, ensure_ascii=False, sort_keys=True)
            skip_rules = variable.get("skip_rules") or []
            if skip_rules:
                enriched.at[idx, "skip_rules_json"] = json.dumps(skip_rules, ensure_ascii=False, sort_keys=True)
        print(f"[OK] Parsed detailed codebook: {cycle} {file_name} variables={len(variables)}")

    return enriched, category_maps

def write_parquet_if_available(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        print(f"[WARN] Could not write parquet {path}: {exc}")
        return False
    return True

def main() -> None:
    args = parse_args()
    records = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    for cycle in args.cycles:
        suffix = CYCLE_INFO[cycle]["suffix"]
        begin_year = CYCLE_BEGIN_YEAR[cycle]
        
        for module in args.modules:
            component = MODULE_COMPONENT.get(module)
            if component is None:
                print(f"[WARN] No component mapping for module: {module}")
                continue
                
            var_df = load_variable_list(component, begin_year)
            if var_df is None:
                print(f"[WARN] Could not parse variable list for {cycle} {module}")
                continue
                
            actual_module = module_file_prefix(module, cycle)
                
            # Handle special 2019-2020 Pre-Pandemic P_ naming
            if cycle == "2019-2020" or suffix == "P":
                file_name_download = f"P_{actual_module}"
                file_name_table_variant1 = f"P_{actual_module}"
                file_name_table_variant2 = f"{actual_module}_P"
                
                filtered = var_df[var_df["Data File Name"].astype(str).str.upper() == file_name_table_variant1].copy()
                
                if filtered.empty:
                    filtered = var_df[var_df["Data File Name"].astype(str).str.upper() == file_name_table_variant2].copy()
                
                file_name = file_name_download
                
            else:
                file_name = f"{actual_module}_{suffix}"
                filtered = var_df[var_df["Data File Name"].astype(str).str.upper() == file_name].copy()
                
            if filtered.empty:
                print(f"[WARN] No matching rows for file {file_name} in {cycle}")
                continue
                
            filtered["cycle"] = cycle
            filtered["module"] = actual_module
            filtered["file_name"] = file_name
            filtered = filtered.rename(
                columns={
                    "Variable Name": "variable_name",
                    "Variable Description": "variable_description",
                    "Data File Description": "file_description",
                    "Begin Year": "begin_year",
                    "EndYear": "end_year",
                }
            )
            filtered["variable_name"] = filtered["variable_name"].astype(str).str.upper()
            keep_cols = [
                "cycle",
                "module",
                "file_name",
                "variable_name",
                "variable_description",
                "file_description",
                "begin_year",
                "end_year",
                "Component",
                "source_url",
            ]
            records.append(filtered[keep_cols])
            print(f"[OK] Extracted metadata: {cycle} {module} rows={len(filtered)}")

    if not records:
        raise RuntimeError("No codebooks parsed. Check cycle/module selections.")

    all_df = pd.concat(records, ignore_index=True)

    metadata_parquet = args.out_dir / "nhanes_metadata.parquet"
    if args.skip_detailed_codebooks:
        wrote_parquet = write_parquet_if_available(all_df, metadata_parquet)
        if wrote_parquet:
            print(f"[OK] Saved metadata parquet: {metadata_parquet} rows={len(all_df)}")
        return

    enriched_df, category_maps = enrich_with_detailed_codebooks(all_df, timeout_s=args.timeout_s)
    category_json = args.out_dir / "nhanes_category_label_maps.json"
    wrote_enriched_parquet = write_parquet_if_available(enriched_df, metadata_parquet)
    category_json.write_text(json.dumps(category_maps, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    if wrote_enriched_parquet:
        print(f"[OK] Saved metadata parquet: {metadata_parquet} rows={len(enriched_df)}")
    print(f"[OK] Saved category label maps JSON: {category_json} maps={len(category_maps)}")

if __name__ == "__main__":
    main()
