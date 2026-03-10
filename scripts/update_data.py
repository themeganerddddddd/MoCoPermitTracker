import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATASET_ID = "i26v-w6bd"

SOURCE_URLS = [
    f"https://data.montgomerycountymd.gov/api/views/{DATASET_ID}/rows.csv?accessType=DOWNLOAD",
    f"https://data.montgomerycountymd.gov/resource/{DATASET_ID}.csv?$limit=50000",
]

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "permits.json")

TARGET_USECODES = {
    "BIOSCIENCE",
    "BUSINESS BUILDING",
    "INDUSTRIAL BUILDING",
}

REQUEST_TIMEOUT = 60


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def fetch_csv_text() -> str:
    last_error = None

    headers = {
        "User-Agent": "Mozilla/5.0 PermitTracker/1.0"
    }

    for url in SOURCE_URLS:
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if resp.text and "permitno" in resp.text.lower():
                return resp.text
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to fetch dataset from all known endpoints. Last error: {last_error}")


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_location(row: Dict[str, Any]) -> str:
    parts = [
        clean_text(row.get("stno")),
        clean_text(row.get("predir")),
        clean_text(row.get("stname")),
        clean_text(row.get("suffix")),
        clean_text(row.get("postdir")),
        clean_text(row.get("city")),
        clean_text(row.get("state")),
        clean_text(row.get("zip")),
    ]
    parts = [p for p in parts if p]

    if row.get("location") and str(row.get("location")).strip():
        return clean_text(row.get("location"))

    return " ".join(parts)


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None

    s = str(value).strip().replace(",", "").replace("$", "")
    if not s:
        return None

    try:
        return float(s)
    except ValueError:
        return None


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"${value:,.0f}" if value.is_integer() else f"${value:,.2f}"


def format_sqft(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value.is_integer():
        return f"{int(value):,} sf"
    return f"{value:,.2f} sf"


def parse_date_string(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    return s


def normalize_usecode(value: str) -> str:
    return clean_text(value).upper()


def sort_key_addeddate(item: Dict[str, Any]) -> str:
    return item.get("addeddate_raw", "") or ""


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:
    print("Fetching permit data...")
    csv_text = fetch_csv_text()

    reader = csv.DictReader(io.StringIO(csv_text))
    filtered: List[Dict[str, Any]] = []

    for row in reader:
        usecode_raw = clean_text(row.get("usecode"))
        usecode_norm = normalize_usecode(usecode_raw)

        if usecode_norm not in TARGET_USECODES:
            continue

        declaredvaluation_num = parse_number(row.get("declaredvaluation"))
        buildingarea_num = parse_number(row.get("buildingarea"))
        worktype = clean_text(row.get("worktype")) or "—"

        permit = {
            "location": build_location(row),
            "status": clean_text(row.get("status")) or "—",
            "usecode": usecode_raw or "—",
            "declaredvaluation": format_currency(declaredvaluation_num),
            "declaredvaluation_num": declaredvaluation_num or 0,
            "buildingarea": format_sqft(buildingarea_num),
            "buildingarea_num": buildingarea_num or 0,
            "worktype": worktype,
            "description": clean_text(row.get("description")) or "—",
            "addeddate": parse_date_string(row.get("addeddate")),
            "issueddate": parse_date_string(row.get("issueddate")),
            "finaleddate": parse_date_string(row.get("finaleddate")),
            "addeddate_raw": parse_date_string(row.get("addeddate")),
            "issueddate_raw": parse_date_string(row.get("issueddate")),
            "finaleddate_raw": parse_date_string(row.get("finaleddate")),
        }

        filtered.append(permit)

    filtered.sort(key=sort_key_addeddate, reverse=True)

    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "source_dataset": DATASET_ID,
        "record_count": len(filtered),
        "permits": filtered,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(filtered):,} matching permits to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
