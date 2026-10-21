"""Turn the CORD v2 receipts (real OCR text + labels) into the same layout the rest of the project uses:

    data_real/cord_000.txt ...   one document per file (OCR words rebuilt into lines)
    data_real/truth.json         expected fields per document (entries from prepare_sroie.py are kept)

Reads only the `ground_truth` column of the parquet files, so the images are never loaded.
"""
import argparse
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq


def ocr_lines(valid_line: list[dict]) -> str:
    """Rebuild receipt text: group words by printed row, left to right, rows top to bottom."""
    rows: dict[int, list[tuple[int, str]]] = {}
    top: dict[int, int] = {}
    for item in valid_line:
        for w in item["words"]:
            q = w["quad"]
            rows.setdefault(w["row_id"], []).append((q["x1"], w["text"]))
            top[w["row_id"]] = min(top.get(w["row_id"], 10**9), q["y1"])
    return "\n".join(" ".join(t for _, t in sorted(rows[r])) for r in sorted(rows, key=top.get))


def truth_of(gt_parse: dict) -> dict:
    total = gt_parse.get("total") or {}
    sub = gt_parse.get("sub_total") or {}
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):   # CORD stores a single item as a dict instead of a list
        menu = [menu]
    return {
        "total": total.get("total_price"),
        "subtotal": sub.get("subtotal_price"),
        "tax": sub.get("tax_price"),
        "item_count": len(menu),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="real_raw/cord/data")
    ap.add_argument("--out", default="data_real")
    ap.add_argument("--split", default="validation", help="validation (100 docs) or train (800 docs)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truth_path = out / "truth.json"
    others = [t for t in json.loads(truth_path.read_text()) if not t["id"].startswith("cord_")] if truth_path.exists() else []
    truth = []
    for path in sorted(glob.glob(f"{a.src}/{a.split}-*.parquet")):
        for s in pq.read_table(path, columns=["ground_truth"]).column("ground_truth").to_pylist():
            doc = json.loads(s)
            doc_id = f"cord_{len(truth):03d}"
            (out / f"{doc_id}.txt").write_text(ocr_lines(doc["valid_line"]), encoding="utf-8")
            truth.append({"id": doc_id, "expected": truth_of(doc["gt_parse"])})
    truth_path.write_text(json.dumps(truth + others, indent=2))   # keep the SROIE entries if they are already there
    print(f"wrote {len(truth)} CORD documents, truth.json now has {len(truth) + len(others)} entries")


if __name__ == "__main__":
    main()
