"""Turn the SROIE 2019 receipts (real OCR text + labels) into the project's document layout.

    data_real/sroie_000.txt ...   one document per file (OCR text, box coordinates stripped)
    data_real/truth.json          merged with the CORD entries already there

Labels: company, date, total (SROIE) and invoice_number (ryanznie's extra labels). A field that has no
usable label is stored as null, and the evaluation should skip it instead of counting it wrong.
"""
import argparse
import json
import statistics
from pathlib import Path

NO_LABEL = {"", "ambiguous"}


def ocr_text(box_file: Path) -> str:
    """Each line is 'x1,y1,x2,y2,x3,y3,x4,y4,text'. The text itself may contain commas."""
    lines = []
    for line in box_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(",", 8)
        if len(parts) == 9 and parts[8].strip():
            lines.append(parts[8].strip())
    return "\n".join(lines)


def ocr_rows(box_file: Path) -> str:
    """Rebuild the printed rows from the box coordinates. The file lists a label and its amount as separate boxes,
    and often all the labels first and then all the amounts ('NET TOTAL / CASH / CHANGE' ... '39.80 / 40.00 / 0.20'),
    which is hard to read for a person and harder for a small model. Boxes whose vertical centres are close
    belong to one row; each row is written left to right."""
    boxes = []   # (y centre, box height, x left, text)
    for line in box_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(",", 8)
        if len(parts) < 9 or not parts[8].strip():
            continue
        try:
            xs, ys = [int(v) for v in parts[0:8:2]], [int(v) for v in parts[1:8:2]]
        except ValueError:
            continue
        boxes.append((sum(ys) / 4, max(ys) - min(ys), min(xs), parts[8].strip()))
    if not boxes:
        return ""
    tolerance = 0.6 * statistics.median(h for _, h, _, _ in boxes)   # a fraction of a typical text height
    rows: list[list[tuple]] = []
    for box in sorted(boxes):
        if rows and abs(box[0] - statistics.fmean(b[0] for b in rows[-1])) <= tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    return "\n".join(" ".join(b[3] for b in sorted(row, key=lambda b: b[2])) for row in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="real_raw/sroie")
    ap.add_argument("--out", default="data_real")
    ap.add_argument("--layout", choices=["lines", "rows"], default="lines",
                    help="lines: one OCR box per line (data_real). rows: boxes on the same printed row joined (use with --out data_rows)")
    ap.add_argument("--splits", nargs="+", default=["test", "train"], help="test is 347 docs, train is 626")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truth_path = out / "truth.json"
    truth = [t for t in json.loads(truth_path.read_text()) if not t["id"].startswith("sroie_")] if truth_path.exists() else []

    n = 0
    for split in a.splits:
        labels_file = src / split / ("labels.json" if split == "train" else "test_labels.json")
        invoice_numbers = json.loads(labels_file.read_text())
        for box in sorted((src / split / "box").glob("*.txt")):
            ent = json.loads((src / split / "entities" / box.name).read_text())
            inv = invoice_numbers.get(box.stem + ".jpg")
            doc_id = f"sroie_{n:03d}"
            (out / f"{doc_id}.txt").write_text(ocr_rows(box) if a.layout == "rows" else ocr_text(box), encoding="utf-8")
            truth.append({"id": doc_id, "source": f"sroie/{split}/{box.stem}", "expected": {
                "supplier_name": ent.get("company") or None,
                "invoice_date": ent.get("date") or None,
                "total": ent.get("total") or None,
                "invoice_number": None if inv in NO_LABEL else inv,
            }})
            n += 1
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"wrote {n} SROIE documents, truth.json now has {len(truth)} entries")


if __name__ == "__main__":
    main()
