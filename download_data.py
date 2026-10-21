"""Download the real receipt datasets into real_raw/ (git-ignored). Run this once, then prepare_cord.py and prepare_sroie.py.

CORD v2   (naver-clova-ix/cord-v2, 100 validation receipts: OCR words + labels). Images are in the same
          parquet file, so this is ~240 MB even though the code never reads them.
SROIE     (ryanznie/SROIE_2019_with_labels, 973 receipts: OCR boxes + labels + invoice numbers). Text files only.

Licences: CORD is CC-BY-4.0. The SROIE copy's card says 'unknown' (the data is from the ICDAR 2019 challenge),
so treat it as research/learning use.
"""
import argparse
import os

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def download_cord(dest: str):
    snapshot_download("naver-clova-ix/cord-v2", repo_type="dataset", local_dir=dest,
                      allow_patterns=["data/validation-*.parquet"])


def download_sroie(dest: str):
    repo = "ryanznie/SROIE_2019_with_labels"
    wanted = [s.rfilename for s in HfApi().dataset_info(repo).siblings
              if "/box/" in s.rfilename or "/entities/" in s.rfilename
              or s.rfilename in ("train/labels.json", "test/test_labels.json")]
    # One file at a time: ~1900 tiny files, and a dropped connection only costs the file in flight.
    for f in wanted:
        if not os.path.exists(os.path.join(dest, f)):
            hf_hub_download(repo, f, repo_type="dataset", local_dir=dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cord", "sroie"])
    a = ap.parse_args()
    if a.only in (None, "cord"):
        download_cord("real_raw/cord")
    if a.only in (None, "sroie"):
        download_sroie("real_raw/sroie")
    print("done")
