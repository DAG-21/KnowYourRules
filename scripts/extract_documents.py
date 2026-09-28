"""
Downloads government documents listed in research/india_traffic_document_links.xlsx
into data/raw/, and records provenance for every file in data/raw/manifest.csv.

Usage:
    python scripts/extract_documents.py --dry-run
    python scripts/extract_documents.py --limit 5
    python scripts/extract_documents.py --category Central
    python scripts/extract_documents.py --statuses "Verified - Live,Verified - Live (needs special handling)" --insecure-tls
"""

import argparse
import csv
import hashlib
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import openpyxl
import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; IndiaTrafficRulesRAG-DocCollector/1.0)"
DEFAULT_STATUSES = ["Verified - Live"]
MANIFEST_FIELDS = [
    "category", "source_name", "page_url", "file_url", "local_path",
    "http_status", "bytes", "sha256", "valid_pdf", "error", "downloaded_at",
]


def load_rows(xlsx_path: Path, sheet_name: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = []
    for excel_row in ws.iter_rows(min_row=2, values_only=True):
        row = dict(zip(headers, excel_row))
        if row.get("Source Name"):
            rows.append(row)
    return rows


def filter_rows(rows: list[dict], statuses: list[str], category: str | None) -> list[dict]:
    out = []
    for row in rows:
        if row.get("Verification Status") not in statuses:
            continue
        if category and row.get("Category / Hierarchy Level") != category:
            continue
        if not row.get("Document URL"):
            continue
        out.append(row)
    return out


def split_urls(url_field: str) -> list[str]:
    """Split a pipe-separated Document URL cell into absolute URLs.

    Some rows shorthand later entries as a path relative to the first URL's
    domain (e.g. "https://x.gov.in/acts/ | /notifications/"). Those are
    resolved with urljoin. Anything that's neither absolute nor a leading-
    slash relative path (a placeholder template, or a malformed fragment)
    is skipped with a warning rather than guessed at.
    """
    parts = [p.strip() for p in re.split(r"\s*\|\s*", url_field.strip()) if p.strip()]
    resolved = []
    base = None
    for p in parts:
        if "<" in p or ">" in p:
            continue
        if p.startswith("http://") or p.startswith("https://"):
            base = p
            resolved.append(p)
        elif p.startswith("/") and base:
            resolved.append(urljoin(base, p))
        else:
            print(f"  ? skipping unresolvable URL fragment: {p!r}")
    return resolved


def sanitize(name: str, max_len: int = 60) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name).strip("_.")
    return name[:max_len] or "unnamed"


def slugify(text: str, max_len: int = 60) -> str:
    """Lowercase, hyphen-separated identifier safe for filenames and folder names."""
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len].rstrip("-") or "unnamed"


def build_filename(row: dict, file_url: str) -> str:
    """<source-slug>__<original-name-slug>-<urlhash>.pdf

    The URL hash is mandatory, not just a collision fallback: many government
    sites serve distinct documents through a generic path (e.g. download.aspx)
    disambiguated only by a query string (?id=...), which the path-based slug
    alone would collapse into one filename and silently skip as "duplicate".
    Hashing the full URL (query string included) guarantees one file per
    distinct source URL while re-runs stay idempotent.
    """
    source_slug = slugify(row.get("Source Name", "unknown-source"), max_len=40)
    stem = Path(urlparse(file_url).path).stem or "document"
    name_slug = slugify(stem, max_len=50)
    url_hash = hashlib.sha1(file_url.encode()).hexdigest()[:8]
    return f"{source_slug}__{name_slug}-{url_hash}.pdf"


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def find_pdf_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower():
            links.add(urljoin(base_url, href))
    return sorted(links)


def fetch(session: requests.Session, url: str, timeout: int, insecure: bool):
    verify = not insecure
    if insecure:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    return session.get(url, timeout=timeout, verify=verify, allow_redirects=True)


def save_file(content: bytes, dest_path: Path) -> tuple[int, str, bool]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    valid_pdf = content[:4] == b"%PDF"
    return len(content), sha256, valid_pdf


def unique_path(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    i = 2
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def download_one(
    session: requests.Session,
    file_url: str,
    page_url: str,
    dest_dir: Path,
    row: dict,
    timeout: int,
    insecure: bool,
    overwrite: bool,
) -> dict:
    entry = {
        "category": row.get("Category / Hierarchy Level", ""),
        "source_name": row.get("Source Name", ""),
        "page_url": page_url,
        "file_url": file_url,
        "local_path": "",
        "http_status": "",
        "bytes": "",
        "sha256": "",
        "valid_pdf": "",
        "error": "",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        filename = build_filename(row, file_url)
        dest_path = dest_dir / filename
        if dest_path.exists() and not overwrite:
            entry["local_path"] = str(dest_path)
            entry["error"] = "skipped_existing"
            return entry
        if dest_path.exists() and overwrite:
            dest_path = unique_path(dest_dir, filename)

        resp = fetch(session, file_url, timeout, insecure)
        entry["http_status"] = resp.status_code
        if resp.status_code != 200:
            entry["error"] = f"http_{resp.status_code}"
            return entry

        dest_path = unique_path(dest_dir, filename) if dest_path.exists() else dest_path
        size, sha256, valid_pdf = save_file(resp.content, dest_path)
        entry["local_path"] = str(dest_path)
        entry["bytes"] = size
        entry["sha256"] = sha256
        entry["valid_pdf"] = valid_pdf
        if not valid_pdf:
            entry["error"] = "downloaded_but_not_pdf_magic_bytes"
    except requests.exceptions.SSLError as e:
        entry["error"] = f"ssl_error (retry with --insecure-tls): {e}"
    except requests.exceptions.RequestException as e:
        entry["error"] = f"request_error: {e}"
    return entry


def process_row(
    session: requests.Session,
    row: dict,
    output_root: Path,
    timeout: int,
    insecure: bool,
    overwrite: bool,
    delay: float,
    dry_run: bool,
) -> list[dict]:
    category_dir = slugify(row.get("Category / Hierarchy Level", "uncategorized"), max_len=45)
    dest_dir = output_root / category_dir

    manifest_entries = []
    for page_url in split_urls(row.get("Document URL", "")):
        if is_pdf_url(page_url):
            if dry_run:
                print(f"  [dry-run] would download PDF: {page_url}")
                continue
            entry = download_one(session, page_url, page_url, dest_dir, row, timeout, insecure, overwrite)
            manifest_entries.append(entry)
            _report(entry)
            time.sleep(delay)
            continue

        # Treat as a listing/document-category page: fetch it and pull PDF links out.
        try:
            resp = fetch(session, page_url, timeout, insecure)
        except requests.exceptions.SSLError as e:
            entry = {**_blank_entry(row, page_url), "error": f"ssl_error (retry with --insecure-tls): {e}"}
            manifest_entries.append(entry)
            _report(entry)
            continue
        except requests.exceptions.RequestException as e:
            entry = {**_blank_entry(row, page_url), "error": f"request_error: {e}"}
            manifest_entries.append(entry)
            _report(entry)
            continue

        if resp.status_code != 200:
            entry = {**_blank_entry(row, page_url), "http_status": resp.status_code, "error": f"http_{resp.status_code}"}
            manifest_entries.append(entry)
            _report(entry)
            continue

        pdf_links = find_pdf_links(resp.text, page_url)
        if not pdf_links:
            entry = {
                **_blank_entry(row, page_url),
                "http_status": resp.status_code,
                "error": "no_pdf_links_found_on_page (may be JS-rendered - needs manual review)",
            }
            manifest_entries.append(entry)
            _report(entry)
            continue

        for file_url in pdf_links:
            if dry_run:
                print(f"  [dry-run] would download PDF: {file_url}  (found on {page_url})")
                continue
            entry = download_one(session, file_url, page_url, dest_dir, row, timeout, insecure, overwrite)
            manifest_entries.append(entry)
            _report(entry)
            time.sleep(delay)

    return manifest_entries


def _blank_entry(row: dict, page_url: str) -> dict:
    return {
        "category": row.get("Category / Hierarchy Level", ""),
        "source_name": row.get("Source Name", ""),
        "page_url": page_url,
        "file_url": "",
        "local_path": "",
        "http_status": "",
        "bytes": "",
        "sha256": "",
        "valid_pdf": "",
        "error": "",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _report(entry: dict) -> None:
    if entry["error"] == "skipped_existing":
        print(f"  = skip (exists): {entry['local_path']}")
    elif entry["error"]:
        target = entry["file_url"] or entry["page_url"]
        print(f"  ! failed: {target}  ({entry['error']})")
    else:
        print(f"  + saved: {entry['local_path']}  ({entry['bytes']} bytes)")


def write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    new_file = not manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()
        for e in entries:
            writer.writerow(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--excel", default="research/india_traffic_document_links.xlsx")
    parser.add_argument("--sheet", default="Document Source Links")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--manifest", default="data/raw/manifest.csv")
    parser.add_argument("--statuses", default=",".join(DEFAULT_STATUSES),
                         help="Comma-separated Verification Status values to include")
    parser.add_argument("--category", default=None, help="Only process this Category / Hierarchy Level")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N matching rows")
    parser.add_argument("--delay", type=float, default=0.75, help="Seconds between file downloads")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds")
    parser.add_argument("--insecure-tls", action="store_true",
                         help="Disable TLS certificate verification (needed for a few sites with cert issues)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download and keep existing files")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded, without fetching")
    args = parser.parse_args()

    statuses = [s.strip() for s in args.statuses.split(",")]
    rows = load_rows(Path(args.excel), args.sheet)
    rows = filter_rows(rows, statuses, args.category)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Matched {len(rows)} row(s) with status in {statuses}"
          + (f", category={args.category}" if args.category else "") + "\n")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_entries = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row.get('Source Name')}")
        entries = process_row(
            session, row, Path(args.output), args.timeout, args.insecure_tls,
            args.overwrite, args.delay, args.dry_run,
        )
        all_entries.extend(entries)

    if not args.dry_run and all_entries:
        write_manifest(Path(args.manifest), all_entries)

    downloaded = sum(1 for e in all_entries if not e["error"])
    skipped = sum(1 for e in all_entries if e["error"] == "skipped_existing")
    failed = sum(1 for e in all_entries if e["error"] and e["error"] != "skipped_existing")
    print(f"\nDone. Downloaded: {downloaded}, skipped (already existed): {skipped}, failed: {failed}")
    if all_entries and not args.dry_run:
        print(f"Manifest written to: {args.manifest}")


if __name__ == "__main__":
    sys.exit(main())
