"""
Search NASA OSDR (Open Science Data Repository) and download study files.
No API key needed. Endpoints (from NASA's OSDR Developer API docs):
  search : https://osdr.nasa.gov/osdr/data/search?term=<keywords>&from=0&size=10
  files  : https://osdr.nasa.gov/osdr/data/osd/files/<number>      e.g. 87
  files' remote_url is relative to https://osdr.nasa.gov

Usage:
  python nasa_osdr_fetch.py search "astronaut exercise"
  python nasa_osdr_fetch.py files 87
  python nasa_osdr_fetch.py download 87 --match csv --out data/nasa/
Run it on your own machine (needs internet). If the JSON layout differs, use `--raw` to inspect it.
pip install requests
"""
import argparse, json, os, re, sys
import requests

BASE = "https://osdr.nasa.gov"
SEARCH, FILES = f"{BASE}/osdr/data/search", f"{BASE}/osdr/data/osd/files"

def get(url, **params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def walk(o, key):
    """Yield every value stored under `key` anywhere in a nested JSON object (layout-tolerant)."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: yield v
            yield from walk(v, key)
    elif isinstance(o, list):
        for v in o: yield from walk(v, key)

def search(term, size=10, raw=False):
    data = get(SEARCH, term=term, **{"from": 0, "size": size})
    if raw: print(json.dumps(data, indent=1)[:6000]); return
    hits = data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []
    for h in hits:
        src = h.get("_source", h)
        acc = src.get("Accession") or src.get("accession") or h.get("_id")
        title = src.get("Study Title") or src.get("study title") or src.get("title") or ""
        print(f"{acc}\t{str(title)[:110]}")
    if not hits: print("No hits parsed. Re-run with --raw to inspect the response.")

def list_files(study, raw=False):
    num = re.sub(r"\D", "", study)                               # "OSD-87" -> "87"
    data = get(f"{FILES}/{num}")
    if raw: print(json.dumps(data, indent=1)[:6000]); return []
    files = [f for lst in walk(data, "study_files") for f in lst]
    for f in files:
        print(f"{f.get('file_name')}\t{f.get('file_size')} bytes")
    return files

def download(study, match, out):
    files = [f for f in list_files(study) if re.search(match, f.get("file_name", ""), re.I)]
    os.makedirs(out, exist_ok=True)
    for f in files:
        url = f["remote_url"] if f["remote_url"].startswith("http") else BASE + f["remote_url"]
        dest = os.path.join(out, f["file_name"])
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 16): fh.write(chunk)
        print("saved", dest)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "files", "download"])
    ap.add_argument("arg")
    ap.add_argument("--match", default=r"\.(csv|txt|zip)$")
    ap.add_argument("--out", default="data/nasa")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    if a.cmd == "search": search(a.arg, raw=a.raw)
    elif a.cmd == "files": list_files(a.arg, raw=a.raw)
    else: download(a.arg, a.match, a.out)
