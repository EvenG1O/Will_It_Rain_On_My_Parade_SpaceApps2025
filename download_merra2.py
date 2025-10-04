#!/usr/bin/env python3
import requests
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import sys

# Config
DOWNLOAD_DIR = Path("MERRA2_daily")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

S3_LIST_FILE = Path("merra2_s3_paths.txt")
if not S3_LIST_FILE.exists():
    print(f"Missing {S3_LIST_FILE}. Put one s3:// path per line.")
    sys.exit(1)

# Authentication: either .netrc (recommended) OR environment variables EARTHDATA_USER/EARTHDATA_PASS
USERNAME = os.environ.get("EARTHDATA_USER", "")
PASSWORD = os.environ.get("EARTHDATA_PASS", "")

# If both blank, requests will still use ~/.netrc when using requests.Session() w/ auth=None
USE_AUTH = bool(USERNAME and PASSWORD)

# Tuning
MAX_WORKERS = 3       # adjust for parallel downloads 
RETRIES = 5
CHUNK_SIZE = 1024 * 1024  # 1 MB
INTER_FILE_DELAY = 3   # seconds between starting files 

session = requests.Session()
# If you want to force .netrc usage when username/password blank, requests will check netrc automatically
auth = None
if USE_AUTH:
    auth = requests.auth.HTTPBasicAuth(USERNAME, PASSWORD)

def s3_to_https(s3_path: str) -> str:
    return s3_path.replace(
        "s3://gesdisc-cumulus-prod-protected/",
        "https://data.gesdisc.earthdata.nasa.gov/data/"
    )

def download_single(url: str, outpath: Path) -> bool:
    # Skip existing
    if outpath.exists():
        return True

    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, auth=auth, stream=True, timeout=60) as r:
                # If server responds with redirect to login, r.status_code might be 401/302
                if r.status_code == 200:
                    total = int(r.headers.get("content-length", 0))
                    with open(outpath, "wb") as f, tqdm(
                        total=total, unit="B", unit_scale=True, desc=outpath.name, leave=False
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                    return True
                else:
                    print(f"[{outpath.name}] Attempt {attempt}: HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[{outpath.name}] Attempt {attempt}: {e}")
        # Backoff
        sleep_time = 5 * attempt
        print(f"[{outpath.name}] Waiting {sleep_time}s before retry...")
        time.sleep(sleep_time)


    print(f"[{outpath.name}] Failed after {RETRIES} attempts")
    return False

def main():
    s3_paths = [line.strip() for line in S3_LIST_FILE.read_text().splitlines() if line.strip()]
    # Build list of (url, outpath)
    tasks = []
    for s3 in s3_paths:
        url = s3_to_https(s3)
        filename = DOWNLOAD_DIR / url.split("/")[-1]
        tasks.append((url, filename))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_task = {}
        for url, outpath in tasks:
            # skip already present files to avoid wasted threads
            if outpath.exists():
                print(f"Skipping {outpath.name} (already exists)")
                continue
            # stagger start a bit
            time.sleep(INTER_FILE_DELAY)
            future = ex.submit(download_single, url, outpath)
            future_to_task[future] = outpath

        # collect results
        for fut in as_completed(future_to_task):
            outpath = future_to_task[fut]
            try:
                ok = fut.result()
                if ok:
                    print(f"Done: {outpath.name}")
                else:
                    print(f"Failed: {outpath.name}")
            except Exception as e:
                print(f"Exception for {outpath.name}: {e}")

if __name__ == "__main__":
    main()
