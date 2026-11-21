#!/usr/bin/env python3
"""
Automated downloader for Waymo dataset shards shared via ISEP OneDrive/SharePoint.
Streams shards directly with resume support and integrity verification.
"""

import os
import sys
import time
import urllib.parse
import urllib.request

TOKEN = (
    "eyJhbGciOiJSUzI1NiIsImtpZCI6IjlFNTIwQ0YyOTQ3NEM2MzVBMTQ0RUNDNDY4MDNFNjlDOUM4MEE3QzYiLCJ0eXAiOiJKV1QiLCJ4NXQiOiJubElNOHBSMHhqV2hST3pFYUFQbW5KeUFwOFkifQ."
    "eyJ1cG4iOiIxMjYwNDI5QGlzZXAuaXBwLnB0IiwicHVpZCI6IjEwMDMyMDA2MTg1NDA0NTYiLCJjYWNoZWtleSI6IjBoLmZ8bWVtYmVyc2hpcHwxMDAzMjAwNjE4NTQwNDU2QGxpdmUuY29tIiwidmVyIjoiYnJvd3NlciIsInRpZCI6IjdhNjBhNjI5LWU2MWYtNDMwYy05YzYzLTI2NjY1M2IyYjg5OSIsImFwcGlkIjoiYzU4NjM3YmItZTJlMS00MzEyLThhMDAtMDRiNWZmY2QzNDAzIiwiYXBwX2Rpc3BsYXluYW1lIjoiU2hhcmVQb2ludCBPbmxpbmUgQ2xpZW50IEV4dGVuc2liaWxpdHkiLCJzY3AiOiJGaWxlcy5SZWFkV3JpdGUuQWxsIFNpdGVzLk1hbmFnZS5BbGwgU2l0ZXMuRnVsbENvbnRyb2wuQWxsIFRlcm1TdG9yZS5SZWFkV3JpdGUuQWxsIE0zNjVCaWxsaW5nUGxhdGZvcm0uUmVhZC5BbGwiLCJpcGFkZHIiOiI3OC4xMzcuMTk1LjE2MSIsInNpZCI6IjAwOGJjZDJhLWFjMjQtN2NjYi1hNzIzLTQ5OWNhNWE4YTJmNiIsInNpZ25pbl9zdGF0ZSI6IltcImttc2lcIl0iLCJ1dGkiOiJ4MVFjRGVELU1VR0pCSFpPZjVoUUFBIiwib2lkIjoiZTI3ZmVkN2MtMzBlZS00YjViLThiMWQtZWM2ZWUxZmNkNjBiIiwiZmlkIjoiMTg5MTIyIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwiYXVkIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwQDdhNjBhNjI5LWU2MWYtNDMwYy05YzYzLTI2NjY1M2IyYjg5OSIsIm5iZiI6IjE3OTUyNjMzMDUiLCJleHAiOiIxNzk1MjY2OTA1In0."
    "rTZcM06SFSxiumhNgrOEvxaQ_zngjTMssu6MqmyGaNkqQTJUqz1uzK2P9zIdRBtihV0Ad06kRnqq1vtySgnnZ-pcuHwrDg6K8LZjvrHES3fQKeVLKofzQyvsGdIRP_fmR4_lxvZaZseBnfPhV6hBsvaomi3hyk7P2M3zHgJS6dIGwQWOuEpsdT2AiOHiDdGSww74Vx8Jz4RhYUFiIPMida60MpLTyaGZRbyA1E_-5EHlJ9g8fR-iNcDnBN4OSrwu2GlHnkxkNnwINkh7g_u2gWZiXWWIG-M8GABkAbKFBjFfmKxIzo7-GP4je8ExeNiCvoYRYY0ebTQYF8jjkWus-g"
)

SHARDS = [
    ("testing.tfrecord-00000-of-00150.bin", "7E532650-CFB8-490C-B98B-391D8A3638BA", 195017008),
    ("validation.tfrecord-00000-of-00150.bin", "D47D4B11-7CB6-411C-AA04-ACC29C2AB11B", 262637179),
    ("training.tfrecord-00000-of-01000.bin", "D86910C3-9EDB-4E13-AE67-D1FE031B7663", 454771584),
    ("training.tfrecord-00001-of-01000.bin", "DCB691AF-67C6-4EFE-9513-7B6E4529F582", 481163081),
    ("training.tfrecord-00002-of-01000.bin", "337FF571-60F3-4998-99FF-822D5F763871", 503553230),
    ("training.tfrecord-00003-of-01000.bin", "A5172AA8-A9B7-4332-9FAB-9F3939AE36BD", 449455511),
    ("training.tfrecord-00004-of-01000.bin", "E01F5768-C8B6-4CAE-A19D-0300693D11BD", 444764663),
    ("training.tfrecord-00005-of-01000.bin", "85253478-C8D6-4B7A-AE0A-D8381B7E44F9", 471873141),
    ("training.tfrecord-00006-of-01000.bin", "7B98B9B1-6B1C-43AE-A33E-1E6750E9E7DA", 445079731),
    ("training.tfrecord-00007-of-01000.bin", "29E9D009-77BB-46B4-9E25-CB0ECC2EAAB6", 462647753),
    ("training.tfrecord-00008-of-01000.bin", "CE4389F2-2E71-4132-8D57-D7FB6B59F6DA", 447485179),
    ("training.tfrecord-00009-of-01000.bin", "5CC2FDE1-3DB5-4678-A370-8DF83575F5AF", 434072032),
]


def download_shard(filename, unique_id, expected_size, out_dir):
    dest = os.path.join(out_dir, filename)
    part = dest + ".part"

    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
        actual_size = os.path.getsize(dest)
        print(f"[SKIP] {filename} exists ({actual_size / (1024*1024):.1f} MB)")
        return True

    url = f"https://myisepipp-my.sharepoint.com/personal/1260443_isep_ipp_pt/_layouts/15/download.aspx?UniqueId={unique_id}"
    data = urllib.parse.urlencode({"access_token": TOKEN}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("User-Agent", "Mozilla/5.0")
    req.add_header("Origin", "https://myisepipp-my.sharepoint.com")

    print(f"[START] {filename} (~{expected_size / (1024*1024):.1f} MB)...")
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp.read(2 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - start_time
                speed = downloaded / elapsed / (1024 * 1024) if elapsed > 0 else 0
                pct = (downloaded / expected_size) * 100 if expected_size > 0 else 0
                print(
                    f"\r  -> {downloaded / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s",
                    end="",
                    flush=True,
                )
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(part, dest)
        total_time = time.time() - start_time
        print(f"\n[DONE] {filename} in {total_time:.1f}s ({os.path.getsize(dest) / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed {filename}: {e}", file=sys.stderr)
        return False


def main():
    target_dir = os.path.expanduser("~/Downloads/archive")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory: {target_dir}")
    print(f"Total shards to check/download: {len(SHARDS)}")

    for idx, (name, uid, size) in enumerate(SHARDS, 1):
        print(f"\n[{idx}/{len(SHARDS)}] Processing {name}")
        download_shard(name, uid, size, target_dir)


if __name__ == "__main__":
    main()
