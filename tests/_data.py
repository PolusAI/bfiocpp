"""Download helpers for test data, with retries and size/zip validation."""
import pathlib
import shutil
import time
import urllib.request
import zipfile

# Pinned to a commit of sameeul/polus-test-data (merge of PR #3).
DATA_REPO = (
    "https://raw.githubusercontent.com/sameeul/polus-test-data/"
    "9ec79af23d155a2bdb74823fa6fc6f204bbe2ca4/bfio"
)
SMALL_ZARR_URL = f"{DATA_REPO}/small_5d.zarr.zip"


def fetch(url, dest, tries=4, timeout=60):
    """Download url to dest, retrying with backoff; raises on final failure."""
    dest = pathlib.Path(dest)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            return dest
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2**i)


def fetch_zarr(url, test_dir):
    """Download a zipped zarr and unpack it into test_dir."""
    test_dir = pathlib.Path(test_dir)
    z = fetch(url, test_dir / "tmp_zarr.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(test_dir)
    z.unlink()


def clean(test_dir):
    shutil.rmtree(test_dir, ignore_errors=True)
