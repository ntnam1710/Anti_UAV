# -*- coding: utf-8 -*-
"""
Helper utility for downloading remote images and videos via URL
"""
import os
import urllib.request
import tempfile
import mimetypes

def download_file_from_url(url, target_dir=None):
    """
    Downloads an image or video file from an HTTP/HTTPS URL to a local destination directory.
    If target_dir is None, downloads to a dedicated 'temp_downloads' folder.
    Returns the absolute path to the downloaded file.
    """
    if target_dir is None:
        target_dir = os.path.join(tempfile.gettempdir(), "anti_uav_downloads")
    os.makedirs(target_dir, exist_ok=True)

    # Clean filename from URL
    base_name = os.path.basename(url.split("?")[0].split("#")[0])
    if not base_name or "." not in base_name:
        base_name = "downloaded_media.mp4" if "video" in url.lower() else "downloaded_media.jpg"

    dest_path = os.path.join(target_dir, base_name)
    print(f">> Downloading media from: {url}")
    print(f">> Destination: {dest_path}")

    # Set custom User-Agent headers to avoid 403 Forbidden from media CDNs
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)

    print(f"✅ Download complete: {dest_path} ({os.path.getsize(dest_path) / (1024 * 1024):.2f} MB)")
    return dest_path
