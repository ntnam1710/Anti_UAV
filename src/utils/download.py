# -*- coding: utf-8 -*-
"""
Tiện ích hỗ trợ tải dữ liệu hình ảnh và video từ đường dẫn URL từ xa.
"""
import os
import urllib.request
import tempfile
import mimetypes

def download_file_from_url(url, target_dir=None):
    """
    Tải tệp hình ảnh hoặc video từ URL (HTTP/HTTPS) về thư mục cục bộ.
    Nếu target_dir là None, tệp sẽ được lưu vào thư mục tạm thời 'anti_uav_downloads'.
    
    Tham số:
        url (str): Đường dẫn URL của tệp đa phương tiện.
        target_dir (str, optional): Thư mục đích lưu tệp.
        
    Trả về:
        str: Đường dẫn tuyệt đối đến tệp đã tải về.
    """
    if target_dir is None:
        target_dir = os.path.join(tempfile.gettempdir(), "anti_uav_downloads")
    os.makedirs(target_dir, exist_ok=True)

    # Trích xuất tên tệp hợp lệ từ URL
    base_name = os.path.basename(url.split("?")[0].split("#")[0])
    if not base_name or "." not in base_name:
        base_name = "downloaded_media.mp4" if "video" in url.lower() else "downloaded_media.jpg"

    dest_path = os.path.join(target_dir, base_name)
    print(f"[INFO] Dang tai du lieu tu URL: {url}")
    print(f"[INFO] Thu muc luu: {dest_path}")

    # Thiết lập User-Agent để tránh lỗi 403 Forbidden từ các máy chủ CDN
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

    file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[HOAN TAT] Da tai tep thanh cong: {dest_path} ({file_size_mb:.2f} MB)")
    return dest_path
