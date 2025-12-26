"""Test utilities for downloading and working with audio test files."""

import os
import urllib.request


def download_if_not_exists(filename: str, url: str) -> str:
    """
    Download a file to the local tmp directory if it doesn't already exist.
    """
    local_tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(local_tmp_dir, exist_ok=True)
    dest_path = os.path.join(local_tmp_dir, filename)

    if not os.path.exists(dest_path):
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}") from e

    assert os.path.exists(dest_path), f"{filename} not found"
    return dest_path


def download_short_audio() -> str:
    """Download short test audio file to tmp directory if not exists."""
    return download_if_not_exists("example.wav", "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav")


def download_long_audio() -> str:
    """Download long test audio file to tmp directory if not exists."""
    return download_if_not_exists(
        "long_example.wav", "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/long_example.wav"
    )
