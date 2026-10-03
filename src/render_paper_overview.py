#!/usr/bin/env python3
"""Render the paper overview HTML to a vector PDF and preview PNG."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


WIDTH = 1800
HEIGHT = 800
RASTER_SCALE = 2


def find_chromium(explicit: str | None) -> Path | None:
    candidates = [
        explicit,
        os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"),
        "/tmp/gameworld-phase2/ms-playwright/chromium-1124/chrome-linux/chrome",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def render_with_firefox(html: Path, png_path: Path, pdf_path: Path) -> None:
    firefox = shutil.which("firefox")
    if firefox is None:
        raise FileNotFoundError("Neither Playwright Chromium nor Firefox is available.")

    subprocess.run(
        [
            firefox,
            "--headless",
            "--screenshot",
            str(png_path),
            "--window-size",
            f"{WIDTH * RASTER_SCALE},{HEIGHT * RASTER_SCALE}",
            html.as_uri(),
        ],
        check=True,
    )

    from PIL import Image

    with Image.open(png_path) as image:
        expected_size = (WIDTH * RASTER_SCALE, HEIGHT * RASTER_SCALE)
        rendered = image.crop((0, 0, *expected_size)) if image.size != expected_size else image
        rendered.save(png_path)
        rendered.convert("RGB").save(pdf_path, "PDF", resolution=600.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("paper/full_draft_assets/overview_memory_selection.html"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/full_draft_assets/figures"),
    )
    parser.add_argument("--chromium", type=str)
    args = parser.parse_args()

    html = args.html.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chromium = find_chromium(args.chromium)

    pdf_path = output_dir / "F1_memory_selection_overview.pdf"
    png_path = output_dir / "F1_memory_selection_overview.png"

    if sync_playwright is None or chromium is None:
        render_with_firefox(html, png_path, pdf_path)
    else:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(chromium),
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page(
                viewport={"width": WIDTH, "height": HEIGHT},
                device_scale_factor=2,
            )
            page.goto(html.as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            page.screenshot(path=str(png_path), full_page=True)
            page.pdf(
                path=str(pdf_path),
                width=f"{WIDTH}px",
                height=f"{HEIGHT}px",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            browser.close()

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
